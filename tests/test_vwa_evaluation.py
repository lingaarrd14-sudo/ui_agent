"""로컬 VWA 평가기를 직접 검증한다. 브라우저·모델 API·모델 다운로드는 사용하지 않는다."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from ui_agent.models import Decision, StopAction
from ui_agent.vwa_adapter import decision_to_vwa_action
from ui_agent.vwa_config import (
    SHOPPING_SOURCE,
    SITE_DEFAULTS,
    VWA_ROOT,
    configure_environment,
    generate_configs,
)
from ui_agent.vwa_runtime import BrowserEnvActionFactory, load_vwa_bindings
from ui_agent.vwa_tasks import _score_task


class OfficialEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", **SITE_DEFAULTS}), \
                patch("ui_agent.vwa_config.load_local_environment"):
            configure_environment(require_api_key=True)
            cls.bindings = load_vwa_bindings(VWA_ROOT)
        from evaluation_harness import evaluators, helper_functions

        cls.evaluators = evaluators
        cls.helpers = helper_functions

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / "task.json"
        self.factory = BrowserEnvActionFactory(self.bindings, 30, 20)
        self.env = SimpleNamespace(page=self.helpers.PseudoPage(None, "http://site.test/item"))
        self.captioner = Mock()
        self.captioner.get.return_value = Mock()

    def score(self, evaluation, answer="answer", status="success"):
        task = {"intent": "Test goal", "eval": evaluation}
        self.config.write_text(json.dumps(task), encoding="utf-8")
        action = decision_to_vwa_action(Decision(
            state_summary="Evaluation response is ready",
            action=StopAction(kind="stop", status=status, answer=answer),
            expected_outcome="Stop",
        ), self.factory)
        return _score_task(
            bindings=self.bindings, captioner=self.captioner, env=self.env,
            runtime_config=self.config,
            trajectory=[{"observation": {}, "info": {}}, action],
        )

    def test_bindings_use_local_original_evaluator(self):
        self.assertIs(self.bindings.evaluator_router, self.evaluators.evaluator_router)
        self.assertEqual(Path(self.evaluators.__file__).resolve(),
                         VWA_ROOT / "evaluation_harness" / "evaluators.py")

    def test_generated_unachievable_tasks_keep_references_and_accept_na(self):
        with patch("ui_agent.vwa_config.site_urls", return_value=SITE_DEFAULTS):
            configs = generate_configs(Path(self.temp.name))
        checked = 0
        raw = json.loads(
            (VWA_ROOT / "config_files" / "vwa" / SHOPPING_SOURCE).read_text()
        )
        originals = {task["task_id"]: task for task in raw}
        for path in configs:
            task = json.loads(path.read_text())
            refs = task["eval"].get("reference_answers") or {}
            if refs.get("fuzzy_match") != "N/A":
                continue
            with self.subTest(task_id=task["task_id"]):
                original = originals[task["task_id"]]["eval"]
                self.assertEqual(refs, original["reference_answers"])
                self.assertEqual(task["eval"]["string_note"], original["string_note"])
                with patch.object(
                    self.helpers,
                    "generate_from_openai_chat_completion",
                    side_effect=AssertionError("N/A must not call a judge"),
                ):
                    self.assertEqual(self.score(task["eval"], "N/A"), 1.0)
                checked += 1
        self.assertGreater(checked, 0)

    def test_unachievable_reason_uses_original_judge_fallback(self):
        evaluation = {"eval_types": ["string_match"],
                      "reference_answers": {"fuzzy_match": "N/A"},
                      "string_note": "The item is not sold here."}
        for reply, expected in (("same", 1.0), ("different", 0.0)):
            with self.subTest(reply=reply), patch.object(
                self.helpers, "generate_from_openai_chat_completion", return_value=reply
            ) as judge:
                self.assertEqual(self.score(evaluation, "No matching item exists."), expected)
                prompt = judge.call_args.kwargs["messages"][-1]["content"]
                self.assertIn(evaluation["string_note"], prompt)
                self.assertIn("no matching item exists.", prompt)

    def test_fuzzy_answer_uses_original_judge(self):
        evaluation = {"eval_types": ["string_match"],
                      "reference_answers": {"fuzzy_match": ["blue kayak"]}}
        for reply, expected in (("correct", 1.0), ("incorrect", 0.0), ("partially correct", 0.0)):
            with self.subTest(reply=reply), patch.object(
                self.helpers, "generate_from_openai_chat_completion", return_value=reply
            ):
                self.assertEqual(self.score(evaluation, "a blue boat"), expected)

    def test_agent_status_does_not_override_score(self):
        evaluation = {"eval_types": ["string_match"],
                      "reference_answers": {"exact_match": "answer"}}
        self.assertEqual(self.score(evaluation, "wrong", "success"), 0.0)
        self.assertEqual(self.score(evaluation, "answer", "blocked"), 1.0)

    def test_url_html_combination_requires_both(self):
        evaluation = {
            "eval_types": ["url_match", "program_html"],
            "reference_url": "http://site.test/item",
            "program_html": [{"url": "last", "locator": "",
                              "required_contents": {"exact_match": "expected content"}}],
        }
        for url, content, expected in (
            ("http://site.test/item", "expected content", 1.0),
            ("http://site.test/wrong", "expected content", 0.0),
            ("http://site.test/item", "wrong content", 0.0),
        ):
            with self.subTest(url=url, content=content):
                self.env.page = self.helpers.PseudoPage(SimpleNamespace(content=lambda: content), url)
                self.assertEqual(self.score(evaluation), expected)

    def test_image_ssim_uses_original_captioner_binding(self):
        pixels = Image.new("RGB", (16, 16), "white")
        reference = Path(self.temp.name) / "reference.png"
        pixels.save(reference)
        buffer = io.BytesIO()
        pixels.save(buffer, format="PNG")
        image_element = SimpleNamespace(get_attribute=lambda name: "/image.png")
        page = SimpleNamespace(get_by_role=lambda role: SimpleNamespace(all=lambda: [image_element]))
        self.env.page = self.helpers.PseudoPage(page, "http://site.test/item")
        evaluation = {"eval_types": ["page_image_query"], "page_image_query": [{
            "eval_image_class": "", "eval_image_url": "last",
            "eval_fuzzy_image_match": str(reference),
        }]}
        with patch.object(self.evaluators.requests, "get", return_value=SimpleNamespace(
            raw=io.BytesIO(buffer.getvalue())
        )):
            self.assertEqual(self.score(evaluation), 1.0)
        self.captioner.get.assert_called_once()

    def test_image_vqa_receives_captioner_and_scores_answer(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), "blue").save(buffer, format="PNG")
        page = SimpleNamespace(get_by_role=lambda role: SimpleNamespace(all=lambda: [
            SimpleNamespace(get_attribute=lambda name: "/image.png")
        ]))
        self.env.page = self.helpers.PseudoPage(page, "http://site.test/item")
        evaluation = {"eval_types": ["page_image_query"], "page_image_query": [{
            "eval_image_class": "", "eval_image_url": "last",
            "eval_vqa": [{"question": "What color?", "answer": "blue"}],
        }]}
        caption_fn = Mock(return_value=["blue"])
        self.captioner.get.side_effect = None
        self.captioner.get.return_value = caption_fn
        with patch.object(self.evaluators.requests, "get", return_value=SimpleNamespace(
            raw=io.BytesIO(buffer.getvalue())
        )):
            self.assertEqual(self.score(evaluation), 1.0)
        self.assertEqual(caption_fn.call_args.args[1], ["Q: What color? A:"])

if __name__ == "__main__":
    unittest.main()
