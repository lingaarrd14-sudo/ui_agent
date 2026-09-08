"""VisualWebArena 어댑터를 실제 브라우저나 API 없이 검증한다."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ui_agent import vwa_config
from ui_agent.models import ClickAction, Decision, DoneAction, ExecutionResult
from ui_agent.vwa_adapter import VisualWebArenaAdapter, decision_to_vwa_action
from ui_agent.vwa_runtime import BrowserEnvActionFactory
from ui_agent.vwa_results import persist_run_config, read_latest_results
from ui_agent.vwa_tasks import run_selected_tasks


class FakeActionFactory:
    def click(self, x, y):
        return {"kind": "click", "coords": [x, y]}

    def type_text(self, text):
        return {"kind": "type", "text": text}

    def press(self, key):
        return {"kind": "press", "key": key}

    def scroll(self, direction):
        return {"kind": "scroll", "direction": direction}

    def stop(self, answer):
        return {"kind": "stop", "answer": answer}


class FakeVWAActionTypes:
    MOUSE_CLICK = "mouse_click"


class FakeVWABindings:
    action_types = FakeVWAActionTypes()

    @staticmethod
    def create_none_action():
        return {}


class FakeTaskBindings:
    def __init__(self):
        self.trajectory = None

    @staticmethod
    def create_stop_action(answer):
        return {"kind": "stop", "answer": answer}

    def evaluator_router(self, runtime_config, captioning_fn=None):
        def evaluate(trajectory, evaluated_config, page):
            self.trajectory = trajectory
            return 1.0

        return evaluate


class FakePage:
    def __init__(self, url="https://example.com"):
        self.url = url


class FakeTaskEnvironment:
    def __init__(self):
        self.page = FakePage()

    def reset(self, options):
        current = state()
        return current["observation"], current["info"]


def state(color=255, url="https://example.com", failure=""):
    return {
        "observation": {"image": np.full((20, 30, 3), color, dtype=np.uint8)},
        "info": {"page": FakePage(url), "fail_error": failure},
    }


class OneDecisionPolicy:
    def __init__(self, decision):
        self.decision = decision

    def decide(
        self, task, observation, history, reference_images=(), feedback=()
    ):
        return self.decision


class VisualWebArenaPolicyTests(unittest.TestCase):
    def test_generated_configs_exclude_tasks_with_viewport_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "config_files" / "vwa"
            source_dir.mkdir(parents=True)
            source = source_dir / "tasks.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "task_id": 1,
                            "intent": "exclude",
                            "start_url": "__SHOPPING__/",
                            "viewport_size": {"width": 430, "height": 932},
                            "eval": {"eval_types": []},
                        },
                        {
                            "task_id": 2,
                            "intent": "keep",
                            "start_url": "__SHOPPING__/",
                            "eval": {"eval_types": []},
                        },
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch.object(vwa_config, "VWA_ROOT", root),
                patch.dict(vwa_config.DOMAIN_SOURCES, {"shopping": "tasks.json"}),
                patch.object(
                    vwa_config, "site_urls", return_value=vwa_config.SITE_DEFAULTS
                ),
            ):
                generated = vwa_config.generate_configs(
                    root / "results", ["shopping"]
                )

            self.assertEqual([path.name for path in generated["shopping"]], ["2.json"])
            self.assertEqual(json.loads(generated["shopping"][0].read_text())["task_id"], 2)
            self.assertIn("viewport_size", json.loads(source.read_text())[0])

    def test_decision_schema_accepts_benchmark_answer(self):
        decision = Decision(
            action=DoneAction(kind="done", status="success", summary="blue kayak"),
            expected_outcome="Return the requested answer",
        )
        self.assertEqual(decision.action.summary, "blue kayak")

    def test_click_can_target_viewport_origin(self):
        decision = Decision(
            action=ClickAction(kind="click", x=0, y=0),
            expected_outcome="Click the corner",
        )
        self.assertEqual((decision.action.x, decision.action.y), (0, 0))

        action = decision_to_vwa_action(decision, FakeActionFactory())

        self.assertEqual(action["coords"], [0, 0])
        self.assertIn('"kind":"click"', action["raw_prediction"])

    def test_vwa_runtime_normalizes_pixel_click_coordinates(self):
        factory = BrowserEnvActionFactory(
            FakeVWABindings(), viewport_width=1280, viewport_height=2048
        )

        action = factory.click(640, 512)

        self.assertEqual(action["action_type"], "mouse_click")
        self.assertAlmostEqual(float(action["coords"][0]), 0.5)
        self.assertAlmostEqual(float(action["coords"][1]), 0.25)

    def test_adapter_uses_raw_image_and_records_execution(self):
        decision = Decision(
            action=ClickAction(kind="click", x=10, y=10),
            expected_outcome="Change the page",
        )
        adapter = VisualWebArenaAdapter(
            policy=OneDecisionPolicy(decision),
            action_factory=FakeActionFactory(),
            viewport_width=30,
            viewport_height=20,
        )

        proposed, before, action = adapter.decide("goal", state())
        record = adapter.record(
            before,
            proposed,
            ExecutionResult(ok=True),
            state(color=0, url="https://example.com/next"),
        )

        self.assertEqual(action["kind"], "click")
        self.assertEqual(before.viewport_width, 30)
        self.assertTrue(record.result.state_changed)
        self.assertEqual(record.result.after_url, "https://example.com/next")

    def test_result_directory_rejects_incompatible_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            persist_run_config(result_dir, {"viewport": [1280, 720]})

            with self.assertRaisesRegex(RuntimeError, "viewport"):
                persist_run_config(result_dir, {"viewport": [640, 720]})

    def test_latest_result_wins_after_error_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            records = [
                {"domain": "reddit", "task_id": 0, "score": None},
                {"domain": "reddit", "task_id": 0, "score": 1.0},
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            latest = read_latest_results(path)

            self.assertEqual(latest[("reddit", 0)]["score"], 1.0)

    def test_task_runner_persists_official_score_for_done_action(self):
        decision = Decision(
            action=DoneAction(kind="done", status="success", summary="answer"),
            expected_outcome="Return the answer",
        )
        options = SimpleNamespace(
            max_steps=2,
            max_proposals=3,
            viewport_width=30,
            viewport_height=20,
            save_traces=False,
        )
        bindings = FakeTaskBindings()

        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            config_path = result_dir / "task.json"
            config_path.write_text(
                json.dumps(
                    {
                        "task_id": 7,
                        "intent": "return answer",
                        "eval": {"eval_types": []},
                    }
                ),
                encoding="utf-8",
            )

            run_selected_tasks(
                options=options,
                bindings=bindings,
                policy=OneDecisionPolicy(decision),
                action_factory=FakeActionFactory(),
                captioner=object(),
                env=FakeTaskEnvironment(),
                result_dir=result_dir,
                selected=[("reddit", config_path)],
            )

            latest = read_latest_results(result_dir / "results.jsonl")
            self.assertEqual(latest[("reddit", 7)]["score"], 1.0)
            self.assertEqual(bindings.trajectory[-1]["kind"], "stop")


if __name__ == "__main__":
    unittest.main()
