"""VisualWebArena 어댑터를 실제 브라우저나 API 없이 검증한다."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ui_agent import vwa_config
from ui_agent.models import (
    ClickAction,
    Decision,
    ExecutionResult,
    GoBackAction,
    GoForwardAction,
    HoverAction,
    PressAction,
    ScrollAction,
    StopAction,
    TypeAction,
)
from ui_agent.vwa_adapter import VisualWebArenaAdapter, decision_to_vwa_action
from ui_agent.vwa_runtime import BrowserEnvActionFactory
from ui_agent.vwa_results import persist_run_config, read_latest_results
from ui_agent.vwa_tasks import _run_agent_steps, run_selected_tasks
from vwa_benchmark import VWA_SLEEP_AFTER_EXECUTION, parse_args


class FakeActionFactory:
    def click(self, x, y):
        return {"kind": "click", "coords": [x, y]}

    def hover(self, x, y):
        return {"kind": "hover", "coords": [x, y]}

    def type_text(self, text):
        return {"kind": "type", "text": text}

    def press(self, key):
        return {"kind": "press", "key": key}

    def scroll(self, direction):
        return {"kind": "scroll", "direction": direction}

    def go_back(self):
        return {"kind": "go_back"}

    def go_forward(self):
        return {"kind": "go_forward"}

    def stop(self, answer):
        return {"kind": "stop", "answer": answer}


class FakeVWAActionTypes:
    MOUSE_CLICK = "mouse_click"
    MOUSE_HOVER = "mouse_hover"


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
            page.url = "https://example.com/evaluation-target"
            return 1.0

        return evaluate


class FakeCaptioner:
    def get(self):
        return None


class FakePage:
    def __init__(self, url="https://example.com"):
        self.url = url


class FakeDetachedPage:
    """VWA가 observation info에 넣는 context 없는 page snapshot."""

    def __init__(self, url="https://example.com"):
        self.url = url


class FakeTaskEnvironment:
    def __init__(self):
        self.page = FakePage()
        self.actions = []

    def reset(self, options):
        current = state()
        return current["observation"], current["info"]

    def step(self, action):
        self.actions.append(action)
        current = state()
        return current["observation"], 0, False, False, current["info"]


class FakeTerminatingEnvironment(FakeTaskEnvironment):
    def step(self, action):
        self.actions.append(action)
        current = state()
        return current["observation"], 0, True, False, current["info"]


def state(color=255, url="https://example.com", failure=""):
    return {
        "observation": {"image": np.full((20, 30, 3), color, dtype=np.uint8)},
        "info": {"page": FakeDetachedPage(url), "fail_error": failure},
    }


class OneDecisionPolicy:
    def __init__(self, decision):
        self.decision = decision

    def decide(
        self,
        task,
        observation,
        history,
        reference_images=(),
        previous_observation=None,
    ):
        return self.decision


class VisualWebArenaPolicyTests(unittest.TestCase):
    def test_benchmark_defaults_match_vwa_evaluation_timing_and_cpu_captioner(self):
        with patch.object(sys, "argv", ["vwa_benchmark.py"]):
            args = parse_args()

        self.assertEqual(VWA_SLEEP_AFTER_EXECUTION, 2.5)
        self.assertEqual(args.eval_caption_device, "cpu")

    def test_generated_configs_exclude_viewport_and_multi_tab_tasks(self):
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
                        {
                            "task_id": 3,
                            "intent": "exclude multi-tab",
                            "start_url": "__SHOPPING__/a |AND| __SHOPPING__/b",
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

    def test_task_selection_excludes_ids_before_batching(self):
        paths = [Path(f"{task_id}.json") for task_id in range(5)]

        selected = vwa_config.task_selection(
            {"shopping": paths}, start=1, end=3, excluded_task_ids={1, 3}
        )

        self.assertEqual([path.name for _, path in selected], ["2.json", "4.json"])

    def test_decision_schema_accepts_benchmark_answer(self):
        decision = Decision(
            state_summary="Answer is visible",
            action=StopAction(kind="stop", status="success", answer="blue kayak"),
            expected_outcome="Return the requested answer",
        )
        self.assertEqual(decision.action.answer, "blue kayak")

    def test_click_can_target_viewport_origin(self):
        decision = Decision(
            state_summary="Target is visible",
            action=ClickAction(kind="click", x=0, y=0),
            expected_outcome="Click the corner",
        )
        self.assertEqual((decision.action.x, decision.action.y), (0, 0))

        action = decision_to_vwa_action(decision, FakeActionFactory())

        self.assertEqual(action["coords"], [0, 0])
        self.assertIn('"kind":"click"', action["raw_prediction"])

    def test_vwa_action_space_maps_page_and_low_level_actions(self):
        cases = [
            (HoverAction(kind="hover", x=4, y=5), "hover"),
            (TypeAction(kind="type", text="query"), "type"),
            (PressAction(kind="press", key_comb="Enter"), "press"),
            (ScrollAction(kind="scroll", direction="down"), "scroll"),
            (GoBackAction(kind="go_back"), "go_back"),
            (GoForwardAction(kind="go_forward"), "go_forward"),
        ]
        factory = FakeActionFactory()

        for selected, expected_kind in cases:
            with self.subTest(action=selected):
                decision = Decision(
                    state_summary="Control is visible",
                    action=selected,
                    expected_outcome="change page",
                )
                converted = decision_to_vwa_action(decision, factory)
                self.assertEqual(converted["kind"], expected_kind)
                self.assertIn(f'"kind":"{expected_kind}"', converted["raw_prediction"])

    def test_vwa_runtime_normalizes_pixel_click_coordinates(self):
        factory = BrowserEnvActionFactory(
            FakeVWABindings(), viewport_width=1280, viewport_height=2048
        )

        action = factory.click(640, 512)

        self.assertEqual(action["action_type"], "mouse_click")
        self.assertAlmostEqual(float(action["coords"][0]), 0.5)
        self.assertAlmostEqual(float(action["coords"][1]), 0.25)

    def test_vwa_runtime_hover_coordinates_are_python_float_compatible(self):
        factory = BrowserEnvActionFactory(
            FakeVWABindings(), viewport_width=1280, viewport_height=720
        )

        action = factory.hover(640, 360)

        self.assertIsInstance(action["coords"][0], float)
        self.assertIsInstance(action["coords"][1], float)

    def test_adapter_uses_raw_image_and_records_execution(self):
        decision = Decision(
            state_summary="Target is visible",
            action=ClickAction(kind="click", x=10, y=10),
            expected_outcome="Change the page",
        )
        adapter = VisualWebArenaAdapter(
            policy=OneDecisionPolicy(decision),
            action_factory=FakeActionFactory(),
            viewport_width=30,
            viewport_height=20,
        )

        proposed, action = adapter.decide("goal", state())
        record = adapter.record(proposed, ExecutionResult(ok=True))

        self.assertEqual(action["kind"], "click")
        self.assertTrue(record.result.ok)

    def test_adapter_ignores_url_from_detached_page_snapshot(self):
        decision = Decision(
            state_summary="Task is complete",
            action=StopAction(kind="stop", status="success", answer="done"),
            expected_outcome="Stop",
        )
        adapter = VisualWebArenaAdapter(
            policy=OneDecisionPolicy(decision),
            action_factory=FakeActionFactory(),
            viewport_width=30,
            viewport_height=20,
        )

        observation = adapter.observe(state(url="https://secret.example/path"))

        self.assertNotIn("url", observation.model_dump())
        self.assertNotIn("open_tabs", observation.model_dump())

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

    def test_task_runner_persists_official_score_for_stop_action(self):
        decision = Decision(
            state_summary="Answer is ready",
            action=StopAction(kind="stop", status="success", answer="answer"),
            expected_outcome="Return the answer",
        )
        options = SimpleNamespace(
            max_steps=2,
            repeating_action_failure_th=5,
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
                captioner=FakeCaptioner(),
                env=FakeTaskEnvironment(),
                result_dir=result_dir,
                selected=[("reddit", config_path)],
            )

            latest = read_latest_results(result_dir / "results.jsonl")
            self.assertEqual(latest[("reddit", 7)]["score"], 1.0)
            self.assertEqual(latest[("reddit", 7)]["steps"], 0)
            self.assertEqual(latest[("reddit", 7)]["final_url"], "https://example.com")
            self.assertEqual(bindings.trajectory[-1]["kind"], "stop")

    def test_unachievable_na_stops_without_execution(self):
        decision = Decision(
            state_summary="No matching item exists",
            action=StopAction(kind="stop", status="success", answer="N/A"),
            expected_outcome="Report that the task is unachievable",
        )
        options = SimpleNamespace(
            max_steps=5,
            repeating_action_failure_th=5,
            viewport_width=30,
            viewport_height=20,
        )
        bindings = FakeTaskBindings()
        env = FakeTaskEnvironment()
        runtime_config = Path("unachievable.json")

        trajectory, records, calls = _run_agent_steps(
            options=options,
            bindings=bindings,
            policy=OneDecisionPolicy(decision),
            action_factory=FakeActionFactory(),
            env=env,
            task={"intent": "navigate to an item that does not exist"},
            runtime_config=runtime_config,
            reference_images=[],
        )
        self.assertEqual(calls, 1)
        self.assertEqual(env.actions, [])
        self.assertFalse(records[-1]["executed"])
        self.assertEqual(trajectory[-1]["answer"], "N/A")

    def test_environment_termination_uses_original_empty_stop_placeholder(self):
        decision = Decision(
            state_summary="Target is visible",
            action=ClickAction(kind="click", x=10, y=10),
            expected_outcome="Finish navigation",
        )
        env = FakeTerminatingEnvironment()

        trajectory, records, calls = _run_agent_steps(
            options=SimpleNamespace(
                max_steps=5,
                repeating_action_failure_th=5,
                viewport_width=30,
                viewport_height=20,
            ),
            bindings=FakeTaskBindings(),
            policy=OneDecisionPolicy(decision),
            action_factory=FakeActionFactory(),
            env=env,
            task={"intent": "goal"},
            runtime_config=Path("unused.json"),
            reference_images=[],
        )

        self.assertEqual(calls, 1)
        self.assertTrue(records[-1]["executed"])
        self.assertEqual(trajectory[-1]["answer"], "")

    def test_task_runner_records_repetition_stop_without_extra_model_call(self):
        for max_steps, expected_calls in ((10, 3), (2, 2), (3, 3)):
            with self.subTest(max_steps=max_steps):
                env = FakeTaskEnvironment()
                trajectory, records, calls = _run_agent_steps(
                    options=SimpleNamespace(
                        max_steps=max_steps, repeating_action_failure_th=3,
                        viewport_width=30, viewport_height=20,
                    ),
                    bindings=FakeTaskBindings(),
                    policy=OneDecisionPolicy(Decision(
                        state_summary="Target is visible",
                        action=ClickAction(kind="click", x=10, y=10),
                        expected_outcome="Click the target",
                    )),
                    action_factory=FakeActionFactory(), env=env,
                    task={"intent": "goal"}, runtime_config=Path("unused.json"),
                    reference_images=[],
                )
                self.assertEqual(calls, expected_calls)
                self.assertEqual(len(env.actions), expected_calls)
                self.assertEqual(len(trajectory), 2 * expected_calls + 2)
                self.assertEqual(trajectory[-1]["kind"], "stop")
                if max_steps > 3:
                    self.assertFalse(records[-1]["executed"])
                    self.assertEqual(records[-1]["decision"]["action"]["status"], "blocked")
                    self.assertEqual(
                        trajectory[-1]["answer"],
                        "Early stop: Same action for 3 times",
                    )
                else:
                    self.assertEqual(
                        trajectory[-1]["answer"],
                        f"Early stop: Reach max steps {max_steps}",
                    )


if __name__ == "__main__":
    unittest.main()
