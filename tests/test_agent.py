"""외부 API 호출 없이 vision-only 에이전트의 핵심 제어 흐름을 검증한다."""

import base64
import io
import json
import unittest

from PIL import Image
from pydantic import ValidationError

from ui_agent.controller import VisionAgentController
from ui_agent.models import (
    ClickAction,
    Decision,
    ExecutionResult,
    GoBackAction,
    GoForwardAction,
    HoverAction,
    Observation,
    PressAction,
    ScrollAction,
    StopAction,
    TypeAction,
)
from ui_agent.policy import INSTRUCTIONS, OpenAIVisionPolicy


def screenshot(color: str = "white") -> str:
    image = Image.new("RGB", (64, 64), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def observation(color: str = "white") -> Observation:
    return Observation(
        screenshot_base64=screenshot(color),
        viewport_width=1280,
        viewport_height=720,
    )


class SequencePolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def decide(
        self,
        task,
        current,
        history,
        reference_images=(),
        previous_observation=None,
    ):
        self.calls.append(
            {
                "history": list(history),
                "reference_images": list(reference_images),
                "previous_observation": previous_observation,
            }
        )
        action = self.actions.pop(0)
        return Decision(
            state_summary="현재 화면",
            action=action,
            expected_outcome="화면 변경",
        )


class FakeResponses:
    def __init__(self, decision):
        self.decision = decision
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return type("Response", (), {"output_parsed": self.decision})()


class FakeOpenAIClient:
    def __init__(self, decision):
        self.responses = FakeResponses(decision)


class AgentTests(unittest.TestCase):
    def test_decision_schema_avoids_gateway_unsupported_discriminator(self):
        schema = json.dumps(Decision.model_json_schema())
        self.assertNotIn('"discriminator"', schema)
        for removed_action in ("goto", "new_tab", "tab_focus", "close_tab"):
            self.assertNotIn(f'"{removed_action}"', schema)
        self.assertNotIn('"message"', schema)

    def test_coordinate_schema_rejects_out_of_range_and_fractional_values(self):
        for kind in ("click", "hover"):
            for axis in ("x", "y"):
                for value in (-1, 1001, 0.5):
                    with self.subTest(kind=kind, axis=axis, value=value):
                        with self.assertRaises(ValidationError):
                            Decision.model_validate({
                                "state_summary": "Target visible",
                                "action": {"kind": kind, "x": 500, "y": 500, axis: value},
                                "expected_outcome": "Move to target",
                            })

    def test_controller_allows_retry_and_nearby_click_without_reprompt(self):
        actions = [
            ClickAction(kind="click", x=120, y=120),
            ClickAction(kind="click", x=120, y=120),
            ClickAction(kind="click", x=125, y=123),
        ]
        policy = SequencePolicy(actions)
        controller = VisionAgentController(policy)
        current = observation()
        result = ExecutionResult(ok=True)

        for expected in actions:
            decision = controller.propose("테스트", current)
            self.assertEqual(decision.action, expected)
            controller.record(decision, result)
        self.assertEqual(controller.model_calls, 3)

    def test_controller_allows_returning_to_previous_screen(self):
        policy = SequencePolicy(
            [ClickAction(kind="click", x=100, y=100) for _ in range(3)]
        )
        controller = VisionAgentController(policy)
        state_a = observation(color="white")
        state_b = observation(color="black")

        first = controller.propose("테스트", state_a)
        controller.record(first, ExecutionResult(ok=True))
        second = controller.propose("테스트", state_b)
        controller.record(second, ExecutionResult(ok=True))
        third = controller.propose("테스트", state_a)

        self.assertIsInstance(third.action, ClickAction)
        self.assertIsNone(policy.calls[0]["previous_observation"])
        self.assertEqual(policy.calls[1]["previous_observation"], state_a)
        self.assertEqual(policy.calls[2]["previous_observation"], state_b)

    def test_gpt_and_controller_keep_normalized_coordinates_in_history(self):
        for action in (ClickAction(kind="click", x=500, y=900),
                       HoverAction(kind="hover", x=1000, y=1000)):
            with self.subTest(action=action):
                expected = Decision(state_summary="Target visible", action=action,
                                    expected_outcome="Move to target")
                client = FakeOpenAIClient(expected)
                controller = VisionAgentController(OpenAIVisionPolicy(client, "gpt-5.6-luna"))
                current = observation().model_copy(update={"viewport_width": 640, "viewport_height": 360})
                decision = controller.propose("goal", current)
                self.assertEqual(decision, expected)
                controller.record(decision, ExecutionResult(ok=True))
                controller.propose("goal", current)
                request = client.responses.request
                self.assertEqual(request["instructions"], INSTRUCTIONS)
                self.assertIn("integer coordinates normalized to 0-1000", request["instructions"])
                self.assertIs(request["text_format"], Decision)
                prompt = request["input"][0]["content"][0]["text"]
                recent = json.loads(prompt.split("Recent executed actions: ", 1)[1])
                self.assertEqual(recent[0]["action"], action.model_dump())

    def test_controller_passes_task_reference_images(self):
        policy = SequencePolicy(
            [StopAction(kind="stop", status="success", answer="완료")]
        )
        controller = VisionAgentController(policy)

        controller.propose("테스트", observation(), reference_images=["reference"])

        self.assertEqual(policy.calls[0]["reference_images"], ["reference"])

    def test_repetition_threshold_uses_execution_history_without_model_call(self):
        for action in (
            ClickAction(kind="click", x=10, y=10),
            TypeAction(kind="type", text="query"),
            PressAction(kind="press", key_comb="Enter"),
            ScrollAction(kind="scroll", direction="down"),
            HoverAction(kind="hover", x=10, y=10),
            GoBackAction(kind="go_back"),
        ):
            with self.subTest(action=action):
                policy = SequencePolicy([action] * 3)
                controller = VisionAgentController(policy, repeating_action_failure_th=3)
                for index in range(3):
                    decision = controller.propose("goal", observation())
                    self.assertEqual(decision.action, action)
                    # 실행 성공 여부와 무관하게 실제 실행은 반복 횟수에 포함한다.
                    controller.record(decision, ExecutionResult(ok=False))
                self.assertEqual(
                    [step.repeat_count for step in controller.history], [1, 2, 3]
                )
                self.assertEqual(controller.history[-1].state_summary, "현재 화면")
                self.assertIn(
                    '"repeat_count": 3',
                    OpenAIVisionPolicy._build_prompt(
                        "goal", observation(), controller.history
                    ),
                )
                stopped = controller.propose("goal", observation())
                self.assertEqual(stopped.action.status, "blocked")
                self.assertEqual(controller.model_calls, 3)
                self.assertEqual(len(policy.calls), 3)

    def test_intervening_action_resets_repetition(self):
        cases = (
            (TypeAction(kind="type", text="query"), ClickAction(kind="click", x=10, y=10)),
            (ScrollAction(kind="scroll", direction="down"), ScrollAction(kind="scroll", direction="up")),
        )
        for repeated, intervening in cases:
            with self.subTest(action=repeated):
                actions = [repeated, repeated, intervening, repeated, repeated]
                controller = VisionAgentController(SequencePolicy(actions), repeating_action_failure_th=3)
                for expected in actions:
                    decision = controller.propose("goal", observation())
                    self.assertEqual(decision.action, expected)
                    controller.record(decision, ExecutionResult(ok=True))
                self.assertEqual(
                    [step.repeat_count for step in controller.history], [1, 2, 1, 1, 2]
                )

    def test_action_equivalence_matches_low_level_actions(self):
        cases = [
            (ClickAction(kind="click", x=120, y=120),
             ClickAction(kind="click", x=125, y=123), False),
            (ScrollAction(kind="scroll", direction="down"),
             ScrollAction(kind="scroll", direction="down"), True),
            (ScrollAction(kind="scroll", direction="down"),
             ScrollAction(kind="scroll", direction="up"), False),
            (PressAction(kind="press", key_comb="a"),
             PressAction(kind="press", key_comb="A"), False),
            (TypeAction(kind="type", text="a"),
             TypeAction(kind="type", text="A"), False),
            (PressAction(kind="press", key_comb="a"),
             TypeAction(kind="type", text="a"), False),
            (GoBackAction(kind="go_back"), GoBackAction(kind="go_back"), True),
            (GoBackAction(kind="go_back"), GoForwardAction(kind="go_forward"), False),
        ]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertEqual(left == right, expected)

    def test_controller_rejects_empty_actions_without_execution_or_retry(self):
        for action in (
            TypeAction(kind="type", text=""),
            PressAction(kind="press", key_comb=" "),
        ):
            with self.subTest(action=action):
                controller = VisionAgentController(SequencePolicy([action]))
                self.assertEqual(controller.propose("goal", observation()).action.status,
                                 "blocked")
                self.assertEqual(controller.model_calls, 1)
                self.assertEqual(controller.history, [])

    def test_controller_rejects_nonpositive_threshold(self):
        for threshold in (0, -1):
            with self.assertRaises(ValueError):
                VisionAgentController(SequencePolicy([]), threshold)

    def test_policy_sends_previous_current_and_reference_images(self):
        done = Decision(
            state_summary="완료 화면",
            action=StopAction(kind="stop", status="success", answer="완료"),
            expected_outcome="종료",
        )
        client = FakeOpenAIClient(done)
        policy = OpenAIVisionPolicy(client, "vision-model")
        current_image = screenshot("white")
        previous_image = screenshot("red")
        reference_image = screenshot("black")

        policy.decide(
            "테스트",
            observation().model_copy(
                update={"screenshot_base64": current_image}
            ),
            history=[],
            reference_images=[reference_image],
            previous_observation=observation().model_copy(
                update={"screenshot_base64": previous_image}
            ),
        )

        content = client.responses.request["input"][0]["content"]
        image_urls = [
            item["image_url"] for item in content if item["type"] == "input_image"
        ]
        self.assertEqual(
            image_urls,
            [
                f"data:image/png;base64,{previous_image}",
                f"data:image/png;base64,{current_image}",
                f"data:image/png;base64,{reference_image}",
            ],
        )
        self.assertIn("Previous screenshot", content[1]["text"])
        self.assertIn("Current screenshot", content[3]["text"])
        self.assertIn("Recent executed actions:", content[0]["text"])
        self.assertNotIn("Current URL", content[0]["text"])
        self.assertNotIn("Open tabs", content[0]["text"])


if __name__ == "__main__":
    unittest.main()
