"""외부 API 호출 없이 vision-only 에이전트의 핵심 제어 흐름을 검증한다."""

import base64
import io
import json
import unittest

from PIL import Image
from pydantic import ValidationError

from ui_agent.controller import VisionAgentController, is_equivalent
from ui_agent.models import (
    ClickAction,
    Decision,
    DoneAction,
    ExecutionResult,
    Observation,
    PressAction,
    ScrollAction,
    TypeAction,
)
from ui_agent.runner import AgentRunner, DomainEvaluator
from ui_agent.policy import INSTRUCTIONS, OpenAIVisionPolicy
from ui_agent.playwright_runtime import BrowserSession


def screenshot(color: str = "white") -> str:
    image = Image.new("RGB", (64, 64), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def observation(
    url: str = "https://example.com", color: str = "white"
) -> Observation:
    return Observation(
        url=url,
        screenshot_base64=screenshot(color),
        viewport_width=1280,
        viewport_height=720,
    )


class FakePerception:
    def __init__(self, value: Observation):
        self.value = value

    def capture(self) -> Observation:
        return self.value


class DonePolicy:
    def decide(
        self, task, current, history, reference_images=()
    ):
        return Decision(
            action=DoneAction(kind="done", status="success", summary="완료"),
            expected_outcome="종료",
        )


class NeverExecutor:
    def execute(self, action):
        return ExecutionResult(ok=True)


class ClickPolicy:
    def decide(
        self, task, current, history, reference_images=()
    ):
        return Decision(
            action=ClickAction(kind="click", x=10, y=10),
            expected_outcome="화면이 변경됩니다.",
        )


class CountingExecutor:
    def __init__(self):
        self.calls = 0

    def execute(self, action):
        self.calls += 1
        return ExecutionResult(ok=True)


class SequencePolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def decide(
        self, task, current, history, reference_images=()
    ):
        self.calls.append(
            {
                "history": list(history),
                "reference_images": list(reference_images),
            }
        )
        action = self.actions.pop(0)
        return Decision(action=action, expected_outcome="화면 변경")


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


class FakePage:
    def __init__(self):
        self.closed = False
        self.focused = False

    def is_closed(self):
        return self.closed

    def bring_to_front(self):
        self.focused = True


class FakeContext:
    def __init__(self, pages):
        self.pages = pages


class AgentTests(unittest.TestCase):
    def test_browser_session_adopts_new_page(self):
        first = FakePage()
        context = FakeContext([first])
        session = BrowserSession(context, first)
        popup = FakePage()
        context.pages.append(popup)

        self.assertIs(session.current_page(), popup)
        self.assertTrue(popup.focused)

    def test_decision_schema_avoids_gateway_unsupported_discriminator(self):
        schema = json.dumps(Decision.model_json_schema())
        self.assertNotIn('"discriminator"', schema)

    def test_invalid_click_is_rejected(self):
        with self.assertRaises(ValidationError):
            Decision.model_validate(
                {
                    "action": {"kind": "click", "x": -1, "y": 10},
                    "expected_outcome": "클릭",
                }
            )

    def test_domain_evaluator_accepts_subdomain(self):
        evaluator = DomainEvaluator("openai.com")
        self.assertTrue(evaluator.is_success(observation("https://chat.openai.com/")))
        self.assertFalse(evaluator.is_success(observation("https://notopenai.com/")))

    def test_runner_stops_on_done(self):
        runner = AgentRunner(
            FakePerception(observation()),
            DonePolicy(),
            NeverExecutor(),
            DomainEvaluator(None),
        )
        result = runner.run("테스트", max_steps=3)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.steps, [])

    def test_runner_stops_after_five_executed_repeats(self):
        executor = CountingExecutor()
        runner = AgentRunner(
            FakePerception(observation()),
            ClickPolicy(),
            executor,
            DomainEvaluator(None),
        )

        result = runner.run("테스트", max_steps=10)

        self.assertEqual(executor.calls, 5)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(len(result.steps), 5)
        self.assertEqual(result.steps[0].expected_outcome, "화면이 변경됩니다.")

    def test_controller_allows_retry_and_nearby_click_without_reprompt(self):
        policy = SequencePolicy(
            [
                ClickAction(kind="click", x=120, y=120),
                ClickAction(kind="click", x=120, y=120),
                ClickAction(kind="click", x=125, y=123),
            ]
        )
        controller = VisionAgentController(policy)
        current = observation()
        for expected_x in (120, 120, 125):
            decision = controller.propose("테스트", current)
            self.assertEqual(decision.action.x, expected_x)
            controller.record(current, decision, ExecutionResult(ok=True), current.url)
        self.assertEqual(controller.model_calls, 3)

    def test_controller_allows_returning_to_previous_screen(self):
        policy = SequencePolicy(
            [ClickAction(kind="click", x=100, y=100) for _ in range(3)]
        )
        controller = VisionAgentController(policy)
        state_a = observation(color="white")
        state_b = observation(color="black")

        first = controller.propose("테스트", state_a)
        controller.record(state_a, first, ExecutionResult(ok=True), state_b.url)
        second = controller.propose("테스트", state_b)
        controller.record(state_b, second, ExecutionResult(ok=True), state_a.url)
        self.assertIsInstance(controller.propose("테스트", state_a).action, ClickAction)

    def test_controller_rejects_viewport_boundary_coordinate(self):
        policy = SequencePolicy(
            [
                ClickAction(kind="click", x=1280, y=10),
                ClickAction(kind="click", x=1279, y=10),
            ]
        )
        controller = VisionAgentController(policy)

        decision = controller.propose("테스트", observation())

        self.assertEqual(decision.action.status, "blocked")
        self.assertIn("뷰포트", decision.action.summary)
        self.assertEqual(controller.model_calls, 1)
        self.assertEqual(controller.history, [])

    def test_controller_passes_task_reference_images(self):
        policy = SequencePolicy(
            [DoneAction(kind="done", status="success", summary="완료")]
        )
        controller = VisionAgentController(policy)

        controller.propose("테스트", observation(), reference_images=["reference"])

        self.assertEqual(policy.calls[0]["reference_images"], ["reference"])

    def test_repetition_threshold_uses_execution_history_without_model_call(self):
        for action in (
            ClickAction(kind="click", x=10, y=10),
            TypeAction(kind="type", text="query"),
            PressAction(kind="press", key="Enter"),
            ScrollAction(kind="scroll", delta_y=500),
        ):
            with self.subTest(action=action):
                policy = SequencePolicy([action] * 3)
                controller = VisionAgentController(policy, repeating_action_failure_th=3)
                current = observation()
                for index in range(3):
                    decision = controller.propose("goal", current)
                    self.assertEqual(decision.action, action)
                    # 실패한 실행과 URL 변화도 VWA처럼 반복 횟수에 포함한다.
                    controller.record(current, decision, ExecutionResult(ok=False),
                                      f"https://example.com/{index}")
                self.assertEqual(
                    [step.repeat_count for step in controller.history], [1, 2, 3]
                )
                self.assertIn(
                    '"repeat_count": 3',
                    OpenAIVisionPolicy._build_prompt(
                        "goal", current, controller.history
                    ),
                )
                stopped = controller.propose("goal", current)
                self.assertEqual(stopped.action.status, "blocked")
                self.assertEqual(controller.model_calls, 3)
                self.assertEqual(len(policy.calls), 3)

    def test_intervening_action_resets_keyboard_typing_repetition(self):
        typing = TypeAction(kind="type", text="query")
        click = ClickAction(kind="click", x=10, y=10)
        actions = [typing, typing, click, typing, typing]
        policy = SequencePolicy(actions)
        controller = VisionAgentController(policy, repeating_action_failure_th=3)
        current = observation()
        for expected in actions:
            decision = controller.propose("goal", current)
            self.assertEqual(decision.action, expected)
            controller.record(current, decision, ExecutionResult(ok=True), current.url)
        self.assertEqual(
            [step.repeat_count for step in controller.history], [1, 2, 1, 1, 2]
        )

    def test_action_equivalence_matches_low_level_actions(self):
        cases = [
            (ClickAction(kind="click", x=120, y=120),
             ClickAction(kind="click", x=125, y=123), False),
            (ScrollAction(kind="scroll", delta_y=100),
             ScrollAction(kind="scroll", delta_y=900), True),
            (ScrollAction(kind="scroll", delta_y=100),
             ScrollAction(kind="scroll", delta_y=-100), False),
            (PressAction(kind="press", key="a"),
             PressAction(kind="press", key="A"), False),
            (TypeAction(kind="type", text="a"),
             TypeAction(kind="type", text="A"), False),
            (PressAction(kind="press", key="a"),
             TypeAction(kind="type", text="a"), False),
        ]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertEqual(is_equivalent(left, right), expected)

    def test_controller_rejects_empty_actions_without_execution_or_retry(self):
        for action in (TypeAction(kind="type", text=""),
                       PressAction(kind="press", key=" "),
                       ScrollAction(kind="scroll", delta_y=0)):
            with self.subTest(action=action):
                controller = VisionAgentController(SequencePolicy([action]))
                self.assertEqual(controller.propose("goal", observation()).action.status,
                                 "blocked")
                self.assertEqual(controller.model_calls, 1)
                self.assertEqual(controller.history, [])

    def test_controller_rejects_nonpositive_threshold(self):
        for threshold in (0, -1):
            with self.assertRaises(ValueError):
                VisionAgentController(ClickPolicy(), threshold)

    def test_policy_sends_only_current_screenshot_and_reference_image(self):
        self.assertIn("repeat_count", INSTRUCTIONS)
        done = Decision(
            action=DoneAction(kind="done", status="success", summary="완료"),
            expected_outcome="종료",
        )
        client = FakeOpenAIClient(done)
        policy = OpenAIVisionPolicy(client, "vision-model")
        current_image = screenshot("white")
        reference_image = screenshot("black")

        policy.decide(
            "테스트",
            observation().model_copy(
                update={"screenshot_base64": current_image}
            ),
            history=[],
            reference_images=[reference_image],
        )

        content = client.responses.request["input"][0]["content"]
        image_urls = [
            item["image_url"] for item in content if item["type"] == "input_image"
        ]
        self.assertEqual(
            image_urls,
            [
                f"data:image/png;base64,{current_image}",
                f"data:image/png;base64,{reference_image}",
            ],
        )
        self.assertIn("Recent executed actions:", content[0]["text"])


if __name__ == "__main__":
    unittest.main()
