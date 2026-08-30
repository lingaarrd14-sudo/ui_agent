"""외부 API 호출 없이 vision-only 에이전트의 핵심 제어 흐름을 검증한다."""

import base64
import io
import json
import unittest

from PIL import Image
from pydantic import ValidationError

from ui_agent.controller import VisionAgentController, VisualState
from ui_agent.models import (
    ClickAction,
    Decision,
    DoneAction,
    ExecutionResult,
    Observation,
)
from ui_agent.runner import AgentRunner, DomainEvaluator
from ui_agent.policy import OpenAIVisionPolicy
from ui_agent.playwright_runtime import BrowserSession


def screenshot(color: str = "white", changed_pixel: bool = False) -> str:
    image = Image.new("RGB", (64, 64), color)
    if changed_pixel:
        image.putpixel((0, 0), (0, 0, 0))
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
        self, task, current, history, reference_images=(), feedback=()
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
        self, task, current, history, reference_images=(), feedback=()
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
        self, task, current, history, reference_images=(), feedback=()
    ):
        self.calls.append(
            {
                "history": list(history),
                "reference_images": list(reference_images),
                "feedback": list(feedback),
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

    def test_runner_blocks_repeated_action_when_screen_did_not_change(self):
        executor = CountingExecutor()
        runner = AgentRunner(
            FakePerception(observation()),
            ClickPolicy(),
            executor,
            DomainEvaluator(None),
        )

        result = runner.run("테스트", max_steps=2)

        self.assertEqual(executor.calls, 1)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(len(result.steps), 1)
        self.assertFalse(result.steps[0].result.state_changed)
        self.assertEqual(result.steps[0].expected_outcome, "화면이 변경됩니다.")

    def test_controller_reprompts_after_nearby_stalled_click(self):
        policy = SequencePolicy(
            [
                ClickAction(kind="click", x=120, y=120),
                ClickAction(kind="click", x=125, y=123),
                ClickAction(kind="click", x=300, y=200),
            ]
        )
        controller = VisionAgentController(policy)
        current = observation()
        first = controller.propose("테스트", current)
        controller.record(current, first, ExecutionResult(ok=True), current)

        recovered = controller.propose("테스트", current)

        self.assertEqual((recovered.action.x, recovered.action.y), (300, 200))
        self.assertEqual(controller.last_proposal_count, 2)
        self.assertEqual(len(controller.last_rejections), 1)
        self.assertIn("변하지 않았습니다", policy.calls[-1]["feedback"][0])

    def test_controller_blocks_state_cycle(self):
        policy = SequencePolicy(
            [ClickAction(kind="click", x=100, y=100) for _ in range(5)]
        )
        controller = VisionAgentController(policy)
        state_a = observation(color="white")
        state_b = observation(color="black")

        first = controller.propose("테스트", state_a)
        controller.record(state_a, first, ExecutionResult(ok=True), state_b)
        second = controller.propose("테스트", state_b)
        controller.record(state_b, second, ExecutionResult(ok=True), state_a)
        blocked = controller.propose("테스트", state_a)

        self.assertIsInstance(blocked.action, DoneAction)
        self.assertEqual(blocked.action.status, "blocked")
        self.assertEqual(controller.last_proposal_count, 3)
        self.assertTrue(
            all("상태 순환" in reason for reason in controller.last_rejections)
        )

    def test_controller_rejects_viewport_boundary_coordinate(self):
        policy = SequencePolicy(
            [
                ClickAction(kind="click", x=1280, y=10),
                ClickAction(kind="click", x=1279, y=10),
            ]
        )
        controller = VisionAgentController(policy)

        decision = controller.propose("테스트", observation())

        self.assertEqual(decision.action.x, 1279)
        self.assertIn("뷰포트", controller.last_rejections[0])

    def test_controller_passes_task_reference_images(self):
        policy = SequencePolicy(
            [DoneAction(kind="done", status="success", summary="완료")]
        )
        controller = VisionAgentController(policy)

        controller.propose("테스트", observation(), reference_images=["reference"])

        self.assertEqual(policy.calls[0]["reference_images"], ["reference"])

    def test_visual_state_ignores_single_pixel_noise(self):
        before = observation()
        after = before.model_copy(
            update={"screenshot_base64": screenshot("white", changed_pixel=True)}
        )

        self.assertTrue(
            VisualState.from_observation(before).equivalent(
                VisualState.from_observation(after)
            )
        )

    def test_visual_state_detects_large_change(self):
        self.assertFalse(
            VisualState.from_observation(observation(color="white")).equivalent(
                VisualState.from_observation(observation(color="black"))
            )
        )

    def test_policy_sends_only_current_screenshot_and_reference_image(self):
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
            feedback=["이전 좌표는 화면 밖입니다."],
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
        self.assertIn("이전 좌표", content[0]["text"])


if __name__ == "__main__":
    unittest.main()
