import json
import unittest

from pydantic import ValidationError

from ui_agent.browser import BrowserSession
from ui_agent.models import (
    ClickAction,
    Decision,
    DoneAction,
    ExecutionResult,
    Observation,
)
from ui_agent.runner import AgentRunner, DomainEvaluator


def observation(url: str = "https://example.com") -> Observation:
    return Observation(
        url=url,
        screenshot_base64="image",
        viewport_width=1280,
        viewport_height=720,
    )


class FakePerception:
    def __init__(self, value: Observation):
        self.value = value

    def capture(self) -> Observation:
        return self.value


class DonePolicy:
    def decide(self, task, current, history):
        return Decision(
            action=DoneAction(kind="done", status="success", summary="완료"),
            expected_outcome="종료",
        )


class NeverExecutor:
    def execute(self, action):
        return ExecutionResult(ok=True)


class ClickPolicy:
    def decide(self, task, current, history):
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
        self.assertEqual(len(result.steps), 2)
        self.assertFalse(result.steps[0].result.state_changed)
        self.assertIn("반복 실행", result.steps[1].result.message)
        self.assertEqual(result.steps[0].expected_outcome, "화면이 변경됩니다.")


if __name__ == "__main__":
    unittest.main()
