"""실제 SDK를 사용하되 HTTP를 mock하여 Gemini 연결만 검증한다."""

import json
import os
import unittest
from unittest.mock import patch

import httpx2
from openai import OpenAI
from pydantic import ValidationError

from ui_agent.models import ClickAction, Decision, ExecutionResult, Observation, StepRecord
from ui_agent.policy import INSTRUCTIONS, OpenAIVisionPolicy
from ui_agent.vwa_config import configure_environment


class GeminiPolicyTests(unittest.TestCase):
    def setUp(self):
        self.decision = Decision(
            state_summary="Button visible", action=ClickAction(kind="click", x=830, y=120),
            expected_outcome="Button opens",
        )
        self.choices = [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": self.decision.model_dump_json()},
        }]
        self.requests = []
        self.client = OpenAI(
            api_key="test", base_url="https://example.test/v1/", max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.respond)),
        )
        self.addCleanup(self.client.close)
        self.policy = OpenAIVisionPolicy(self.client, "gemini-3.8-flash")
        self.observation = Observation(
            screenshot_base64="current", viewport_width=1280, viewport_height=720,
        )

    def respond(self, request):
        self.requests.append(request)
        return httpx2.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "gemini-3.8-flash", "choices": self.choices,
        })

    def test_chat_request_preserves_images_schema_and_normalized_coordinates(self):
        previous = self.observation.model_copy(update={"screenshot_base64": "previous"})
        history = [StepRecord(**self.decision.model_dump(), result=ExecutionResult(ok=True))]
        decision = self.policy.decide(
            "Click the button", self.observation, history, ["reference"], previous,
        )
        self.assertEqual(decision, self.decision)
        self.assertEqual(self.requests[0].url.path, "/v1/chat/completions")
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["messages"][0]["content"], INSTRUCTIONS)
        self.assertIn("integer coordinates normalized to 0-1000", INSTRUCTIONS)
        self.assertNotIn("tools", body)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        schema = body["response_format"]["json_schema"]["schema"]
        for name in ("ClickAction", "HoverAction"):
            for axis in ("x", "y"):
                field = schema["$defs"][name]["properties"][axis]
                self.assertEqual((field["minimum"], field["maximum"]), (0, 1000))
        content = body["messages"][1]["content"]
        self.assertIn("Viewport: 1280x720", content[0]["text"])
        recent = json.loads(content[0]["text"].split("Recent executed actions: ", 1)[1])
        self.assertEqual(recent[0]["action"], self.decision.action.model_dump())
        self.assertEqual(
            [part["image_url"]["url"] for part in content if part["type"] == "image_url"],
            [f"data:image/png;base64,{name}" for name in ("previous", "current", "reference")],
        )

    def test_empty_refused_and_invalid_responses_are_rejected(self):
        self.choices[0]["message"] = {"role": "assistant", "content": None, "refusal": "Refused"}
        with self.assertRaises(RuntimeError):
            self.policy.decide("Click", self.observation, [])
        self.choices[0]["message"] = {"role": "assistant", "content": "invalid json"}
        with self.assertRaises(ValidationError):
            self.policy.decide("Click", self.observation, [])
        invalid = self.decision.model_dump()
        invalid["action"]["x"] = 1001
        self.choices[0]["message"] = {"role": "assistant", "content": json.dumps(invalid)}
        with self.assertRaises(ValidationError):
            self.policy.decide("Click", self.observation, [])
        self.choices = []
        with self.assertRaises(RuntimeError):
            self.policy.decide("Click", self.observation, [])

    def test_gemini_credentials_are_separate_from_openai_judge(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "judge", "GEMINI_API_KEY": "agent",
                                    "OPENAI_BASE_URL": "https://judge.test/v1"}, clear=True), \
                patch("ui_agent.vwa_config.load_local_environment"):
            self.assertEqual(configure_environment(True, "gemini-3.8-flash"),
                             ("agent", "https://generativelanguage.googleapis.com/v1beta/openai/"))
            self.assertEqual(configure_environment(True, "gpt-5.6-terra"),
                             ("judge", "https://judge.test/v1"))
            del os.environ["GEMINI_API_KEY"]
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                configure_environment(True, "gemini-3.8-flash")
