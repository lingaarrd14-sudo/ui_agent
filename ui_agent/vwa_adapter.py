"""vision-only 제어기와 VisualWebArena 브라우저 액션 사이의 얇은 어댑터."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any, Protocol

from PIL import Image

from .controller import VisionAgentController, VisionPolicy
from .models import (
    ClickAction,
    Decision,
    DoneAction,
    ExecutionResult,
    Observation,
    PressAction,
    ScrollAction,
    StepRecord,
    TypeAction,
)


class VWAActionFactory(Protocol):
    """VisualWebArena 버전에 종속된 액션 생성 함수의 최소 계약."""

    def click(self, x: int, y: int) -> Any: ...

    def type_text(self, text: str) -> Any: ...

    def press(self, key: str) -> Any: ...

    def scroll(self, direction: str) -> Any: ...

    def stop(self, answer: str) -> Any: ...


def pil_to_base64(image: Image.Image) -> str:
    """Pillow 이미지를 OpenAI 입력용 PNG base64로 변환한다."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def observation_from_vwa_state(
    state: dict[str, Any], viewport_width: int, viewport_height: int
) -> Observation:
    """VWA의 image 관찰만 사용해 공통 Observation을 만든다."""
    pixels = state["observation"]["image"]
    image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(pixels)
    return Observation(
        url=state["info"]["page"].url,
        screenshot_base64=pil_to_base64(image),
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )


def decision_to_vwa_action(decision: Decision, factory: VWAActionFactory) -> Any:
    """공통 액션을 VWA 저수준 Playwright 액션으로 변환한다."""
    selected = decision.action
    if isinstance(selected, ClickAction):
        action = factory.click(selected.x, selected.y)
    elif isinstance(selected, TypeAction):
        action = factory.type_text(selected.text)
    elif isinstance(selected, PressAction):
        action = factory.press(selected.key)
    elif isinstance(selected, ScrollAction):
        action = factory.scroll("up" if selected.delta_y < 0 else "down")
    elif isinstance(selected, DoneAction):
        action = factory.stop(selected.summary)
    else:  # pragma: no cover - Pydantic가 액션 합집합을 제한한다.
        raise TypeError(f"지원하지 않는 액션입니다: {selected!r}")
    action["raw_prediction"] = decision.model_dump_json()
    return action


class VisualWebArenaAdapter:
    """VWA 상태를 공통 제어기에 연결하고 실행 결과를 감사 가능하게 기록한다."""

    def __init__(
        self,
        policy: VisionPolicy,
        action_factory: VWAActionFactory,
        viewport_width: int,
        viewport_height: int,
        max_proposals: int = 3,
    ) -> None:
        self.action_factory = action_factory
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.controller = VisionAgentController(
            policy=policy, max_proposals=max_proposals
        )

    @property
    def history(self) -> list[StepRecord]:
        return self.controller.history

    def observe(self, state: dict[str, Any]) -> Observation:
        return observation_from_vwa_state(
            state, self.viewport_width, self.viewport_height
        )

    def decide(
        self,
        task: str,
        state: dict[str, Any],
        reference_images: Sequence[str] = (),
    ) -> tuple[Decision, Observation, Any]:
        """현재 원시 스크린샷에서 검증된 액션 하나를 제안한다."""
        before = self.observe(state)
        decision = self.controller.propose(task, before, reference_images)
        return decision, before, decision_to_vwa_action(decision, self.action_factory)

    def record(
        self,
        before: Observation,
        decision: Decision,
        result: ExecutionResult,
        after_state: dict[str, Any],
    ) -> StepRecord:
        """VWA 실행 뒤 얻은 원시 스크린샷으로 실제 상태 변화를 검증한다."""
        return self.controller.record(
            before=before,
            decision=decision,
            result=result,
            after=self.observe(after_state),
        )
