"""브라우저 구현과 독립적인 vision-only 에이전트 제어기."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .models import (
    Action,
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
from .state import DEFAULT_CHANGE_THRESHOLD, VisualState


class VisionPolicy(Protocol):
    """원시 스크린샷에서 구조화된 액션 하나를 선택하는 정책."""

    def decide(
        self,
        task: str,
        observation: Observation,
        history: Sequence[StepRecord],
        reference_images: Sequence[str] = (),
        feedback: Sequence[str] = (),
    ) -> Decision: ...


@dataclass(frozen=True)
class Transition:
    """실제로 실행된 액션의 전후 시각 상태."""

    before: VisualState
    action_signature: tuple[object, ...]
    after: VisualState
    changed: bool


def action_signature(action: Action, click_bucket: int = 12) -> tuple[object, ...]:
    """좌표의 미세한 흔들림을 같은 행동으로 취급하는 액션 식별자."""
    if isinstance(action, ClickAction):
        return ("click", action.x // click_bucket, action.y // click_bucket)
    if isinstance(action, TypeAction):
        return ("type", action.text)
    if isinstance(action, PressAction):
        return ("press", action.key.lower())
    if isinstance(action, ScrollAction):
        direction = "up" if action.delta_y < 0 else "down"
        return ("scroll", direction)
    if isinstance(action, DoneAction):
        return ("done", action.status, action.summary)
    raise TypeError(f"지원하지 않는 액션입니다: {action!r}")


class LoopGuard:
    """동일 상태 재시도와 여러 단계를 거친 상태 순환을 실행 전에 차단한다."""

    def __init__(
        self,
        change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
        history_limit: int = 30,
    ) -> None:
        self.change_threshold = change_threshold
        self.history_limit = history_limit
        self._transitions: list[Transition] = []

    def rejection_reason(self, observation: Observation, action: Action) -> str | None:
        """현재 상태에서 액션을 실행하면 안 되는 구체적인 이유를 반환한다."""
        validation_error = self._validate(action, observation)
        if validation_error:
            return validation_error

        current = VisualState.from_observation(observation)
        signature = action_signature(action)
        for transition in reversed(self._transitions):
            if transition.action_signature != signature:
                continue
            if transition.before.equivalent(current, self.change_threshold):
                if transition.changed:
                    return (
                        "이 화면에서 사실상 같은 액션을 이미 실행한 뒤 다시 이 화면으로 "
                        "돌아왔습니다. 상태 순환을 피하고 다른 액션을 선택하세요."
                    )
                return (
                    "이 화면에서 사실상 같은 액션을 실행했지만 화면이나 URL이 "
                    "변하지 않았습니다. 다른 대상을 선택하세요."
                )
        return None

    def record(
        self,
        before: Observation,
        action: Action,
        after: Observation,
        changed: bool,
    ) -> None:
        """실행된 전이를 제한된 크기로 기록한다."""
        self._transitions.append(
            Transition(
                before=VisualState.from_observation(before),
                action_signature=action_signature(action),
                after=VisualState.from_observation(after),
                changed=changed,
            )
        )
        if len(self._transitions) > self.history_limit:
            del self._transitions[: -self.history_limit]

    @staticmethod
    def _validate(action: Action, observation: Observation) -> str | None:
        if isinstance(action, ClickAction) and (
            action.x >= observation.viewport_width
            or action.y >= observation.viewport_height
        ):
            return (
                f"클릭 좌표 ({action.x}, {action.y})가 뷰포트 "
                f"{observation.viewport_width}x{observation.viewport_height} 밖입니다."
            )
        if isinstance(action, TypeAction) and not action.text:
            return "빈 문자열은 입력할 수 없습니다."
        if isinstance(action, PressAction) and not action.key.strip():
            return "빈 키 이름은 누를 수 없습니다."
        if isinstance(action, ScrollAction) and action.delta_y == 0:
            return "스크롤 거리는 0일 수 없습니다."
        return None


class VisionAgentController:
    """정책 제안, 안전 검증, 실행 결과 기록을 한 흐름으로 관리한다."""

    def __init__(
        self,
        policy: VisionPolicy,
        max_proposals: int = 3,
        change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
    ) -> None:
        if max_proposals < 1:
            raise ValueError("max_proposals는 1 이상이어야 합니다.")
        self.policy = policy
        self.max_proposals = max_proposals
        self.history: list[StepRecord] = []
        self.loop_guard = LoopGuard(change_threshold=change_threshold)
        self.last_rejections: list[str] = []
        self.last_proposal_count = 0

    def propose(
        self,
        task: str,
        observation: Observation,
        reference_images: Sequence[str] = (),
    ) -> Decision:
        """실행 가능한 결정을 얻거나 안전한 blocked 결정을 합성한다."""
        feedback: list[str] = []
        self.last_rejections = []
        self.last_proposal_count = 0
        for _ in range(self.max_proposals):
            self.last_proposal_count += 1
            decision = self.policy.decide(
                task,
                observation,
                self.history,
                reference_images=reference_images,
                feedback=feedback,
            )
            if isinstance(decision.action, DoneAction):
                return decision

            reason = self.loop_guard.rejection_reason(observation, decision.action)
            if reason is None:
                return decision
            message = (
                f"Rejected proposal {decision.action.model_dump(mode='json')}: {reason}"
            )
            feedback.append(message)
            self.last_rejections.append(message)

        return Decision(
            action=DoneAction(
                kind="done",
                status="blocked",
                summary=(
                    "안전 검증을 통과하는 새 액션을 선택하지 못해 반복 실행을 "
                    "중단했습니다."
                ),
            ),
            expected_outcome="반복 또는 잘못된 액션을 실행하지 않고 종료합니다.",
        )

    def record(
        self,
        before: Observation,
        decision: Decision,
        result: ExecutionResult,
        after: Observation,
    ) -> StepRecord:
        """실행 결과를 실제 시각 변화로 보정하고 다음 판단 이력에 추가한다."""
        before_state = VisualState.from_observation(before)
        after_state = VisualState.from_observation(after)
        changed = not before_state.equivalent(
            after_state, self.loop_guard.change_threshold
        )
        normalized_result = result.model_copy(
            update={
                "before_url": before.url,
                "after_url": after.url,
                "state_changed": changed,
            }
        )
        record = StepRecord(
            url=before.url,
            action=decision.action,
            expected_outcome=decision.expected_outcome,
            result=normalized_result,
        )
        self.history.append(record)
        self.loop_guard.record(before, decision.action, after, changed)
        return record
