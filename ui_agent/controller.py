"""브라우저 구현과 독립적인 vision-only 에이전트 제어기."""

from collections.abc import Sequence
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


class VisionPolicy(Protocol):
    """원시 스크린샷에서 구조화된 액션 하나를 선택하는 정책."""

    def decide(
        self,
        task: str,
        observation: Observation,
        history: Sequence[StepRecord],
        reference_images: Sequence[str] = (),
    ) -> Decision: ...


def is_equivalent(left: Action, right: Action) -> bool:
    """VWA 저수준 액션 기준: 클릭 좌표, 입력 텍스트, 키, 스크롤 방향."""
    if isinstance(left, ScrollAction) and isinstance(right, ScrollAction):
        return (left.delta_y < 0) == (right.delta_y < 0)
    return left == right


def validation_error(action: Action, observation: Observation) -> str | None:
    if isinstance(action, ClickAction) and (
        action.x >= observation.viewport_width or action.y >= observation.viewport_height
    ):
        return f"클릭 좌표 ({action.x}, {action.y})가 뷰포트 밖입니다."
    if isinstance(action, TypeAction) and not action.text:
        return "빈 문자열은 입력할 수 없습니다."
    if isinstance(action, PressAction) and not action.key.strip():
        return "빈 키 이름은 누를 수 없습니다."
    if isinstance(action, ScrollAction) and action.delta_y == 0:
        return "스크롤 거리는 0일 수 없습니다."
    return None


class VisionAgentController:
    """실행 이력으로 조기 종료를 판단하고 단계당 액션 하나를 요청한다."""

    def __init__(self, policy: VisionPolicy, repeating_action_failure_th: int = 5):
        if repeating_action_failure_th < 1:
            raise ValueError("repeating_action_failure_th는 1 이상이어야 합니다.")
        self.policy = policy
        self.repeating_action_failure_th = repeating_action_failure_th
        self.history: list[StepRecord] = []
        self.model_calls = 0

    def propose(
        self,
        task: str,
        observation: Observation,
        reference_images: Sequence[str] = (),
    ) -> Decision:
        """VWA early_stop처럼 실행된 동일 액션이 임계치에 도달하면 종료한다."""
        k = self.repeating_action_failure_th
        if self.history and self.history[-1].repeat_count >= k:
            return self._blocked(f"동일 액션을 {k}회 연속 실행해 조기 종료했습니다.")

        self.model_calls += 1
        decision = self.policy.decide(
            task, observation, self.history, reference_images=reference_images
        )
        reason = validation_error(decision.action, observation)
        return self._blocked(reason) if reason else decision

    @staticmethod
    def _blocked(reason: str) -> Decision:
        return Decision(
            action=DoneAction(kind="done", status="blocked", summary=reason),
            expected_outcome="실행을 종료합니다.",
        )

    def record(
        self,
        before: Observation,
        decision: Decision,
        result: ExecutionResult,
        after_url: str,
    ) -> StepRecord:
        """실행 결과와 URL을 기록하고 화면 변화 판단은 다음 관찰에 맡긴다."""
        repeat_count = 1
        if self.history and is_equivalent(self.history[-1].action, decision.action):
            repeat_count = self.history[-1].repeat_count + 1
        record = StepRecord(
            url=before.url,
            action=decision.action,
            repeat_count=repeat_count,
            expected_outcome=decision.expected_outcome,
            result=result.model_copy(update={"after_url": after_url}),
        )
        self.history.append(record)
        return record
