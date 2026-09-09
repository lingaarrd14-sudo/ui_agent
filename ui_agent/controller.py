"""브라우저 구현과 독립적인 vision-only 에이전트 제어기."""

from collections.abc import Sequence
from typing import Protocol

from .models import (
    Action,
    ClickAction,
    Decision,
    ExecutionResult,
    HoverAction,
    Observation,
    PressAction,
    StepRecord,
    StopAction,
    TypeAction,
    ScrollAction,
)


class VisionPolicy(Protocol):
    """원시 스크린샷에서 구조화된 액션 하나를 선택하는 정책."""

    def decide(
        self,
        task: str,
        observation: Observation,
        history: Sequence[StepRecord],
        reference_images: Sequence[str] = (),
        previous_observation: Observation | None = None,
    ) -> Decision: ...


def is_equivalent(left: Action, right: Action) -> bool:
    """VWA 액션 동등성처럼 같은 종류의 인자 없는 액션도 동일하게 본다."""
    if isinstance(left, ScrollAction) and isinstance(right, ScrollAction):
          return True
    return left == right


def validation_error(action: Action, observation: Observation) -> str | None:
    if isinstance(action, (ClickAction, HoverAction)) and (
        action.x >= observation.viewport_width or action.y >= observation.viewport_height
    ):
        return f"{action.kind} 좌표 ({action.x}, {action.y})가 뷰포트 밖입니다."
    if isinstance(action, TypeAction) and not action.text:
        return "빈 문자열은 입력할 수 없습니다."
    if isinstance(action, PressAction) and not action.key_comb.strip():
        return "빈 키 이름은 누를 수 없습니다."
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
        self.previous_observation: Observation | None = None

    def propose(
        self,
        task: str,
        observation: Observation,
        reference_images: Sequence[str] = (),
    ) -> Decision:
        """VWA early_stop처럼 실행된 동일 액션이 임계치에 도달하면 종료한다."""
        k = self.repeating_action_failure_th
        if self.history and self.history[-1].repeat_count >= k:
            return self._blocked(f"Early stop: Same action for {k} times")

        self.model_calls += 1
        decision = self.policy.decide(
            task,
            observation,
            self.history,
            reference_images=reference_images,
            previous_observation=self.previous_observation,
        )
        self.previous_observation = observation
        reason = validation_error(decision.action, observation)
        return self._blocked(reason) if reason else decision

    @staticmethod
    def _blocked(reason: str) -> Decision:
        return Decision(
            state_summary=reason,
            action=StopAction(kind="stop", status="blocked", answer=reason),
            expected_outcome="실행을 종료합니다.",
        )

    def record(
        self,
        decision: Decision,
        result: ExecutionResult,
    ) -> StepRecord:
        """실행 결과를 기록하고 화면 변화 판단은 다음 관찰에 맡긴다."""
        repeat_count = 1
        if self.history and is_equivalent(self.history[-1].action, decision.action):
            repeat_count = self.history[-1].repeat_count + 1
        record = StepRecord(
            state_summary=decision.state_summary,
            action=decision.action,
            repeat_count=repeat_count,
            expected_outcome=decision.expected_outcome,
            result=result,
        )
        self.history.append(record)
        return record
