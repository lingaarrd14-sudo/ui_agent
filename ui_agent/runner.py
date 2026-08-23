"""관찰, 판단, 실행, 검증을 연결하는 UI 에이전트 실행 루프."""

from collections.abc import Sequence
from typing import Protocol
from urllib.parse import urlparse

from .models import (
    Action,
    Decision,
    DoneAction,
    ExecutionResult,
    Observation,
    RunResult,
    StepRecord,
)


class Perception(Protocol):
    """현재 브라우저 상태를 관찰하는 컴포넌트의 인터페이스."""

    def capture(self) -> Observation: ...


class Policy(Protocol):
    """관찰 결과에서 다음 액션을 결정하는 컴포넌트의 인터페이스."""

    def decide(
        self, task: str, observation: Observation, history: Sequence[StepRecord]
    ) -> Decision: ...


class Executor(Protocol):
    """결정된 액션을 외부 UI에 적용하는 컴포넌트의 인터페이스."""

    def execute(self, action: Action) -> ExecutionResult: ...


class DomainEvaluator:
    """현재 URL이 선택적 목표 도메인에 도착했는지 판정한다."""

    def __init__(self, target_domain: str | None):
        self.target_domain = target_domain.lower() if target_domain else None

    def is_success(self, observation: Observation) -> bool:
        """정확한 호스트 또는 그 하위 도메인이면 성공으로 처리한다."""
        if not self.target_domain:
            return False
        hostname = (urlparse(observation.url).hostname or "").lower()
        return hostname == self.target_domain or hostname.endswith(
            f".{self.target_domain}"
        )


class AgentRunner:
    """컴포넌트를 순서대로 호출하고 실행 이력과 종료 상태를 관리한다."""

    def __init__(self, perception, policy, executor, evaluator):
        self.perception = perception
        self.policy = policy
        self.executor = executor
        self.evaluator = evaluator

    def run(self, task: str, max_steps: int) -> RunResult:
        """목표가 완료되거나 최대 단계에 도달할 때까지 액션을 반복한다."""
        history: list[StepRecord] = []

        for step in range(1, max_steps + 1):
            observation = self.perception.capture()
            if self.evaluator.is_success(observation):
                # 도메인 판정은 모델 호출보다 먼저 수행해 불필요한 요청을 피한다.
                return RunResult(
                    status="success", summary="목표 도메인에 도착했습니다.", steps=history
                )

            decision = self.policy.decide(task, observation, history)
            print(f"{step}: {decision.action.model_dump(mode='json')}")

            if isinstance(decision.action, DoneAction):
                return RunResult(
                    status=decision.action.status,
                    summary=decision.action.summary,
                    steps=history,
                )

            if self._is_stalled_repeat(decision.action, history):
                # 변화가 없었던 동일 액션을 다시 실행해 무한 클릭하는 상황을 막는다.
                result = ExecutionResult(
                    ok=False,
                    message="화면 변화가 없어 동일한 액션의 반복 실행을 차단했습니다.",
                    before_url=observation.url,
                    after_url=observation.url,
                )
            else:
                result = self.executor.execute(decision.action)
                # 실행 후 다시 관찰해 URL뿐 아니라 화면 자체의 변화도 검증한다.
                after = self.perception.capture()
                result = result.model_copy(
                    update={
                        "before_url": observation.url,
                        "after_url": after.url,
                        "state_changed": (
                            observation.url != after.url
                            or observation.screenshot_base64
                            != after.screenshot_base64
                        ),
                    }
                )

            history.append(
                StepRecord(
                    url=observation.url,
                    action=decision.action,
                    expected_outcome=decision.expected_outcome,
                    result=result,
                )
            )

        return RunResult(
            status="max_steps", summary="최대 실행 횟수에 도달했습니다.", steps=history
        )

    @staticmethod
    def _is_stalled_repeat(action: Action, history: list[StepRecord]) -> bool:
        """직전과 같은 실패 또는 무변화 액션인지 확인한다."""
        if not history:
            return False
        previous = history[-1]
        return previous.action == action and (
            not previous.result.ok or not previous.result.state_changed
        )
