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
    def capture(self) -> Observation: ...


class Policy(Protocol):
    def decide(
        self, task: str, observation: Observation, history: Sequence[StepRecord]
    ) -> Decision: ...


class Executor(Protocol):
    def execute(self, action: Action) -> ExecutionResult: ...


class DomainEvaluator:
    def __init__(self, target_domain: str | None):
        self.target_domain = target_domain.lower() if target_domain else None

    def is_success(self, observation: Observation) -> bool:
        if not self.target_domain:
            return False
        hostname = (urlparse(observation.url).hostname or "").lower()
        return hostname == self.target_domain or hostname.endswith(
            f".{self.target_domain}"
        )


class AgentRunner:
    def __init__(self, perception, policy, executor, evaluator):
        self.perception = perception
        self.policy = policy
        self.executor = executor
        self.evaluator = evaluator

    def run(self, task: str, max_steps: int) -> RunResult:
        history: list[StepRecord] = []

        for step in range(1, max_steps + 1):
            observation = self.perception.capture()
            if self.evaluator.is_success(observation):
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
                result = ExecutionResult(
                    ok=False,
                    message="화면 변화가 없어 동일한 액션의 반복 실행을 차단했습니다.",
                    before_url=observation.url,
                    after_url=observation.url,
                )
            else:
                result = self.executor.execute(decision.action)
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
        if not history:
            return False
        previous = history[-1]
        return previous.action == action and (
            not previous.result.ok or not previous.result.state_changed
        )
