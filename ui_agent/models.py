"""모델 판단과 브라우저 실행 사이에서 교환하는 데이터 계약을 정의한다."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """모델이 정의되지 않은 필드를 임의로 반환하지 못하게 하는 기반 모델."""

    model_config = ConfigDict(extra="forbid")


class ClickAction(StrictModel):
    """뷰포트의 절대 좌표를 클릭한다."""

    kind: Literal["click"]
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class HoverAction(StrictModel):
    """뷰포트의 절대 좌표로 마우스를 이동한다."""

    kind: Literal["hover"]
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class TypeAction(StrictModel):
    """현재 포커스된 요소에 텍스트를 입력한다."""

    kind: Literal["type"]
    text: str


class PressAction(StrictModel):
    """Playwright가 이해하는 키 이름을 눌러 보낸다."""

    kind: Literal["press"]
    key_comb: str


class ScrollAction(StrictModel):
    """VisualWebArena와 같이 한 뷰포트 위나 아래로 스크롤한다."""

    kind: Literal["scroll"]
    direction: Literal["up", "down"]


class GoBackAction(StrictModel):
    kind: Literal["go_back"]


class GoForwardAction(StrictModel):
    kind: Literal["go_forward"]


class StopAction(StrictModel):
    """VisualWebArena STOP과 내부 종료 상태를 함께 표현한다."""

    kind: Literal["stop"]
    status: Literal["success", "blocked"]
    answer: str


# discriminator를 강제로 지정하지 않아 OpenAI 호환 게이트웨이에서도
# `anyOf` 기반 Structured Outputs 스키마를 사용할 수 있게 한다.
Action = (
    ClickAction
    | HoverAction
    | TypeAction
    | PressAction
    | ScrollAction
    | GoBackAction
    | GoForwardAction
    | StopAction
)


class Decision(StrictModel):
    """현재 화면 요약, 단일 액션, 그 액션에서 기대하는 결과."""

    state_summary: str
    action: Action
    expected_outcome: str


class Observation(StrictModel):
    """판단 시점의 화면 이미지와 좌표계 크기."""

    screenshot_base64: str
    viewport_width: int
    viewport_height: int


class ExecutionResult(StrictModel):
    """다음 모델 판단에 공개할 액션 실행 성공 여부."""

    ok: bool


class StepRecord(StrictModel):
    """다음 판단에 제공할 한 단계의 결정 및 실행 이력."""

    state_summary: str
    action: Action
    repeat_count: int = 1
    expected_outcome: str
    result: ExecutionResult
