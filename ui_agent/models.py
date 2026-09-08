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


class TypeAction(StrictModel):
    """현재 포커스된 요소에 텍스트를 입력한다."""

    kind: Literal["type"]
    text: str


class PressAction(StrictModel):
    """Playwright가 이해하는 키 이름을 눌러 보낸다."""

    kind: Literal["press"]
    key: str


class ScrollAction(StrictModel):
    """세로 방향으로 제한된 거리만큼 스크롤한다."""

    kind: Literal["scroll"]
    delta_y: int = Field(ge=-2000, le=2000)


class DoneAction(StrictModel):
    """모델이 작업 성공 또는 진행 불가를 선언한다."""

    kind: Literal["done"]
    status: Literal["success", "blocked"]
    summary: str


# discriminator를 강제로 지정하지 않아 OpenAI 호환 게이트웨이에서도
# `anyOf` 기반 Structured Outputs 스키마를 사용할 수 있게 한다.
Action = ClickAction | TypeAction | PressAction | ScrollAction | DoneAction


class Decision(StrictModel):
    """모델이 선택한 단일 액션과 그 액션에서 기대하는 결과."""

    action: Action
    expected_outcome: str


class Observation(StrictModel):
    """판단 시점의 URL, 화면 이미지, 좌표계 크기."""

    url: str
    screenshot_base64: str
    viewport_width: int
    viewport_height: int


class ExecutionResult(StrictModel):
    """액션 실행 성공 여부와 실행 후 URL, 새 페이지 생성 여부."""

    ok: bool
    message: str = ""
    after_url: str = ""
    opened_new_page: bool = False


class StepRecord(StrictModel):
    """다음 판단에 제공할 한 단계의 결정 및 실행 이력."""

    url: str
    action: Action
    repeat_count: int = 1
    expected_outcome: str
    result: ExecutionResult


class RunResult(StrictModel):
    """에이전트 종료 상태와 전체 실행 이력."""

    status: Literal["success", "blocked", "max_steps"]
    summary: str
    steps: list[StepRecord]
