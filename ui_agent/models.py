from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClickAction(StrictModel):
    kind: Literal["click"]
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class TypeAction(StrictModel):
    kind: Literal["type"]
    text: str


class PressAction(StrictModel):
    kind: Literal["press"]
    key: str


class ScrollAction(StrictModel):
    kind: Literal["scroll"]
    delta_y: int = Field(ge=-2000, le=2000)


class DoneAction(StrictModel):
    kind: Literal["done"]
    status: Literal["success", "blocked"]
    summary: str


Action = ClickAction | TypeAction | PressAction | ScrollAction | DoneAction


class Decision(StrictModel):
    action: Action
    expected_outcome: str


class Observation(StrictModel):
    url: str
    screenshot_base64: str
    viewport_width: int
    viewport_height: int


class ExecutionResult(StrictModel):
    ok: bool
    message: str = ""
    before_url: str = ""
    after_url: str = ""
    state_changed: bool = False
    opened_new_page: bool = False


class StepRecord(StrictModel):
    url: str
    action: Action
    expected_outcome: str
    result: ExecutionResult


class RunResult(StrictModel):
    status: Literal["success", "blocked", "max_steps"]
    summary: str
    steps: list[StepRecord]
