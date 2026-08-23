from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from .browser import BrowserSession
from .models import (
    Action,
    ClickAction,
    ExecutionResult,
    PressAction,
    ScrollAction,
    TypeAction,
)


class PlaywrightExecutor:
    def __init__(
        self, session: BrowserSession, width: int, height: int, settle_ms: int = 800
    ):
        self.session = session
        self.width = width
        self.height = height
        self.settle_ms = settle_ms

    def execute(self, action: Action) -> ExecutionResult:
        page = self.session.current_page()
        try:
            if isinstance(action, ClickAction):
                if action.x > self.width or action.y > self.height:
                    raise ValueError(f"잘못된 클릭 좌표: ({action.x}, {action.y})")
                page.mouse.click(action.x, action.y)
            elif isinstance(action, TypeAction):
                page.keyboard.type(action.text)
            elif isinstance(action, PressAction):
                page.keyboard.press(action.key)
            elif isinstance(action, ScrollAction):
                page.mouse.wheel(0, action.delta_y)
            else:
                raise ValueError(f"실행할 수 없는 액션: {action.kind}")

            try:
                page.wait_for_timeout(self.settle_ms)
            except PlaywrightError:
                pass

            active_page = self.session.current_page()
            try:
                active_page.wait_for_load_state("domcontentloaded", timeout=3000)
            except PlaywrightTimeoutError:
                pass
            return ExecutionResult(ok=True, opened_new_page=active_page is not page)
        except (PlaywrightError, ValueError) as exc:
            return ExecutionResult(ok=False, message=str(exc))
