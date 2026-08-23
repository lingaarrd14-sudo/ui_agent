"""구조화된 에이전트 액션을 Playwright 브라우저 조작으로 변환한다."""

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
    """현재 페이지에서 클릭, 입력, 키 입력, 스크롤 액션을 실행한다."""

    def __init__(
        self, session: BrowserSession, width: int, height: int, settle_ms: int = 800
    ):
        self.session = session
        self.width = width
        self.height = height
        self.settle_ms = settle_ms

    def execute(self, action: Action) -> ExecutionResult:
        """액션 하나를 실행하고 오류 및 새 페이지 생성 여부를 반환한다."""
        page = self.session.current_page()
        try:
            if isinstance(action, ClickAction):
                if action.x > self.width or action.y > self.height:
                    raise ValueError(f"잘못된 클릭 좌표: ({action.x}, {action.y})")
                page.mouse.click(action.x, action.y)
            elif isinstance(action, TypeAction):
                # 모델이 먼저 클릭해 포커스한 요소에 텍스트를 입력한다.
                page.keyboard.type(action.text)
            elif isinstance(action, PressAction):
                page.keyboard.press(action.key)
            elif isinstance(action, ScrollAction):
                page.mouse.wheel(0, action.delta_y)
            else:
                raise ValueError(f"실행할 수 없는 액션: {action.kind}")

            # 즉시 다음 화면을 찍으면 애니메이션 중간 상태를 볼 수 있어 잠시 기다린다.
            try:
                page.wait_for_timeout(self.settle_ms)
            except PlaywrightError:
                pass

            active_page = self.session.current_page()
            try:
                # 모든 액션이 탐색을 일으키는 것은 아니므로 타임아웃은 허용한다.
                active_page.wait_for_load_state("domcontentloaded", timeout=3000)
            except PlaywrightTimeoutError:
                pass
            return ExecutionResult(ok=True, opened_new_page=active_page is not page)
        except (PlaywrightError, ValueError) as exc:
            return ExecutionResult(ok=False, message=str(exc))
