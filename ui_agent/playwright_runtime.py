"""단독 실행용 Playwright 페이지 관리, 화면 관찰, 액션 실행 경계."""

import base64

from playwright.sync_api import BrowserContext, Error as PlaywrightError, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from .models import (
    Action,
    ClickAction,
    ExecutionResult,
    Observation,
    PressAction,
    ScrollAction,
    TypeAction,
)


class BrowserSession:
    """팝업과 새 탭을 포함해 현재 제어할 페이지를 추적한다."""

    def __init__(self, context: BrowserContext, page: Page):
        self.context = context
        self._page = page
        self._known_pages = list(context.pages)

    def current_page(self) -> Page:
        """가장 최근에 열린 유효한 페이지를 활성 페이지로 반환한다."""
        pages = [page for page in self.context.pages if not page.is_closed()]
        if not pages:
            raise RuntimeError("열려 있는 브라우저 페이지가 없습니다.")

        # 직전 확인 이후 생긴 팝업이나 새 탭을 우선 제어 대상으로 삼는다.
        new_pages = [page for page in pages if page not in self._known_pages]
        if new_pages:
            self._page = new_pages[-1]
        elif self._page.is_closed():
            self._page = pages[-1]

        self._known_pages = pages
        try:
            self._page.bring_to_front()
        except PlaywrightError:
            # 포커스 전환 실패가 전체 에이전트 실행을 중단시키지는 않게 한다.
            pass
        return self._page


class ScreenshotPerception:
    """활성 페이지의 URL과 PNG 스크린샷을 수집한다."""

    def __init__(self, session: BrowserSession, width: int, height: int):
        self.session = session
        self.width = width
        self.height = height

    def capture(self) -> Observation:
        """현재 페이지를 캡처해 직렬화 가능한 관찰 객체로 반환한다."""
        page = self.session.current_page()
        screenshot = base64.b64encode(page.screenshot()).decode("ascii")
        return Observation(
            url=page.url,
            screenshot_base64=screenshot,
            viewport_width=self.width,
            viewport_height=self.height,
        )


class PlaywrightExecutor:
    """현재 페이지에서 구조화된 브라우저 액션을 실행한다."""

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
            self._apply_action(page, action)
            active_page = self._wait_until_settled(page)
            return ExecutionResult(ok=True, opened_new_page=active_page is not page)
        except (PlaywrightError, ValueError) as exc:
            return ExecutionResult(ok=False, message=str(exc))

    def _apply_action(self, page: Page, action: Action) -> None:
        """액션 모델 하나를 대응하는 Playwright 호출로 변환한다."""
        if isinstance(action, ClickAction):
            if action.x >= self.width or action.y >= self.height:
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

    def _wait_until_settled(self, page: Page) -> Page:
        """애니메이션과 선택적 페이지 탐색이 끝날 시간을 준다."""
        try:
            page.wait_for_timeout(self.settle_ms)
        except PlaywrightError:
            # 액션 도중 기존 탭이 닫혀도 새로 열린 탭을 계속 확인한다.
            pass

        active_page = self.session.current_page()
        try:
            # 모든 액션이 탐색을 일으키는 것은 아니므로 타임아웃은 허용한다.
            active_page.wait_for_load_state("domcontentloaded", timeout=3000)
        except PlaywrightTimeoutError:
            pass
        return active_page
