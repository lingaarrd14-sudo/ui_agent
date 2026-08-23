"""에이전트가 관찰하고 조작할 활성 Playwright 페이지를 관리한다."""

from playwright.sync_api import BrowserContext, Error as PlaywrightError, Page


class BrowserSession:
    """팝업과 새 탭을 포함해 현재 제어할 페이지를 추적한다."""

    def __init__(self, context: BrowserContext, page: Page):
        """초기 페이지와 이미 열려 있는 페이지 목록을 기억한다."""
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
            # 포커스 전환 실패가 전체 에이전트 실행을 중단시키지는 않게 한다.
            self._page.bring_to_front()
        except PlaywrightError:
            pass
        return self._page
