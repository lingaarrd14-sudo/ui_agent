from playwright.sync_api import BrowserContext, Error as PlaywrightError, Page


class BrowserSession:
    """Tracks the page the agent should currently observe and control."""

    def __init__(self, context: BrowserContext, page: Page):
        self.context = context
        self._page = page
        self._known_pages = list(context.pages)

    def current_page(self) -> Page:
        pages = [page for page in self.context.pages if not page.is_closed()]
        if not pages:
            raise RuntimeError("열려 있는 브라우저 페이지가 없습니다.")

        new_pages = [page for page in pages if page not in self._known_pages]
        if new_pages:
            self._page = new_pages[-1]
        elif self._page.is_closed():
            self._page = pages[-1]

        self._known_pages = pages
        try:
            self._page.bring_to_front()
        except PlaywrightError:
            pass
        return self._page
