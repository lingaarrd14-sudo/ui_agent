import base64

from .browser import BrowserSession
from .models import Observation


class ScreenshotPerception:
    def __init__(self, session: BrowserSession, width: int, height: int):
        self.session = session
        self.width = width
        self.height = height

    def capture(self) -> Observation:
        page = self.session.current_page()
        screenshot = base64.b64encode(page.screenshot()).decode("ascii")
        return Observation(
            url=page.url,
            screenshot_base64=screenshot,
            viewport_width=self.width,
            viewport_height=self.height,
        )
