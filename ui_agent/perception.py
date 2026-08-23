"""현재 브라우저 화면을 비전 모델이 처리할 관찰 데이터로 변환한다."""

import base64

from .browser import BrowserSession
from .models import Observation


class ScreenshotPerception:
    """활성 페이지의 URL과 PNG 스크린샷을 수집한다."""

    def __init__(self, session: BrowserSession, width: int, height: int):
        self.session = session
        self.width = width
        self.height = height

    def capture(self) -> Observation:
        """현재 페이지를 캡처해 직렬화 가능한 관찰 객체로 반환한다."""
        page = self.session.current_page()
        # Responses API의 data URL에 넣을 수 있도록 원시 PNG를 base64로 인코딩한다.
        screenshot = base64.b64encode(page.screenshot()).decode("ascii")
        return Observation(
            url=page.url,
            screenshot_base64=screenshot,
            viewport_width=self.width,
            viewport_height=self.height,
        )
