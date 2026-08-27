"""스크린샷을 안정적으로 비교하기 위한 작은 시각 상태 표현."""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from .models import Observation


SAMPLE_SIZE = (32, 32)
DEFAULT_CHANGE_THRESHOLD = 0.003


@dataclass(frozen=True)
class VisualState:
    """URL과 축소된 회색조 화면으로 구성한 비교 가능한 브라우저 상태."""

    url: str
    pixels: bytes

    @classmethod
    def from_observation(cls, observation: Observation) -> "VisualState":
        """큰 원본 스크린샷을 보관하지 않고 비교용 표본만 만든다."""
        try:
            raw = base64.b64decode(observation.screenshot_base64, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                sampled = image.convert("L").resize(SAMPLE_SIZE)
                pixels = sampled.tobytes()
        except (ValueError, UnidentifiedImageError, OSError):
            # 테스트 대역이나 손상된 관찰도 결정적으로 비교할 수 있게 한다.
            pixels = hashlib.sha256(
                observation.screenshot_base64.encode("utf-8")
            ).digest()
        return cls(url=observation.url, pixels=pixels)

    def distance(self, other: "VisualState") -> float:
        """두 축소 화면의 평균 절대 픽셀 차이를 0~1 범위로 반환한다."""
        if len(self.pixels) != len(other.pixels):
            return 1.0
        if not self.pixels:
            return 0.0
        total = sum(abs(left - right) for left, right in zip(self.pixels, other.pixels))
        return total / (len(self.pixels) * 255)

    def equivalent(
        self,
        other: "VisualState",
        threshold: float = DEFAULT_CHANGE_THRESHOLD,
    ) -> bool:
        """URL이 같고 작은 렌더링 잡음 외에는 같은 화면인지 판정한다."""
        return self.url == other.url and self.distance(other) <= threshold


def observation_changed(
    before: Observation,
    after: Observation,
    threshold: float = DEFAULT_CHANGE_THRESHOLD,
) -> bool:
    """URL 또는 의미 있는 화면 픽셀 변화가 있었는지 판정한다."""
    return not VisualState.from_observation(before).equivalent(
        VisualState.from_observation(after), threshold
    )
