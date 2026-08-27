"""환경변수에서 UI 에이전트 실행 설정을 읽는다."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """한 번의 에이전트 실행에 사용하는 변경 불가능한 설정 모음."""

    api_key: str | None
    base_url: str | None
    model: str
    task: str
    start_url: str
    target_domain: str | None
    max_steps: int = 15
    viewport_width: int = 1280
    viewport_height: int = 720

    @classmethod
    def from_env(cls) -> "Settings":
        """프로젝트의 `.env`와 현재 프로세스 환경에서 설정을 생성한다."""
        load_dotenv()
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL"),
            model=os.getenv("MODEL", "gpt-5.6-terra"),
            task=os.getenv(
                "TASK",
                "경제 뉴스 좀 찾아줘",
            ),
            start_url=os.getenv("START_URL", "http://localhost:9999/"),
            target_domain=os.getenv("TARGET_DOMAIN"),
            max_steps=int(os.getenv("MAX_STEPS", "15")),
            viewport_width=int(os.getenv("VIEWPORT_WIDTH", "1280")),
            viewport_height=int(os.getenv("VIEWPORT_HEIGHT", "720")),
        )
