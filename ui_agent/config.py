import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
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
        load_dotenv()
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://factchat-cloud.mindlogic.ai/v1/gateway",
            ),
            model=os.getenv("MODEL", "gpt-5.6-terra"),
            task=os.getenv(
                "TASK",
                "NAVER 뉴스로 이동해서 트럼프 관련 경제 뉴스 하나를 클릭하세요",
            ),
            start_url=os.getenv("START_URL", "https://www.naver.com"),
            target_domain=os.getenv("TARGET_DOMAIN"),
            max_steps=int(os.getenv("MAX_STEPS", "15")),
        )
