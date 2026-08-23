"""UI 에이전트의 컴포넌트를 조립하고 브라우저 실행을 관리한다."""

from openai import OpenAI
from playwright.sync_api import sync_playwright

from ui_agent.browser import BrowserSession
from ui_agent.config import Settings
from ui_agent.executor import PlaywrightExecutor
from ui_agent.perception import ScreenshotPerception
from ui_agent.policy import OpenAIVisionPolicy
from ui_agent.runner import AgentRunner, DomainEvaluator


def main() -> None:
    """환경설정을 읽고 하나의 UI 에이전트 실행을 시작한다."""
    settings = Settings.from_env()

    # base_url이 비어 있으면 OpenAI SDK가 제공하는 기본 엔드포인트를 사용한다.
    client_options = {"api_key": settings.api_key}
    if settings.base_url:
        client_options["base_url"] = settings.base_url
    client = OpenAI(**client_options)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page(
            viewport={
                "width": settings.viewport_width,
                "height": settings.viewport_height,
            }
        )
        page.goto(settings.start_url)

        # 이후 팝업이나 새 탭이 생겨도 모든 컴포넌트가 같은 세션을 공유한다.
        session = BrowserSession(page.context, page)

        # 관찰, 판단, 실행, 성공 판정을 명시적으로 분리해 각각 교체할 수 있다.
        runner = AgentRunner(
            perception=ScreenshotPerception(
                session, settings.viewport_width, settings.viewport_height
            ),
            policy=OpenAIVisionPolicy(client, settings.model),
            executor=PlaywrightExecutor(
                session, settings.viewport_width, settings.viewport_height
            ),
            evaluator=DomainEvaluator(settings.target_domain),
        )
        result = runner.run(settings.task, settings.max_steps)
        print(f"종료: {result.status} - {result.summary}")

        # 실행 결과를 사용자가 확인할 때까지 headed 브라우저를 유지한다.
        input("Enter를 누르면 브라우저를 종료합니다.")
        browser.close()


if __name__ == "__main__":
    main()
