from openai import OpenAI
from playwright.sync_api import sync_playwright

from ui_agent.browser import BrowserSession
from ui_agent.config import Settings
from ui_agent.executor import PlaywrightExecutor
from ui_agent.perception import ScreenshotPerception
from ui_agent.policy import OpenAIVisionPolicy
from ui_agent.runner import AgentRunner, DomainEvaluator


def main() -> None:
    settings = Settings.from_env()
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
        session = BrowserSession(page.context, page)

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

        input("Enter를 누르면 브라우저를 종료합니다.")
        browser.close()


if __name__ == "__main__":
    main()
