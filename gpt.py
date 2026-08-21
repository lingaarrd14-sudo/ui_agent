import base64
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-5.6-sol")
START_URL = os.getenv("START_URL", "https://www.naver.com")
TASK = os.getenv("TASK", "NAVER 뉴스로 이동해서 트럼프 관련 경제 뉴스 하나를 클릭하세요")
MAX_STEPS = 15
VIEWPORT = {"width": 1280, "height": 720}

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://factchat-cloud.mindlogic.ai/v1/gateway"),
)


def next_action(page, history):
    screenshot = base64.b64encode(page.screenshot()).decode()
    prompt = f"""You are a visual web agent.
Goal: {TASK}
Current URL: {page.url}
Previous actions: {json.dumps(history[-5:], ensure_ascii=False)}

Choose exactly one next action from:
{{"action":"click","x":0,"y":0}}
{{"action":"type","text":"..."}}
{{"action":"press","key":"Enter"}}
{{"action":"scroll","delta":500}}
{{"action":"done"}}

Use only the screenshot. Return JSON only.
Click bounds: x=0..{VIEWPORT['width']}, y=0..{VIEWPORT['height']}.
"""
    response = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot}",
                    "detail": "high",
                },
            ],
        }],
    )
    text = response.output_text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)


def execute(page, action):
    kind = action["action"]
    if kind == "click":
        x, y = action["x"], action["y"]
        if not (0 <= x <= VIEWPORT["width"] and 0 <= y <= VIEWPORT["height"]):
            raise ValueError(f"잘못된 클릭 좌표: ({x}, {y})")
        page.mouse.click(x, y)
    elif kind == "type":
        page.keyboard.type(action["text"])
    elif kind == "press":
        page.keyboard.press(action["key"])
    elif kind == "scroll":
        page.mouse.wheel(0, action["delta"])
    elif kind == "done":
        return False
    else:
        raise ValueError(f"알 수 없는 동작: {kind}")
    return True


def main():
    history = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(START_URL)

        for step in range(1, MAX_STEPS + 1):
            action = next_action(page, history)
            history.append(action)
            print(f"{step}: {action}")
            if not execute(page, action):
                break
            page.wait_for_timeout(1200)

        input("Enter를 누르면 브라우저를 종료합니다.")
        browser.close()


if __name__ == "__main__":
    main()
