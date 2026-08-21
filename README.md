# UI Agent

화면을 보고 클릭·입력하는 최소 구성의 웹 에이전트입니다.

## 실행

```powershell
pip install -r requirements.txt
playwright install chromium
python gpt.py
```

`.env`에 `OPENAI_API_KEY`를 설정하세요. 필요하면 `TASK`, `START_URL`, `MODEL`, `OPENAI_BASE_URL`도 변경할 수 있습니다. 목표 사이트가 명확하면 `TARGET_DOMAIN=openai.com`처럼 지정하면 도착 즉시 종료합니다.
