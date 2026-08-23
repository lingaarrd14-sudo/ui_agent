# UI Agent

GPT VLM이 화면을 보고 클릭·입력하는 최소 구성의 웹 에이전트입니다.

`ui_agent/` 아래에서 화면 관찰, 모델 판단, 액션 실행, 성공 판정을 분리했습니다.

## 실행

```powershell
pip install -r requirements.txt
playwright install chromium
python gpt.py
```

`.env`에 `OPENAI_API_KEY`를 설정하세요. 필요하면 `TASK`, `START_URL`, `MODEL`, `OPENAI_BASE_URL`, `MAX_STEPS`도 변경할 수 있습니다. 목표 사이트가 명확하면 `TARGET_DOMAIN=openai.com`처럼 지정하면 도착 즉시 종료합니다.

## 테스트

```powershell
python -m unittest discover -s tests -p "test_*.py"
```
