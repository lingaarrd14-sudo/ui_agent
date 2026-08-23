# UI Agent

비전 모델이 브라우저 화면을 보고 클릭, 입력, 키보드 조작, 스크롤을 수행하는
최소 구성의 웹 UI 에이전트입니다. Python Playwright로 브라우저를 제어하고,
OpenAI Responses API의 구조화된 출력으로 매 단계의 액션을 결정합니다.

## 동작 방식

에이전트는 최대 `MAX_STEPS`만큼 다음 과정을 반복합니다.

1. 현재 페이지의 URL과 스크린샷을 관찰합니다.
2. 목표, 현재 화면, 최근 실행 기록을 모델에 전달합니다.
3. 모델이 반환한 액션 하나를 Playwright로 실행합니다.
4. 실행 전후 화면을 비교하고, 변화가 없는데 같은 액션이 반복되면 차단합니다.
5. 모델이 완료를 선언하거나 목표 도메인에 도착하면 종료합니다.

새 탭이나 팝업이 열리면 가장 최근에 열린 페이지로 제어 대상을 전환합니다.

## 프로젝트 구조

| 경로 | 역할 |
| --- | --- |
| `agent.py` | 설정, OpenAI 클라이언트, 브라우저와 각 컴포넌트를 조립하는 실행 진입점 |
| `ui_agent/config.py` | `.env`와 환경변수 설정 로드 |
| `ui_agent/models.py` | 관찰, 액션, 실행 결과를 표현하는 Pydantic 모델 |
| `ui_agent/perception.py` | 현재 페이지의 URL과 스크린샷 수집 |
| `ui_agent/policy.py` | 비전 모델에 다음 액션 요청 |
| `ui_agent/executor.py` | 모델이 선택한 액션을 Playwright로 실행 |
| `ui_agent/browser.py` | 현재 페이지 및 새 탭 추적 |
| `ui_agent/runner.py` | 관찰-판단-실행 루프와 종료 조건 관리 |
| `tests/test_agent.py` | 스키마, 새 탭, 종료 및 반복 액션 차단 단위 테스트 |

## 설치 및 실행

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
python agent.py
```

프로젝트 루트의 `.env`에 API 키와 실행 설정을 입력합니다.

```dotenv
OPENAI_API_KEY=your-api-key
TASK=NAVER 뉴스로 이동해서 경제 뉴스 하나를 클릭하세요
START_URL=https://www.naver.com
MODEL=gpt-5.6-terra
MAX_STEPS=15
```

지원하는 환경변수는 다음과 같습니다.

| 이름 | 설명 | 기본값 |
| --- | --- | --- |
| `OPENAI_API_KEY` | API 인증 키 | 없음 |
| `OPENAI_BASE_URL` | OpenAI 호환 API 주소 | `https://factchat-cloud.mindlogic.ai/v1/gateway` |
| `MODEL` | 액션을 결정할 비전 모델 | `gpt-5.6-terra` |
| `TASK` | 브라우저에서 수행할 목표 | NAVER 뉴스 예제 |
| `START_URL` | 처음 열 페이지 | `https://www.naver.com` |
| `TARGET_DOMAIN` | 도착만으로 성공 처리할 선택적 도메인 | 없음 |
| `MAX_STEPS` | 최대 액션 결정 횟수 | `15` |

> [!CAUTION]
> `OPENAI_BASE_URL`을 지정하지 않으면 코드에 설정된 Mindlogic 게이트웨이를
> 사용합니다. API 키를 입력하기 전에 신뢰할 수 있는 엔드포인트인지 확인하세요.
> OpenAI SDK의 기본 엔드포인트를 사용하려면 `.env`에서
> `OPENAI_BASE_URL=`처럼 빈 값으로 명시할 수 있습니다.

`TARGET_DOMAIN=openai.com`처럼 설정하면 해당 도메인이나 하위 도메인에
도착하는 즉시 성공으로 종료합니다. 페이지 안에서 추가 작업이 필요한 목표에는
이 값을 사용하지 않거나, 최종 도착 도메인만 지정하세요.

## 테스트

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

단위 테스트는 실제 API 호출이나 실제 로그인 같은 외부 작업을 수행하지
않습니다. 실제 에이전트 실행은 브라우저를 열고 모델 API를 호출합니다.
