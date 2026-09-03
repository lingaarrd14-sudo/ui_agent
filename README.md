# UI Agent

원시 브라우저 스크린샷을 보고 좌표 클릭, 입력, 키 입력, 스크롤을 수행하는
vision-only 웹 에이전트입니다. 모델 입력에는 SoM 표시, DOM/접근성 트리,
이미지 캡션이 들어가지 않습니다. VisualWebArena task에 원래 포함된 reference
image만 별도의 원본 이미지로 전달합니다.

## 신뢰성 경계

실행 흐름은 VisualWebArena의 `observe → decide → act → observe → evaluate` 구조를
따르되, 액션 결정과 안전 검증은 `ui_agent`가 소유합니다.

- 모델은 Pydantic 스키마에 맞는 액션 하나만 반환합니다.
- 클릭은 현재 뷰포트 안의 좌표만 허용합니다.
- 실행 전후 화면을 축소 비교해 렌더링 잡음과 실제 변화를 구분합니다.
- 변화가 없었던 동일 액션, 12px 이내로 좌표만 바꾼 재클릭, 이전 화면으로
  돌아온 뒤의 동일 액션을 실행 전에 거절합니다.
- 한 단계에서 최대 3번 다른 액션을 요청하고, 계속 같은 제안을 하면 실제로
  실행하지 않고 `blocked`로 종료합니다. 무한 재시도는 없습니다.
- VisualWebArena 결과의 성공 여부는 모델의 자기 선언이 아니라 공식 evaluator가
  최종 판정합니다.

## 구조

| 경로 | 역할 |
| --- | --- |
| `agent.py` | 단독 실행 설정과 컴포넌트 조립 |
| `ui_agent/models.py` | 관찰, 구조화 액션, 실행 결과 계약 |
| `ui_agent/controller.py` | 시각 상태 비교, 제안 검증, 반복·순환 차단 |
| `ui_agent/policy.py` | OpenAI Responses API vision 정책 |
| `ui_agent/playwright_runtime.py` | Playwright 페이지 관리, 화면 관찰, 액션 실행 |
| `ui_agent/runner.py` | 단독 실행 루프 |
| `ui_agent/vwa_adapter.py` | 공통 액션과 VWA 저수준 액션 사이의 얇은 변환 계층 |
| `ui_agent/vwa_config.py` | VWA 환경변수, 원본 task 변환과 사전 검증 |
| `ui_agent/vwa_runtime.py` | VWA import, 인증, 액션 factory, 평가 captioner 경계 |
| `ui_agent/vwa_tasks.py` | task 실행, 공식 평가, task별 결과 저장 |
| `ui_agent/vwa_results.py` | 재현 가능한 run identity와 append-only 결과 관리 |
| `vwa_benchmark.py` | CLI 설정과 VWA task/auth/browser 컴포넌트 조립 |
| `summarize_benchmark.py` | 재시도 결과를 중복 제거한 요약 출력 |

`vwa_benchmark.py`는 VisualWebArena의 `browser_env`, `auto_login`, 공식
`evaluation_harness`만 사용합니다. VWA의 `PromptAgent`, SoM 프롬프트,
토크나이저, caption 기반 observation은 가져오지 않습니다. 단,
`page_image_query` task의 공식 점수를 계산할 때 VWA evaluator가 BLIP-2를 사용할
수 있습니다. 이것은 평가 단계 전용이며 에이전트 입력에는 들어가지 않습니다.

생성 config, 로그인 state, trace, 점수는 전부 지정한 `result-dir` 아래에 기록되며
`visualwebarena` checkout에는 쓰지 않습니다.

## 환경 설정

Python 3.11 가상환경을 프로젝트 안에 별도로 만듭니다. WebArena 가상환경은
사용하지 않습니다.

```bash
cd /home/default/agent/ui_agent
python3.11 -m venv --prompt ui_agent .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
python -m nltk.downloader punkt punkt_tab
```

`ui_agent/.env` 예시:

```dotenv
OPENAI_API_KEY=your-api-key
MODEL=gpt-5.6-terra
```

공식 OpenAI API를 쓸 때는 `OPENAI_BASE_URL`과 `BASE_URL`을 넣지 않으면 됩니다.
호환 게이트웨이를 쓸 때만 둘 중 하나를 설정합니다.

```dotenv
OPENAI_BASE_URL=https://example-gateway.invalid/v1
```

단독 실행에서 사용할 수 있는 환경변수는 `TASK`, `START_URL`, `TARGET_DOMAIN`,
`MAX_STEPS`, `VIEWPORT_WIDTH`, `VIEWPORT_HEIGHT`입니다.

## VisualWebArena 실행

먼저 API 비용이나 브라우저 실행 없이 VWA import와 task 계약을 확인합니다.

```bash
python vwa_benchmark.py \
  --domain reddit \
  --start 0 \
  --end 1 \
  --viewport-width 1280 \
  --viewport-height 720 \
  --result-dir benchmark_results/reddit_validate \
  --validate-only
```

그다음 task 하나만 smoke test합니다.

```bash
python vwa_benchmark.py \
  --domain reddit \
  --start 0 \
  --end 1 \
  --model gpt-5.6-terra \
  --max-steps 15 \
  --viewport-width 1280 \
  --viewport-height 720 \
  --result-dir benchmark_results/reddit_smoke
```


Reddit 전체 210개 task를 같은 화면 크기로 실행하려면 다음 범위를 사용합니다.

```bash
python vwa_benchmark.py \
  --domain reddit \
  --start 0 \
  --end 210 \
  --model gpt-5.6-terra \
  --max-steps 30 \
  --viewport-width 1280 \
  --viewport-height 720 \
  --result-dir benchmark_results/reddit_1280x720
```

`--end`는 포함되지 않는 Python slice의 끝값입니다. 화면 크기 비교 실험에서는
모델, task 범위, `max-steps`, 사이트 초기 상태를 고정하고 viewport와
`result-dir`만 바꾸세요.

VWA task 중 일부는 사이트 상태를 변경합니다. 이 runner는 벤치마크 도중 Docker
컨테이너를 암묵적으로 삭제하거나 초기화하지 않고, 선택 범위의 mutable task 수를
시작 전에 경고합니다. 비교할 두 run 앞에서 동일한 VWA reset 절차를 적용하고,
긴 run을 여러 batch로 나눈다면 두 조건 모두 같은 경계에서 reset해야 합니다.
Reddit 전체 run 전에는 VWA 원본 스크립트를 별도로 실행합니다.

```bash
cd /home/default/agent/visualwebarena
bash scripts/reset_reddit.sh
cd /home/default/agent/ui_agent
```

기존 result directory를 컨테이너 reset 뒤 이어서 쓸 때는 로그인 state도 다시
만들도록 같은 benchmark 명령에 `--refresh-auth`를 추가하세요. 새 result directory는
auth 파일이 없으므로 자동으로 새 로그인 state를 만듭니다.

중단된 결과 디렉터리로 같은 명령을 다시 실행하면 점수가 기록된 task는 건너뛰고
오류 task만 재시도합니다. `results.jsonl`은 append-only이고 `summary.json`과
요약 스크립트는 task별 최신 기록만 사용합니다.

```bash
python summarize_benchmark.py benchmark_results/reddit_1280x720
```

`run_config.json`에는 VWA/agent Git 상태, uncommitted source까지 반영한 SHA-256,
정확한 프롬프트, 모델, API endpoint 종류, viewport가 저장됩니다. API key는
저장하지 않습니다.

## 테스트

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

단위 테스트는 API, 브라우저, Docker를 사용하지 않습니다.
