# UI Agent

원시 브라우저 스크린샷을 보고 좌표 클릭, 입력, 키 입력, 스크롤을 수행하는
vision-only 웹 에이전트입니다. 모델 입력에는 SoM 표시, DOM/접근성 트리,
이미지 캡션이 들어가지 않습니다. VisualWebArena task에 원래 포함된 reference
image만 별도의 원본 이미지로 전달합니다.

## 신뢰성 경계

실행 흐름은 VisualWebArena의 `observe → decide → act → observe → evaluate` 구조를
따르되, 액션 결정과 유효성 검증은 `ui_agent`가 소유합니다.

- 모델은 Pydantic 스키마에 맞는 액션 하나만 반환합니다.
- 클릭은 현재 뷰포트 안의 좌표만 허용합니다.
- 단계당 모델을 한 번 호출하고, 실제 실행 이력에서 동일 액션이 기본 5회 연속
  나타나면 다음 모델 호출 전에 `blocked`로 종료합니다.
- 클릭은 정수 좌표, 입력은 텍스트, 키 입력은 키 문자열, 스크롤은 방향으로
  동일성을 비교합니다. 화면 무변화나 이전 화면 재방문만으로 차단하지 않습니다.
- 화면 변화는 다음 스크린샷에서 모델이 판단합니다. 별도 픽셀 비교나 순환 추적,
  거절 피드백을 넣은 재제안 호출은 없습니다. 유효하지 않은 액션은 실행 없이 종료합니다.
- VisualWebArena 결과의 성공 여부는 모델의 자기 선언이 아니라 공식 evaluator가
  최종 판정합니다.

## 구조

| 경로 | 역할 |
| --- | --- |
| `agent.py` | 단독 실행 설정과 컴포넌트 조립 |
| `ui_agent/models.py` | 관찰, 구조화 액션, 실행 결과 계약 |
| `ui_agent/controller.py` | 실행 이력 기반 반복 조기 종료, 액션 유효성 검증 |
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

## 반복 처리 변경 근거

참고한 로컬 VisualWebArena commit은 `f29e2a9273278e02a7e4b9da05987612425766d0`입니다.

- [`run.py:198`](../visualwebarena/run.py#L198)의 `early_stop`은 화면 비교 없이
  실행된 액션을 검사합니다. 기본 반복 임계치는 5이며, 다음 액션 요청 전에 검사합니다.
- [`browser_env/actions.py:349`](../visualwebarena/browser_env/actions.py#L349)의
  `is_equivalent`에 맞춰 스크롤은 방향, 키와 입력은 원문으로 비교합니다.
  VWA는 정규화된 클릭 좌표에 `np.allclose`를 쓰지만 여기서는 정수 픽셀을 정확히
  비교합니다. 기존 12px 버킷은 제거했습니다.
- VWA의 고수준 `TYPE`은 전체 이력에서 같은 대상을 누적하지만, 이 어댑터의
  `type`은 `KEYBOARD_TYPE`입니다. 따라서 다른 저수준 액션처럼 **연속** 횟수를 셉니다.
  Structured Outputs 파싱 실패는 기존 오류 경로를 유지하며 VWA의 `NONE` 액션
  누적 로직은 도입하지 않습니다.
- [`p_multimodal_cot_id_actree_3s.py`](../visualwebarena/agent/prompts/raw/p_multimodal_cot_id_actree_3s.py)의
  현재 관찰에 유효한 액션 하나를 선택하고 목표 달성 시 종료하는 규칙을 참고했습니다.
  프롬프트의 반복 금지를 없애고 스크롤·재시도를 허용하되 진행 여부를 확인하도록 했습니다.
  실제 저수준 입력 동작에 맞춰 포커스·텍스트 추가·Enter의 역할도 명시했습니다.

반복 임계치는 `--repeating-action-failure-th 5`로 설정하며 run metadata에 저장합니다.
기존 `--max-proposals` 옵션과 `rejected_proposals`, `state_changed` 로그 필드는 제거했습니다.
화면이 변해도 같은 행동이 임계치만큼 연속되면 종료하고, 서로 다른 행동의 순환은
`max-steps`로 제한합니다. 설정과 소스가 달라졌으므로 기존 결과와 별도 디렉터리를 사용하세요.

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
