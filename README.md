# UI Agent

VisualWebArena(VWA)의 Shopping task, 인증, 브라우저 환경, 공식 evaluator를 사용하는
스크린샷 기반 웹 에이전트입니다. 실행 진입점은 `vwa_benchmark.py`입니다.
VWA checkout 기본 경로는 `../visualwebarena`이며, 셸 환경변수 `VWA_ROOT`로 변경할 수 있습니다.

## 모델 입력과 액션

모델에는 task의 목표, 현재 화면, 직전 액션 전 화면 한 장, 화면 크기, 최근 실행
5단계의 짧은 화면 요약과 task에 포함된 reference image를 전달합니다. 첫 단계에는
직전 화면이 없습니다. DOM/접근성 트리, SoM 표시,
생성 캡션, 현재 URL, 탭 목록, evaluator의 정답과 불가능 사유는 전달하지 않습니다.
Reference image는 PNG로 변환해 별도의 이미지 입력으로 보냅니다.

| 액션 | 실행 의미 |
| --- | --- |
| `click`, `hover` | 0~1000 정수 좌표를 VWA의 0~1 좌표로 변환 |
| `type` | 현재 포커스에 텍스트 추가. 기존 내용 삭제와 Enter는 별도 액션 |
| `press` | `key_comb`에 지정한 키 조합 실행 |
| `scroll` | `up`/`down`으로 약 한 화면 이동 |
| `go_back`, `go_forward` | 현재 페이지의 방문 기록 이동 |
| `stop` | 답변을 남기고 task 종료. 브라우저에는 실행하지 않음 |

`goto`, `new_tab`, `tab_focus`, `close_tab`은 현재 액션 스키마에 없습니다.
따라서 원본 VWA와 관찰·액션 공간이 동일한 실험은 아닙니다.

## 종료와 평가

목표를 달성하면 `stop(status="success", answer=...)`를 반환합니다. 정보 요청에는
요청한 답만, 탐색·수정 목표에는 짧은 완료 문구를 씁니다. 어떤 유형의 목표든
충분히 확인한 뒤 본질적으로 불가능하다고 판단하면 `answer="N/A"`로 종료합니다.
확인한 사유는 `expected_outcome`에 남깁니다. 불가능 여부는 모델이 관찰에서 판단하며,
정답 config를 읽어서 자동으로 종료시키지 않습니다.

운영상 진행할 수 없으면 `stop/blocked`를 사용합니다. 내부 `success`/`blocked`는
공식 점수가 아닙니다. 반복이나 실행 오류를 자동으로 `N/A`로 바꾸지 않습니다.

- 같은 액션을 기본 5회 연속 실행하면 다음 모델 호출 전에 종료합니다.
  실행 실패도 횟수에 포함하며, 화면이 달라져도 동일 액션이면 셉니다. 스크롤은
  방향이 달라도 연속 스크롤로 세므로 `down`/`up` 왕복도 5회에서 종료합니다.
- 클릭/hover는 좌표, 입력은 텍스트, press는 키 조합, 스크롤은 방향을 비교합니다.
  VWA처럼 같은 종류의 인자 없는 액션도 동일하게 봅니다.
- 기본 `--max-steps 15`는 브라우저 액션 실행 횟수의 상한입니다.
  상한에 도달하면 추가 모델 호출 없이 STOP을 붙입니다.
- 좌표 범위나 빈 입력 검증에 실패하면 실행 없이 `blocked`로 종료합니다.
  모델 응답 파싱/API 예외는 task의 `error`로 기록합니다. 원본의 파싱 실패
  `NONE` 액션 누적은 구현하지 않습니다.
- 액션 실행 뒤에는 원본 runner와 같이 2.5초를 기다립니다.

채점은 [원본 evaluator](../visualwebarena/evaluation_harness/evaluators.py)를 직접
import해 최종 trajectory, runtime config, 실제 페이지로 실행합니다.

| 평가 유형 | 검사 내용 |
| --- | --- |
| `string_match` | STOP 답변의 문자열·수치·LLM 의미 일치 |
| `url_match` | 종료 시점의 페이지 URL |
| `program_html` | 지정 페이지의 HTML/JS 또는 원본 helper 결과 |
| `page_image_query` | 이미지 SSIM 유사도 또는 BLIP-2 VQA |

여러 평가가 지정되면 원본처럼 점수를 곱합니다. 불가능 정답의 정확한 `N/A`는
LLM 호출 없이 통과하고, 설명형 답변은 원본의 불가능 사유 비교 LLM으로 평가합니다.
BLIP-2 평가 함수는 첫 점수 계산 때 한 번 로드해 전체 run에서 재사용합니다. 기본 CPU이며
`--eval-caption-device cuda`로 변경할 수 있습니다. 평가용 모델과 캡션은 에이전트
입력에 들어가지 않습니다. `--model`은 에이전트 모델만 바꾸며, LLM judge 모델은
로컬 VWA의 `evaluation_harness/helper_functions.py` 설정을 따릅니다.

## 실행 대상

원본 `config_files/vwa/test_shopping.raw.json`을 읽어 결과 디렉터리에 실행용
config를 생성합니다. 다중 탭 task, `viewport_size` 지정 task, Wikipedia 관련
task ID `284 319 345`를 제외하며 현재 로컬 task 파일 기준 실행 대상은 405개입니다.

`--start`/`--end`는 필터링된 목록에 적용하는 반열린 구간 `[start, end)`입니다.
이 값은 task ID가 아니며, `--end`를 생략하면 남은 대상을 모두 선택합니다.

## 설치와 설정

```bash
cd /home/default/agent/ui_agent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
python -m nltk.downloader punkt punkt_tab
```

프로젝트 루트의 `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
MODEL=gpt-5.6-terra
# 호환 게이트웨이를 사용하는 경우에만 지정
# OPENAI_BASE_URL=https://example-gateway.invalid/v1
```

쉘에 이미 설정한 환경변수가 `.env`보다 우선합니다. 호환 gateway를 쓸 때는
`OPENAI_BASE_URL`을 설정하세요. OpenAI 에이전트와 VWA LLM judge가 이 값을 사용합니다.

Gemini는 `.env`에 `GEMINI_API_KEY`를 설정하고 `--model gemini-3.8-flash`로
선택합니다. GPT와 Gemini 에이전트 모두 같은 Chat Completions 메시지, 이미지 입력,
structured output 스키마를 사용합니다. 이미지에는 제공자별 `detail` 힌트를 지정하지
않고 동일한 PNG 데이터를 전달합니다. Gemini 요청만 Google의 OpenAI 호환 endpoint로
보냅니다. 두 모델 모두 click·hover에 0~1000 정수 좌표를 반환하며, 실행 이력에도 같은
단위를 사용합니다. 화면 중심은 항상 `(500, 500)`이고, VWA에 전달할 때만 1000으로
나누어 0~1로 변환합니다. 끝점 1000은 화면 밖을 클릭하지 않도록 마지막 픽셀로
제한합니다. VWA 평가용 `OPENAI_API_KEY`도 필요합니다.
배치는 `MODEL=gemini-3.8-flash bash run_shopping_batches.sh`로 실행합니다.

Shopping 주소 기본값은 `SHOPPING=http://localhost:7770`입니다. 필요하면 같은 이름의
환경변수로 변경하세요. VWA Shopping 사이트는 별도로 구동해야 합니다.

## 실행

브라우저와 모델 API를 호출하지 않고 import와 선택된 config를 확인합니다.
VWA evaluator import가 API client를 만들기 때문에 `OPENAI_API_KEY`는 필요합니다.
원격 사이트 접속과 이미지 다운로드 성공 여부까지 확인하지는 않습니다.

```bash
python vwa_benchmark.py --end 1 \
  --result-dir benchmark_results/shopping_validate --validate-only
```

한 task 실행:

```bash
python vwa_benchmark.py --end 1 --max-steps 15 \
  --result-dir benchmark_results/shopping_smoke
```

필터링된 Shopping 전체 405개를 한 번에 실행:

```bash
python vwa_benchmark.py --max-steps 30 \
  --viewport-width 1280 --viewport-height 720 \
  --result-dir benchmark_results/shopping_1280x720
```

기본은 headless입니다. `--headed`로 창을 표시하고 `--save-traces`로
Playwright trace를 저장할 수 있습니다. 기본 화면 크기는 1280×720입니다.

Shopping 배치 실행은 `bash run_shopping_batches.sh`를 사용합니다. 이 스크립트는
매 배치 전에 원본 Shopping reset 스크립트를 실행하고, 기본 50개씩 headed로
실행합니다. 기본 제외 ID는 `284 319 345`이며 현재 실행 대상은 405개입니다.
`MODEL`, `BATCH_SIZE`, `MAX_STEPS`, `VIEWPORT_WIDTH`, `VIEWPORT_HEIGHT`,
`RESULT_ROOT`를 환경변수로 바꿀 수 있습니다.

SH는 실행 설정을 검증한 뒤 원본 스크립트로 초기화하고, 인증을 새로 만듭니다.
별도의 사이트 준비 대기 루프는 없습니다. 개별 task 오류는 기록하고 다음 배치로
진행합니다. 초기화·로그인·설정
실패는 즉시 중단합니다. 모든 배치를 끝냈어도 task 오류가 남으면 종료 코드는 2입니다.

중단 후에는 같은 설정과 `RESULT_ROOT`로 다시 실행하세요.

```bash
MODEL=gpt-5.6-luna RESULT_ROOT=benchmark_results/shopping_full bash run_shopping_batches.sh
```

최초 실행과 재시작에 같은 명령을 사용합니다. 이미 채점된 task는 성공·실패 모두
건너뛰고, 오류·미완료 task만 처음부터 재실행합니다. 중단된 task의 중간 단계부터
이어가지는 않습니다. 재시작할 때도 배치별 사이트 초기화와 인증 갱신을 수행하므로
이전 task가 바꾼 사이트 상태는 복원되지 않습니다. `RESULT_ROOT`를 생략하면 매번
새 결과 폴더가 생성되어 이어서 실행되지 않습니다. 배치 크기도 동일하게 유지하세요.

## 사이트 상태와 결과

Shopping의 상태 초기화는 별도 원본 스크립트가 필요합니다. 비교 실험에서는
같은 초기 상태와 같은 배치 경계를 사용하세요.
인증 상태는 결과 디렉터리에 생성해 재사용하며 `--refresh-auth`로 갱신합니다.
인증 갱신은 사이트 데이터 초기화와 별개입니다.

생성 config, auth, trace, 결과는 `--result-dir` 아래에 저장합니다.
동일 설정·소스로 재실행하면 이미 점수가 있는 task는 건너뛰고 오류 task를
재시도합니다. 소스, 프롬프트, 모델 등 run metadata가 다르면 같은 결과 디렉터리
재사용을 거절하므로 새 디렉터리를 지정해야 합니다.

- `run_config.json`: Git 상태, 실행 소스 해시, 프롬프트, 모델, endpoint, 좌표 단위, 화면 크기와 선택 범위.
- `results.jsonl`: task별 시도를 추가 기록. 점수 1이면 `pass`, 그 외는 `fail`, 예외는 `error`.
- `step_records`: 모델 결정과 실행 성공 여부. `steps`는 브라우저 실행 횟수이며 STOP은 제외.
- `final_url`: evaluator가 페이지를 이동하기 전 에이전트가 끝낸 URL.
- `summary.json`: task별 최신 시도로 계산한 요약. `score_completed`는 오류를 포함한
  시도 task 수, `score_planned`는 전체 계획 task 수를 분모로 사용.

```bash
python summarize_benchmark.py benchmark_results/shopping_1280x720
```

runner 종료 코드는 배치를 마쳤지만 오류 task가 있으면 2, 없으면 0입니다.
처리되지 않은 설정·실행 준비 예외는 1로 종료합니다. 평가상 `fail`만 있는
경우에도 0을 반환합니다. `model_calls`는 정책 요청 횟수이며 SDK 내부 재시도 횟수나
평가용 LLM 호출 횟수는 포함하지 않습니다.

## 코드와 테스트

`policy.py`는 모델 입력과 결정, `controller.py`는 반복·액션 검증,
`vwa_adapter.py`는 관찰과 액션 변환을 담당합니다. `vwa_config.py`는 task 준비,
`vwa_runtime.py`는 원본 바인딩·인증·captioner, `vwa_tasks.py`는 실행과 평가,
`vwa_results.py`는 결과와 재실행 설정을 담당합니다.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

테스트는 API, 브라우저, Docker를 실행하지 않습니다. `test_vwa_evaluation.py`는
로컬 VWA를 직접 import해 문자열, 불가능 사유 비교, URL/HTML 조합과 이미지 SSIM/VQA를
확인합니다. 선택 대상의 불가능 Shopping task 29개도 `N/A` 채점과
정답 설정 보존을 검사합니다. 외부 LLM 응답·이미지 다운로드·VQA 추론은 mock하므로
실제 사이트 상태나 모델의 불가능 판단 정확도를 검증하는 테스트는 아닙니다.
