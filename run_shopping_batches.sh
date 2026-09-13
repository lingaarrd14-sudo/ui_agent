#!/usr/bin/env bash
# Shopping 태스크를 50개씩 초기화하며 실행합니다.
set -euo pipefail

# 1. 실행 설정 (환경변수로 변경 가능)
MODEL="${MODEL:-gemini-3.8-flash}"
BATCH_SIZE="${BATCH_SIZE:-50}"
MAX_STEPS="${MAX_STEPS:-30}"
VIEWPORT_WIDTH="${VIEWPORT_WIDTH:-1280}"
VIEWPORT_HEIGHT="${VIEWPORT_HEIGHT:-2048}"
SKIP_TASK_IDS=(284 319 345) # Shopping tasks that open Wikipedia.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
VWA_ROOT="${VWA_ROOT:-$SCRIPT_DIR/../visualwebarena}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/benchmark_results/shopping_${MODEL}_${VIEWPORT_WIDTH}x${VIEWPORT_HEIGHT}_$(date +%Y%m%d_%H%M%S)}"
TASK_CONFIG="$VWA_ROOT/config_files/vwa/test_shopping.raw.json"

# 2. 실행 대상 수 계산: 다중 탭·별도 viewport·Wikipedia 태스크 제외
TASK_COUNT="$("$PYTHON" - "$TASK_CONFIG" "${SKIP_TASK_IDS[@]}" <<'PY'
import json
import sys

skip_ids = {int(task_id) for task_id in sys.argv[2:]}
with open(sys.argv[1], encoding="utf-8") as source:
    tasks = json.load(source)
print(sum(
    "viewport_size" not in task
    and "|AND|" not in task.get("start_url", "")
    and task["task_id"] not in skip_ids
    for task in tasks
))
PY
)"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-$TASK_COUNT}"

if ((START_INDEX < 0 || END_INDEX < START_INDEX || END_INDEX > TASK_COUNT)); then
    echo "Invalid task range: START_INDEX=$START_INDEX END_INDEX=$END_INDEX (valid: 0..$TASK_COUNT)" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT"
echo "Running shopping tasks [$START_INDEX, $END_INDEX) in batches of $BATCH_SIZE."
echo "Skipping task IDs: ${SKIP_TASK_IDS[*]}"
echo "Results: $RESULT_ROOT"

# 3. 배치마다 설정 검증 → 초기화 → 실행
exit_status=0
for ((start = START_INDEX; start < END_INDEX; start += BATCH_SIZE)); do
    end=$((start + BATCH_SIZE))
    if ((end > END_INDEX)); then end="$END_INDEX"; fi
    batch_dir="$RESULT_ROOT/batch_${start}_${end}"

    batch_command=(
        "$PYTHON" "$SCRIPT_DIR/vwa_benchmark.py"
        --start "$start" --end "$end"
        --model "$MODEL" --max-steps "$MAX_STEPS"
        --viewport-width "$VIEWPORT_WIDTH" --viewport-height "$VIEWPORT_HEIGHT"
        --result-dir "$batch_dir" --refresh-auth
    )

    # 재시작 설정이 맞지 않으면 사이트 초기화 전에 중단합니다.
    "${batch_command[@]}" --validate-only
    echo "Resetting shopping before tasks [$start, $end)."
    (cd "$VWA_ROOT" && bash -e -o pipefail scripts/reset_shopping.sh)

    status=0
    "${batch_command[@]}" || status=$?
    case "$status" in
        0) ;;                    # 정상 완료 (평가 점수 0도 포함)
        2)                       # 개별 태스크 오류: 다음 배치 진행
            echo "Task errors in $batch_dir; continuing." >&2
            exit_status=2 ;;
        *) exit "$status" ;;     # 로그인·실행 준비 실패: 즉시 중단
    esac
done
exit "$exit_status"
