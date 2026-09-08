#!/usr/bin/env bash
# Run every non-viewport shopping task in reset-separated batches.
set -euo pipefail

# Edit these values for a different experiment.
MODEL="${MODEL:-gpt-5.6-terra}"
BATCH_SIZE="${BATCH_SIZE:-50}"
MAX_STEPS="${MAX_STEPS:-30}"
VIEWPORT_WIDTH="${VIEWPORT_WIDTH:-1280}"
VIEWPORT_HEIGHT="${VIEWPORT_HEIGHT:-720}"
SKIP_TASK_IDS=(284 319 345) # Shopping tasks that open Wikipedia.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
VWA_ROOT="${VWA_ROOT:-$SCRIPT_DIR/../visualwebarena}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/benchmark_results/shopping_$(date +%Y%m%d_%H%M%S)}"
TASK_CONFIG="$VWA_ROOT/config_files/vwa/test_shopping.json"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$TASK_CONFIG" ]]; then
    echo "Shopping config not found: $TASK_CONFIG" >&2
    exit 1
fi

TASK_COUNT="$("$PYTHON" - "$TASK_CONFIG" "${SKIP_TASK_IDS[@]}" <<'PY'
import json
import sys

config_path, *skipped = sys.argv[1:]
skip_ids = {int(task_id) for task_id in skipped}
tasks = json.load(open(config_path, encoding="utf-8"))
print(sum(
    "viewport_size" not in task and task["task_id"] not in skip_ids
    for task in tasks
))
PY
)"

mkdir -p "$RESULT_ROOT"
echo "Running $TASK_COUNT shopping tasks in batches of $BATCH_SIZE."
echo "Skipping task IDs: ${SKIP_TASK_IDS[*]}"
echo "Results: $RESULT_ROOT"

for ((start = 0; start < TASK_COUNT; start += BATCH_SIZE)); do
    end=$((start + BATCH_SIZE))
    if ((end > TASK_COUNT)); then end="$TASK_COUNT"; fi
    batch_dir="$RESULT_ROOT/batch_${start}_${end}"

    echo "Resetting shopping before tasks [$start, $end)."
    (cd "$VWA_ROOT" && bash scripts/reset_shopping.sh)

    "$PYTHON" "$SCRIPT_DIR/vwa_benchmark.py" \
        --domain shopping \
        --start "$start" \
        --end "$end" \
        --headed \
        --exclude-task-ids "${SKIP_TASK_IDS[@]}" \
        --model "$MODEL" \
        --max-steps "$MAX_STEPS" \
        --viewport-width "$VIEWPORT_WIDTH" \
        --viewport-height "$VIEWPORT_HEIGHT" \
        --result-dir "$batch_dir"
done
