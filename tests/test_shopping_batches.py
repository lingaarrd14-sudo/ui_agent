"""Exercise the real shell loop without Docker, browsers, or model calls."""

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ShoppingBatchTests(unittest.TestCase):
    def test_python_uses_the_same_vwa_root_override_as_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", "from ui_agent.vwa_config import VWA_ROOT; print(VWA_ROOT)"],
                env={**os.environ, "VWA_ROOT": directory}, cwd=ROOT,
                text=True, capture_output=True, check=True, timeout=30,
            )
            self.assertEqual(Path(result.stdout.strip()), Path(directory).resolve())

    def run_script(self, mode):
        python = shlex.quote(sys.executable)
        script = shlex.quote(str(ROOT / "run_shopping_batches.sh"))
        # Mock external work, while retaining the actual task count and loop.
        harness = f"""
function {python}() {{
    if [[ "$1" == "-" ]]; then
        command {python} "$@"
    elif [[ " $* " == *" --validate-only "* ]]; then
        echo VALIDATE
        [[ "$AUDIT_MODE" != validation_failure ]]
    else
        echo "RUN $*"
        case "$AUDIT_MODE" in
            task_error) return 2 ;;
            fatal_error) return 1 ;;
        esac
    fi
}}
docker() {{
    echo "DOCKER $1"
    if [[ "$AUDIT_MODE" == reset_failure && "$1" == stop ]]; then
        return 1
    fi
}}
sleep() {{ :; }}
export -f docker sleep
source {script}
"""
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.update(
                PYTHON=sys.executable,
                VWA_ROOT=str(ROOT.parent / "visualwebarena"),
                RESULT_ROOT=directory,
                BATCH_SIZE="50",
                MODEL="audit-model",
                MAX_STEPS="30",
                VIEWPORT_WIDTH="1280",
                VIEWPORT_HEIGHT="720",
                AUDIT_MODE=mode,
            )
            return subprocess.run(
                ["bash"], input=harness, text=True, capture_output=True,
                env=env, cwd=ROOT, timeout=30,
            )

    def test_all_batches_continue_after_task_errors_and_refresh_auth(self):
        result = self.run_script("task_error")
        self.assertEqual(result.returncode, 2, result.stderr)
        runs = [line for line in result.stdout.splitlines() if line.startswith("RUN ")]
        self.assertEqual(len(runs), 9)
        self.assertIn("--start 400 --end 405", runs[-1])
        self.assertTrue(all("--refresh-auth" in line for line in runs))

    def test_success_returns_zero(self):
        result = self.run_script("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("RUN "), 9)

    def test_validation_failure_stops_before_reset(self):
        result = self.run_script("validation_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("DOCKER ", result.stdout)
        self.assertNotIn("RUN ", result.stdout)

    def test_reset_failure_stops_before_starting_a_task(self):
        result = self.run_script("reset_failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DOCKER stop", result.stdout)
        self.assertNotIn("DOCKER run", result.stdout)
        self.assertNotIn("RUN ", result.stdout)

    def test_fatal_runner_failure_stops_remaining_batches(self):
        result = self.run_script("fatal_error")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.count("RUN "), 1)
