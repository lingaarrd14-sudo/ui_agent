"""Run the vision-only UI agent on VisualWebArena without modifying VWA.

VisualWebArena supplies task data, authentication, browser execution, and the
official evaluator. The agent receives the current and previous raw screenshots,
viewport, recent executed actions, and task-provided reference images. It never
receives SoM marks, accessibility-tree text, or generated captions.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from ui_agent.policy import INSTRUCTIONS, OpenAIVisionPolicy
from ui_agent.vwa_config import (
    SHOPPING_DOMAIN,
    SHOPPING_EXCLUDED_TASK_IDS,
    UI_ROOT,
    VWA_ROOT,
    configure_environment,
    generate_configs,
    load_local_environment,
    site_urls,
    task_selection,
    validate_selection,
)
from ui_agent.vwa_results import (
    git_metadata,
    persist_run_config,
    source_digest,
    write_summary,
)
from ui_agent.vwa_runtime import (
    BrowserEnvActionFactory,
    EvaluationCaptioner,
    ensure_auth,
    load_vwa_bindings,
)
from ui_agent.vwa_tasks import run_selected_tasks


VWA_SLEEP_AFTER_EXECUTION = 2.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--repeating-action-failure-th", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get("MODEL", "gpt-5.6-terra"))
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--refresh-auth", action="store_true")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument(
        "--eval-caption-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Used only by official page_image_query evaluation, not by the agent.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate imports and selected task configs without a browser or API call.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=UI_ROOT
        / "benchmark_results"
        / f"shopping_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.end is not None and args.end < args.start:
        parser.error("--end must be greater than or equal to --start")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.repeating_action_failure_th < 1:
        parser.error("--repeating-action-failure-th must be at least 1")
    if args.viewport_width < 1 or args.viewport_height < 1:
        parser.error("viewport dimensions must be positive")
    return args


def build_run_metadata(
    args: argparse.Namespace,
    validation: dict[str, int],
    base_url: str | None,
) -> dict[str, Any]:
    """Capture everything needed to identify and reproduce a run."""
    return {
        "benchmark_git": git_metadata(VWA_ROOT),
        "agent_git": git_metadata(UI_ROOT),
        "agent_source_sha256": source_digest(UI_ROOT),
        "agent": "ui_agent_vision_only",
        "observation": "raw_screenshot",
        "action_space": "single_page_screenshot_actions",
        "action_coordinates": "normalized_0_1000",
        "excludes_multi_tab_tasks": True,
        "uses_som": False,
        "uses_agent_captioning": False,
        "model": args.model,
        "api_base_url": base_url or "OpenAI SDK default",
        "policy_instructions": INSTRUCTIONS,
        "max_steps": args.max_steps,
        "repeating_action_failure_th": args.repeating_action_failure_th,
        "sleep_after_execution": VWA_SLEEP_AFTER_EXECUTION,
        "eval_caption_device": args.eval_caption_device,
        "viewport": [args.viewport_width, args.viewport_height],
        "domain": SHOPPING_DOMAIN,
        "start": args.start,
        "end": args.end,
        "excluded_task_ids": sorted(SHOPPING_EXCLUDED_TASK_IDS),
        "site_urls": site_urls(),
        "validation": validation,
    }


def run() -> int:
    load_local_environment()  # MODEL must be available before parse_args.
    args = parse_args()
    api_key, base_url = configure_environment(
        require_api_key=not args.validate_only, model=args.model
    )
    bindings = load_vwa_bindings(VWA_ROOT)

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_configs(result_dir)
    selected = task_selection(generated, args.start, args.end)
    validation = validate_selection(selected)
    metadata = build_run_metadata(args, validation, base_url)
    persist_run_config(result_dir, metadata)

    if validation["requires_reset"]:
        print(
            f"[state] {validation['requires_reset']} selected tasks can mutate site state. "
            "This runner does not reset Docker services; use VWA reset scripts between "
            "controlled benchmark batches.",
            flush=True,
        )
    if args.validate_only:
        print(
            f"Validated {validation['tasks']} task(s), "
            f"{validation['reference_images']} reference image(s), VWA imports, "
            "and result config. No browser or model API was used.",
            flush=True,
        )
        return 0

    assert api_key is not None
    ensure_auth(result_dir / "auth", args.refresh_auth, bindings.renew_comb)
    client_options: dict[str, Any] = {
        "api_key": api_key,
        "timeout": 180.0,
        "max_retries": 2,
    }
    if base_url:
        client_options["base_url"] = base_url
    policy = OpenAIVisionPolicy(
        OpenAI(**client_options), args.model
    )
    action_factory = BrowserEnvActionFactory(
        bindings, args.viewport_width, args.viewport_height
    )
    captioner = EvaluationCaptioner(bindings, args.eval_caption_device)
    env = bindings.script_browser_env(
        headless=not args.headed,
        observation_type="image",
        current_viewport_only=True,
        viewport_size={"width": args.viewport_width, "height": args.viewport_height},
        save_trace_enabled=args.save_traces,
        sleep_after_execution=VWA_SLEEP_AFTER_EXECUTION,
    )

    results_path = result_dir / "results.jsonl"
    total_planned = len(selected)
    try:
        run_selected_tasks(
            options=args,
            bindings=bindings,
            policy=policy,
            action_factory=action_factory,
            captioner=captioner,
            env=env,
            result_dir=result_dir,
            selected=selected,
        )
    finally:
        env.close()

    summary = write_summary(result_dir, results_path, total_planned)
    # Distinguish completed batches with task errors from fatal setup failures.
    return 2 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(run())
