"""Run the vision-only UI agent on VisualWebArena without modifying VWA.

VisualWebArena supplies task data, authentication, browser execution, and the
official evaluator. The agent receives only the current raw screenshot, URL,
viewport, recent executed actions, and task-provided reference images. It never
receives SoM marks, accessibility-tree text, or generated captions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from ui_agent.models import DoneAction, ExecutionResult
from ui_agent.policy import INSTRUCTIONS, OpenAIVisionPolicy
from ui_agent.vwa_adapter import VisualWebArenaAdapter, pil_to_base64
from ui_agent.vwa_config import (
    DOMAIN_SOURCES,
    UI_ROOT,
    VWA_ROOT,
    configure_environment,
    generate_configs,
    load_local_environment,
    load_reference_images,
    site_urls,
    task_selection,
    validate_selection,
)
from ui_agent.vwa_results import (
    append_jsonl,
    git_metadata,
    persist_run_config,
    read_latest_results,
    source_digest,
    write_summary,
)
from ui_agent.vwa_runtime import (
    BrowserEnvActionFactory,
    EvaluationCaptioner,
    VWABindings,
    ensure_auth,
    load_vwa_bindings,
)


VWA_INSTRUCTIONS = INSTRUCTIONS + """
For an information-retrieval goal, put only the requested answer in done.summary because the
official evaluator scores that text. For navigation or modification goals, use a short factual
completion summary. VisualWebArena scrolling moves approximately one viewport in the sign of
delta_y, so inspect the next screenshot before scrolling again.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["all", *DOMAIN_SOURCES], default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--model", default=os.environ.get("MODEL", "gpt-5.6-terra"))
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--refresh-auth", action="store_true")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument(
        "--eval-caption-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
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
        / f"vwa_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.end is not None and args.end < args.start:
        parser.error("--end must be greater than or equal to --start")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.max_proposals < 1:
        parser.error("--max-proposals must be at least 1")
    if args.viewport_width < 1 or args.viewport_height < 1:
        parser.error("viewport dimensions must be positive")
    return args


def build_run_metadata(
    args: argparse.Namespace,
    domains: list[str],
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
        "uses_som": False,
        "uses_agent_captioning": False,
        "model": args.model,
        "api_base_url": base_url or "OpenAI SDK default",
        "policy_instructions": VWA_INSTRUCTIONS,
        "max_steps": args.max_steps,
        "max_proposals_per_step": args.max_proposals,
        "viewport": [args.viewport_width, args.viewport_height],
        "domains": domains,
        "start": args.start,
        "end": args.end,
        "site_urls": site_urls(),
        "validation": validation,
    }


def write_runtime_config(
    result_dir: Path, domain: str, task: dict[str, Any]
) -> Path:
    """Add run-local auth state to one generated task config."""
    task["storage_state"] = str(
        (result_dir / "auth" / f"{domain}_state.json").resolve()
    )
    runtime_dir = result_dir / "runtime_configs" / domain
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_dir / f"{task['task_id']}.json"
    runtime_config.write_text(
        json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return runtime_config


def execute_task(
    *,
    args: argparse.Namespace,
    bindings: VWABindings,
    policy: OpenAIVisionPolicy,
    action_factory: BrowserEnvActionFactory,
    captioner: EvaluationCaptioner,
    env: Any,
    result_dir: Path,
    domain: str,
    task: dict[str, Any],
    runtime_config: Path,
) -> dict[str, Any]:
    """Execute and officially score exactly one task."""
    reference_pil = load_reference_images(task.get("image"))
    reference_images = [pil_to_base64(image) for image in reference_pil]
    observation, info = env.reset(options={"config_file": str(runtime_config)})
    state: dict[str, Any] = {"observation": observation, "info": info}
    trajectory: list[Any] = [state]
    adapter = VisualWebArenaAdapter(
        policy=policy,
        action_factory=action_factory,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        max_proposals=args.max_proposals,
    )
    step_records: list[dict[str, Any]] = []
    model_calls = 0

    for step in range(1, args.max_steps + 1):
        decision, before, action = adapter.decide(
            task["intent"], state, reference_images
        )
        audit = {
            "step": step,
            "decision": decision.model_dump(mode="json"),
            "rejected_proposals": adapter.controller.last_rejections.copy(),
            "model_calls": adapter.controller.last_proposal_count,
        }
        model_calls += adapter.controller.last_proposal_count
        step_records.append(audit)
        trajectory.append(action)
        print(f"  step {step}: {decision.action.model_dump(mode='json')}", flush=True)

        if isinstance(decision.action, DoneAction):
            audit["executed"] = False
            break

        observation, _, terminated, truncated, info = env.step(action)
        after_state = {"observation": observation, "info": info}
        execution = ExecutionResult(
            ok=not bool(info.get("fail_error")),
            message=info.get("fail_error", ""),
        )
        history_record = adapter.record(before, decision, execution, after_state)
        audit.update(
            {
                "executed": True,
                "result": history_record.result.model_dump(mode="json"),
            }
        )
        state = after_state
        trajectory.append(state)
        if terminated or truncated:
            trajectory.append(bindings.create_stop_action("Environment terminated"))
            break
    else:
        trajectory.append(bindings.create_stop_action("Maximum steps reached"))

    eval_types = task["eval"]["eval_types"]
    caption_fn = captioner.get() if "page_image_query" in eval_types else None
    evaluator = bindings.evaluator_router(runtime_config, captioning_fn=caption_fn)
    score = float(evaluator(trajectory, runtime_config, env.page))
    if args.save_traces:
        trace_dir = result_dir / "traces" / domain
        trace_dir.mkdir(parents=True, exist_ok=True)
        env.save_trace(trace_dir / f"{task['task_id']}.zip")
    return {
        "score": score,
        "steps": len(step_records),
        "model_calls": model_calls,
        "status": "pass" if score == 1.0 else "fail",
        "step_records": step_records,
        "final_url": env.page.url,
    }


def run() -> int:
    load_local_environment()  # MODEL must be available before parse_args.
    args = parse_args()
    api_key, base_url = configure_environment(require_api_key=not args.validate_only)
    bindings = load_vwa_bindings(VWA_ROOT)

    domains = list(DOMAIN_SOURCES) if args.domain == "all" else [args.domain]
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_configs(result_dir, domains)
    selected = task_selection(generated, args.start, args.end)
    validation = validate_selection(selected)
    metadata = build_run_metadata(args, domains, validation, base_url)
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
    ensure_auth(result_dir / "auth", domains, args.refresh_auth, bindings.renew_comb)
    client_options: dict[str, Any] = {
        "api_key": api_key,
        "timeout": 180.0,
        "max_retries": 2,
    }
    if base_url:
        client_options["base_url"] = base_url
    policy = OpenAIVisionPolicy(
        OpenAI(**client_options), args.model, instructions=VWA_INSTRUCTIONS
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
        sleep_after_execution=0.8,
    )

    results_path = result_dir / "results.jsonl"
    completed = {
        key
        for key, record in read_latest_results(results_path).items()
        if record.get("score") is not None
    }
    total_planned = len(selected)
    try:
        for position, (domain, config_path) in enumerate(selected, start=1):
            task = json.loads(config_path.read_text(encoding="utf-8"))
            task_id = int(task["task_id"])
            if (domain, task_id) in completed:
                print(f"[{position}/{total_planned}] skip {domain}/{task_id}", flush=True)
                continue

            runtime_config = write_runtime_config(result_dir, domain, task)
            started = time.monotonic()
            record: dict[str, Any] = {
                "domain": domain,
                "task_id": task_id,
                "intent": task["intent"],
                "score": None,
                "steps": 0,
                "model_calls": 0,
                "status": "error",
            }
            print(
                f"[{position}/{total_planned}] {domain}/{task_id}: {task['intent']}",
                flush=True,
            )
            try:
                record.update(
                    execute_task(
                        args=args,
                        bindings=bindings,
                        policy=policy,
                        action_factory=action_factory,
                        captioner=captioner,
                        env=env,
                        result_dir=result_dir,
                        domain=domain,
                        task=task,
                        runtime_config=runtime_config,
                    )
                )
            except Exception as error:
                record["error"] = repr(error)
                record["traceback"] = traceback.format_exc()
                print(f"  ERROR: {error!r}", flush=True)

            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            append_jsonl(results_path, record)
            write_summary(result_dir, results_path, total_planned)
            print(
                f"  => {record['status']} score={record['score']} "
                f"elapsed={record['elapsed_seconds']}s",
                flush=True,
            )
    finally:
        env.close()

    summary = write_summary(result_dir, results_path, total_planned)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(run())
