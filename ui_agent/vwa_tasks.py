"""VisualWebArena task 실행, 공식 평가, 결과 저장 흐름."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Protocol

from .models import DoneAction, ExecutionResult
from .policy import OpenAIVisionPolicy
from .vwa_adapter import VisualWebArenaAdapter, pil_to_base64
from .vwa_config import load_reference_images
from .vwa_results import (
    append_jsonl,
    read_latest_results,
    write_summary,
)
from .vwa_runtime import BrowserEnvActionFactory, EvaluationCaptioner, VWABindings


class BenchmarkOptions(Protocol):
    """Task 실행에 필요한 CLI 옵션의 최소 계약."""

    max_steps: int
    max_proposals: int
    viewport_width: int
    viewport_height: int
    save_traces: bool


def _write_runtime_config(
    result_dir: Path, domain: str, task: dict[str, Any]
) -> Path:
    """생성된 task에 run 전용 인증 상태 경로를 추가한다."""
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


def _run_agent_steps(
    *,
    options: BenchmarkOptions,
    bindings: VWABindings,
    policy: OpenAIVisionPolicy,
    action_factory: BrowserEnvActionFactory,
    env: Any,
    task: dict[str, Any],
    runtime_config: Path,
    reference_images: list[str],
) -> tuple[list[Any], list[dict[str, Any]], int]:
    """에이전트 액션을 실행하고 공식 평가에 전달할 trajectory를 만든다."""
    observation, info = env.reset(options={"config_file": str(runtime_config)})
    state: dict[str, Any] = {"observation": observation, "info": info}
    trajectory: list[Any] = [state]
    adapter = VisualWebArenaAdapter(
        policy=policy,
        action_factory=action_factory,
        viewport_width=options.viewport_width,
        viewport_height=options.viewport_height,
        max_proposals=options.max_proposals,
    )
    step_records: list[dict[str, Any]] = []
    model_calls = 0

    for step in range(1, options.max_steps + 1):
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

    return trajectory, step_records, model_calls


def _score_task(
    *,
    bindings: VWABindings,
    captioner: EvaluationCaptioner,
    env: Any,
    task: dict[str, Any],
    runtime_config: Path,
    trajectory: list[Any],
) -> float:
    """에이전트 입력과 분리된 VWA 공식 evaluator로 trajectory를 채점한다."""
    eval_types = task["eval"]["eval_types"]
    caption_fn = captioner.get() if "page_image_query" in eval_types else None
    evaluator = bindings.evaluator_router(runtime_config, captioning_fn=caption_fn)
    return float(evaluator(trajectory, runtime_config, env.page))


def _execute_task(
    *,
    options: BenchmarkOptions,
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
    """Task 하나를 실행하고 공식 점수와 감사 기록을 반환한다."""
    reference_pil = load_reference_images(task.get("image"))
    reference_images = [pil_to_base64(image) for image in reference_pil]
    trajectory, step_records, model_calls = _run_agent_steps(
        options=options,
        bindings=bindings,
        policy=policy,
        action_factory=action_factory,
        env=env,
        task=task,
        runtime_config=runtime_config,
        reference_images=reference_images,
    )
    score = _score_task(
        bindings=bindings,
        captioner=captioner,
        env=env,
        task=task,
        runtime_config=runtime_config,
        trajectory=trajectory,
    )
    if options.save_traces:
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


def run_selected_tasks(
    *,
    options: BenchmarkOptions,
    bindings: VWABindings,
    policy: OpenAIVisionPolicy,
    action_factory: BrowserEnvActionFactory,
    captioner: EvaluationCaptioner,
    env: Any,
    result_dir: Path,
    selected: list[tuple[str, Path]],
) -> None:
    """미완료 task를 순서대로 실행하고 각 시도 결과를 즉시 저장한다."""
    results_path = result_dir / "results.jsonl"
    completed = {
        key
        for key, record in read_latest_results(results_path).items()
        if record.get("score") is not None
    }
    total_planned = len(selected)

    for position, (domain, config_path) in enumerate(selected, start=1):
        task = json.loads(config_path.read_text(encoding="utf-8"))
        task_id = int(task["task_id"])
        if (domain, task_id) in completed:
            print(f"[{position}/{total_planned}] skip {domain}/{task_id}", flush=True)
            continue

        runtime_config = _write_runtime_config(result_dir, domain, task)
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
                _execute_task(
                    options=options,
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
            # 한 task의 실패를 기록한 뒤 나머지 선택 범위는 계속 실행한다.
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
