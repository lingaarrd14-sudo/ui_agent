"""Append-only benchmark results and reproducible run identity management."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def read_latest_results(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Read append-only results, keeping the newest retry for each task."""
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = (str(record["domain"]), int(record["task_id"]))
            records[key] = record
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Durably append one completed task attempt."""
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


def write_summary(
    result_dir: Path, results_path: Path, total_planned: int
) -> dict[str, Any]:
    """Summarize the latest attempt for every task."""
    records = list(read_latest_results(results_path).values())
    scored = [record for record in records if record.get("score") is not None]
    passed = sum(float(record["score"]) for record in scored)
    summary = {
        "completed": len(records),
        "planned": total_planned,
        "scored": len(scored),
        "errors": len(records) - len(scored),
        "passed": passed,
        "score_completed": passed / len(records) if records else 0.0,
        "score_planned": passed / total_planned if total_planned else 0.0,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def git_metadata(path: Path) -> dict[str, Any]:
    """Return commit identity and whether local changes exist."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"revision": revision, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def source_digest(project_root: Path) -> str:
    """Fingerprint exact runtime sources, including uncommitted work."""
    sources = [
        project_root / "vwa_benchmark.py",
        project_root / "requirements.txt",
        *(project_root / "ui_agent").glob("*.py"),
    ]
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def persist_run_config(result_dir: Path, metadata: dict[str, Any]) -> None:
    """Write immutable run identity or reject an incompatible resume."""
    config_path = result_dir / "run_config.json"
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid existing run config: {config_path}") from error
        if existing != metadata:
            changed = sorted(
                key
                for key in set(existing) | set(metadata)
                if existing.get(key) != metadata.get(key)
            )
            raise RuntimeError(
                "Result directory belongs to an incompatible run; choose a new "
                f"--result-dir. Changed fields: {', '.join(changed)}"
            )
        return
    config_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
