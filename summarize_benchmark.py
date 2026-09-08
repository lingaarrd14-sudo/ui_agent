#!/usr/bin/env python3
"""Print a compact, retry-aware report for a VWA result directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_latest(path: Path) -> list[dict[str, Any]]:
    """Keep only the newest append-only record for each domain/task pair."""
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        latest[(str(record["domain"]), int(record["task_id"]))] = record
    return list(latest.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    results_path = args.result_dir / "results.jsonl"
    if not results_path.is_file():
        parser.error(f"results file not found: {results_path}")

    rows = load_latest(results_path)
    summary_path = args.result_dir / "summary.json"
    planned = (
        int(json.loads(summary_path.read_text(encoding="utf-8")).get("planned", 0))
        if summary_path.exists()
        else len(rows)
    )
    scored = [row for row in rows if row.get("score") is not None]
    passed = sum(float(row["score"]) for row in scored)
    print(
        f"completed: {len(rows)}/{planned}  scored: {len(scored)}  "
        f"errors: {len(rows) - len(scored)}"
    )
    if rows:
        planned_score = f"{passed / planned:.4f}" if planned else "n/a"
        print(f"score completed: {passed / len(rows):.4f}  score planned: {planned_score}")
    else:
        print("score: n/a")

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain", "unknown"))].append(row)
    for domain, domain_rows in sorted(by_domain.items()):
        domain_passed = sum(
            float(row.get("score", 0) or 0) for row in domain_rows
        )
        average_steps = sum(row.get("steps", 0) for row in domain_rows) / len(
            domain_rows
        )
        print(
            f"{domain:12s} {domain_passed:.0f}/{len(domain_rows)} = "
            f"{domain_passed / len(domain_rows):.4f}  avg steps={average_steps:.1f}"
        )

    actions = Counter(
        step.get("decision", {}).get("action", {}).get("kind")
        for row in rows
        for step in row.get("step_records", [])
    )
    actions.pop(None, None)
    model_calls = sum(int(row.get("model_calls", 0)) for row in rows)
    print(f"actions: {dict(actions)}")
    print(f"model calls: {model_calls}")
    print("recent failures:")
    for row in rows[-5:]:
        if row.get("score") != 1.0:
            print(
                f"  {row['domain']}/{row['task_id']}: {row.get('status')} "
                f"steps={row.get('steps')} {row.get('intent', '')[:100]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
