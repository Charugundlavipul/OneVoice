from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import load_json, load_jsonl, write_json
from experiments.cross_task_generalization.common.run_utils import onevoice_from_childes, onevoice_from_timit, validate_onevoice
from experiments.cross_task_generalization.common.validators import validate_task2_final, validate_task3_final


def count_task2_events(gold_dir: Path) -> Counter[str]:
    c: Counter[str] = Counter()
    for path in gold_dir.glob("*.gold.json"):
        obj = load_json(path)
        for event in obj.get("events", []):
            c[event["event_type"]] += 1
    return c


def count_task3_events(gold_dir: Path) -> Counter[str]:
    c: Counter[str] = Counter()
    for path in gold_dir.glob("*.gold.json"):
        obj = load_json(path)
        for event in obj.get("events", []):
            c[event["event_type"]] += 1
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cross-task setup without API calls.")
    parser.add_argument("--task2-manifest", type=Path, default=Path("experiments/cross_task_generalization/manifests/task2_childes_manifest_enhanced.jsonl"))
    parser.add_argument("--task3-manifest", type=Path, default=Path("experiments/cross_task_generalization/manifests/task3_timit_manifest.jsonl"))
    parser.add_argument("--task2-gold-dir", type=Path, default=Path("experiments/cross_task_generalization/gold/task2_childes_enhanced"))
    parser.add_argument("--task3-gold-dir", type=Path, default=Path("experiments/cross_task_generalization/gold/task3_timit"))
    parser.add_argument("--out", type=Path, default=Path("experiments/cross_task_generalization/reports/setup_validation.json"))
    args = parser.parse_args()

    task2_rows = load_jsonl(args.task2_manifest) if args.task2_manifest.exists() else []
    task3_rows = load_jsonl(args.task3_manifest) if args.task3_manifest.exists() else []
    errors: list[str] = []

    for path in sorted(args.task2_gold_dir.glob("*.gold.json")):
        errs = validate_task2_final(load_json(path))
        errors.extend([f"{path.name}: {e}" for e in errs])
    for path in sorted(args.task3_gold_dir.glob("*.gold.json")):
        errs = validate_task3_final(load_json(path))
        errors.extend([f"{path.name}: {e}" for e in errs])

    onevoice_checks: list[dict[str, Any]] = []
    if task2_rows:
        ok, log = validate_onevoice(onevoice_from_childes(task2_rows[0]), args.out.parent / "onevoice_check_task2")
        onevoice_checks.append({"task": "task2_childes", "ok": ok, "log_tail": log[-1000:]})
    if task3_rows:
        ok, log = validate_onevoice(onevoice_from_timit(task3_rows[0]), args.out.parent / "onevoice_check_task3")
        onevoice_checks.append({"task": "task3_timit", "ok": ok, "log_tail": log[-1000:]})

    report = {
        "ok": not errors and all(row["ok"] for row in onevoice_checks),
        "errors": errors,
        "task2": {
            "manifest_windows": len(task2_rows),
            "source_files": len({r.get("source_file") for r in task2_rows}),
            "gold_files": len(list(args.task2_gold_dir.glob("*.gold.json"))),
            "event_distribution": dict(sorted(count_task2_events(args.task2_gold_dir).items())),
        },
        "task3": {
            "manifest_bundles": len(task3_rows),
            "speakers": len({r.get("speaker_id") for r in task3_rows}),
            "gold_files": len(list(args.task3_gold_dir.glob("*.gold.json"))),
            "event_distribution": dict(sorted(count_task3_events(args.task3_gold_dir).items())),
        },
        "onevoice_checks": onevoice_checks,
    }
    write_json(args.out, report)
    print(f"Wrote setup validation report to {args.out}")
    if not report["ok"]:
        raise SystemExit("Setup validation failed")


if __name__ == "__main__":
    main()
