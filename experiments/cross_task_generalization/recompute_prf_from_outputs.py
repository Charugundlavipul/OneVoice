from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import load_json, write_json
from experiments.cross_task_generalization.common.metrics import count_proxy_f1, prf_from_counts
from experiments.cross_task_generalization.task2_childes.evaluate import event_counts as task2_event_counts
from experiments.cross_task_generalization.task3_timit.evaluate import count_by_utt_type as task3_event_counts


RUNS_ROOT = Path("experiments/cross_task_generalization/runs")
REPORTS_ROOT = Path("experiments/cross_task_generalization/reports")
TASK1_RUNS_ROOT = Path("experiments/legacy/uxssd_setup/runs")
TASK2_GOLD_DIR = Path("experiments/cross_task_generalization/gold/task2_childes_enhanced")
TASK3_GOLD_DIR = Path("experiments/cross_task_generalization/gold/task3_timit")


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def harmonic(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def parse_task_run_name(name: str) -> tuple[str, str] | None:
    parts = name.split("_")
    if len(parts) < 4:
        return None
    if parts[0] == "task2" and parts[1] in {"c0", "c1", "c2"}:
        return parts[1], "_".join(parts[2:-2])
    if parts[0] == "task3" and parts[1] == "clean" and parts[2] in {"c0", "c1", "c2"}:
        return parts[2], "_".join(parts[3:-2])
    return None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(r["tp"]) for r in records)
    pred = sum(int(r["pred"]) for r in records)
    gold = sum(int(r["gold"]) for r in records)
    micro = prf_from_counts(tp, pred, gold)
    return {
        "records": len(records),
        "micro": {
            "tp": tp,
            "pred": pred,
            "gold": gold,
            "precision": micro["precision"],
            "recall": micro["recall"],
            "f1": micro["f1"],
        },
        "macro": {
            "precision": safe_mean([float(r["precision"]) for r in records]),
            "recall": safe_mean([float(r["recall"]) for r in records]),
            "f1": safe_mean([float(r["f1"]) for r in records]),
        },
    }


def task1_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    csv_files = [
        ("gpt-5-mini", TASK1_RUNS_ROOT / "event_counts_gpt-5-mini.csv"),
        ("gpt-5.4-mini", TASK1_RUNS_ROOT / "event_counts_gpt-5.4-mini.csv"),
        ("claude-haiku-4-5-20251001", TASK1_RUNS_ROOT / "event_counts_claude-haiku.csv"),
    ]
    conditions = ["exp1", "exp1b", "exp2", "exp2b"]
    for model, csv_path in csv_files:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for condition in conditions:
            records: list[dict[str, Any]] = []
            for row in rows:
                gold = int(row.get("gold_mispronunciation") or 0) + int(row.get("gold_behavioral_events") or 0)
                pred = int(row.get(f"{condition}_mispronunciation") or 0) + int(row.get(f"{condition}_behavioral_events") or 0)
                tp = min(gold, pred)
                prf = prf_from_counts(tp, pred, gold)
                records.append(
                    {
                        "record_id": row.get("record_id"),
                        "tp": tp,
                        "pred": pred,
                        "gold": gold,
                        "precision": prf["precision"],
                        "recall": prf["recall"],
                        "f1": prf["f1"],
                    }
                )
            out.append(
                {
                    "task": "task1_uxssd",
                    "model": model,
                    "condition": condition,
                    "source": csv_path.as_posix(),
                    **summarize_records(records),
                }
            )
    return out


def prf_record(gold_counts: Counter[tuple[Any, ...]], pred_counts: Counter[tuple[Any, ...]], record_id: str) -> dict[str, Any]:
    f1 = count_proxy_f1(gold_counts, pred_counts)
    return {
        "record_id": record_id,
        "tp": int(f1["tp"]),
        "pred": int(f1["pred"]),
        "gold": int(f1["gold"]),
        "precision": f1["precision"],
        "recall": f1["recall"],
        "f1": f1["f1"],
    }


def task2_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run_dir in sorted(RUNS_ROOT.glob("task2_c*_*")):
        if not run_dir.is_dir():
            continue
        parsed = parse_task_run_name(run_dir.name)
        if not parsed:
            continue
        condition, model = parsed
        records: list[dict[str, Any]] = []
        for gold_path in sorted(TASK2_GOLD_DIR.glob("*.gold.json")):
            window_id = gold_path.stem.replace(".gold", "")
            pred_path = run_dir / window_id / "final_output.json"
            if not pred_path.exists():
                continue
            gold = load_json(gold_path)
            pred = load_json(pred_path)
            records.append(prf_record(task2_event_counts(gold), task2_event_counts(pred), window_id))
        out.append(
            {
                "task": "task2_childes",
                "model": model,
                "condition": condition,
                "source": run_dir.as_posix(),
                **summarize_records(records),
            }
        )
    return out


def task3_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run_dir in sorted(RUNS_ROOT.glob("task3_clean_c*_*")):
        if not run_dir.is_dir():
            continue
        parsed = parse_task_run_name(run_dir.name)
        if not parsed:
            continue
        condition, model = parsed
        records: list[dict[str, Any]] = []
        for gold_path in sorted(TASK3_GOLD_DIR.glob("*.gold.json")):
            bundle_id = gold_path.stem.replace(".gold", "")
            pred_path = run_dir / bundle_id / "final_output.json"
            if not pred_path.exists():
                continue
            gold = load_json(gold_path)
            pred = load_json(pred_path)
            records.append(prf_record(task3_event_counts(gold), task3_event_counts(pred), bundle_id))
        out.append(
            {
                "task": "task3_timit",
                "model": model,
                "condition": condition,
                "source": run_dir.as_posix(),
                **summarize_records(records),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute precision/recall/F1 directly from generated outputs.")
    parser.add_argument("--out", type=Path, default=REPORTS_ROOT / "proper_prf_from_generated_outputs.json")
    args = parser.parse_args()

    results = task1_rows() + task2_rows() + task3_rows()
    payload = {
        "description": "Aggregate precision/recall/F1 recomputed from generated outputs by task, model, and condition. Micro PRF is computed from total TP/pred/gold; macro PRF is the average of per-record PRF.",
        "results": results,
    }
    write_json(args.out, payload)
    print(f"Wrote {len(results)} result rows to {args.out}")


if __name__ == "__main__":
    main()
