from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import load_json, write_json
from experiments.cross_task_generalization.common.metrics import count_proxy_f1, greedy_label_mapping, mae, safe_div
from experiments.cross_task_generalization.common.validators import TASK2_EVENT_TYPES, validate_task2_final


def assignment_map(obj: dict[str, Any]) -> dict[int, str]:
    return {
        int(row["utt_index"]): str(row["pred_speaker_id"])
        for row in obj.get("utterance_speaker_assignments", [])
        if isinstance(row, dict) and isinstance(row.get("utt_index"), int)
    }


def event_counts(obj: dict[str, Any]) -> Counter[tuple[int, str]]:
    c: Counter[tuple[int, str]] = Counter()
    for event in obj.get("events", []) or []:
        if isinstance(event, dict) and event.get("event_type") in TASK2_EVENT_TYPES and isinstance(event.get("utt_index"), int):
            c[(event["utt_index"], event["event_type"])] += 1
    return c


def evaluate_pair(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    errors = validate_task2_final(pred)
    gold_assign = assignment_map(gold)
    pred_assign = assignment_map(pred)
    common_utts = sorted(gold_assign)
    mapping = greedy_label_mapping(
        [gold_assign[i] for i in common_utts],
        [pred_assign.get(i, "") for i in common_utts],
    )
    speaker_correct = sum(1 for i in common_utts if mapping.get(pred_assign.get(i, "")) == gold_assign[i])
    speaker_match = safe_div(speaker_correct, len(common_utts))

    gold_counts = event_counts(gold)
    pred_counts = event_counts(pred)
    f1 = count_proxy_f1(gold_counts, pred_counts)

    gold_totals = []
    pred_totals = []
    for idx in common_utts:
        gold_totals.append(sum(v for (utt, _typ), v in gold_counts.items() if utt == idx))
        pred_totals.append(sum(v for (utt, _typ), v in pred_counts.items() if utt == idx))

    attributable = 0
    attributed_correct = 0
    gold_event_speakers = Counter((e["utt_index"], e["event_type"], e["pred_speaker_id"]) for e in gold.get("events", []))
    for event in pred.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        key_any = (event.get("utt_index"), event.get("event_type"))
        if gold_counts.get(key_any, 0) <= 0:
            continue
        attributable += 1
        mapped = mapping.get(str(event.get("pred_speaker_id", "")), str(event.get("pred_speaker_id", "")))
        if gold_event_speakers.get((event.get("utt_index"), event.get("event_type"), mapped), 0) > 0:
            attributed_correct += 1

    return {
        "window_id": gold["window_id"],
        "valid": not errors,
        "validation_errors": errors,
        "speaker_match": speaker_match,
        "event_count_precision": f1["precision"],
        "event_count_recall": f1["recall"],
        "event_count_f1": f1["f1"],
        "event_count_mae": mae(gold_totals, pred_totals),
        "event_attribution_accuracy": safe_div(attributed_correct, attributable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 2 predictions against gold.")
    parser.add_argument("--gold-dir", type=Path, default=Path("experiments/cross_task_generalization/gold/task2_childes_enhanced"))
    parser.add_argument("--pred-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for gold_path in sorted(args.gold_dir.glob("*.gold.json")):
        window_id = gold_path.stem.replace(".gold", "")
        pred_path = args.pred_run / window_id / "final_output.json"
        if not pred_path.exists():
            continue
        rows.append(evaluate_pair(load_json(gold_path), load_json(pred_path)))

    out = args.out or (args.pred_run / "task2_metrics.json")
    write_json(out, rows)
    csv_path = out.with_suffix(".csv")
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"Wrote {len(rows)} metric rows to {out}")


if __name__ == "__main__":
    main()
