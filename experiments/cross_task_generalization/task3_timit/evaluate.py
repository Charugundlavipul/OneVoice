from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import load_json, write_json
from experiments.cross_task_generalization.common.metrics import count_proxy_f1, mae, safe_div
from experiments.cross_task_generalization.common.validators import TASK3_EVENT_TYPES, validate_task3_final


PHONE_EVENTS = {"long_phone", "short_phone", "closure_segment", "pause_silence_segment", "glottal_stop"}


def count_by_utt_type(obj: dict[str, Any]) -> Counter[tuple[str, str]]:
    c: Counter[tuple[str, str]] = Counter()
    for event in obj.get("events", []) or []:
        if isinstance(event, dict) and event.get("event_type") in TASK3_EVENT_TYPES and isinstance(event.get("utt_id"), str):
            c[(event["utt_id"], event["event_type"])] += 1
    return c


def totals_by_utt(obj: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for event in obj.get("events", []) or []:
        if isinstance(event, dict) and isinstance(event.get("utt_id"), str):
            out[event["utt_id"]] += 1
    return out


def sorted_events(obj: dict[str, Any], utt_id: str, event_type: str) -> list[dict[str, Any]]:
    return sorted(
        [
            e
            for e in obj.get("events", []) or []
            if isinstance(e, dict) and e.get("utt_id") == utt_id and e.get("event_type") == event_type
        ],
        key=lambda e: (float(e.get("start") or 0.0), float(e.get("end") or 0.0), int(e.get("phone_index") or -1), int(e.get("word_index") or -1)),
    )


def matched_event_pairs(gold: dict[str, Any], pred: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    keys = {
        (e.get("utt_id"), e.get("event_type"))
        for e in (gold.get("events") or [])
        if isinstance(e, dict) and isinstance(e.get("utt_id"), str) and isinstance(e.get("event_type"), str)
    }
    keys |= {
        (e.get("utt_id"), e.get("event_type"))
        for e in (pred.get("events") or [])
        if isinstance(e, dict) and isinstance(e.get("utt_id"), str) and isinstance(e.get("event_type"), str)
    }
    for utt_id, event_type in sorted(keys):
        gold_events = sorted_events(gold, utt_id, event_type)
        pred_events = sorted_events(pred, utt_id, event_type)
        for g, p in zip(gold_events, pred_events):
            pairs.append((g, p))
    return pairs


def evaluate_pair(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    errors = validate_task3_final(pred)
    gold_counts = count_by_utt_type(gold)
    pred_counts = count_by_utt_type(pred)
    f1 = count_proxy_f1(gold_counts, pred_counts)

    utt_ids = [row["utt_id"] for row in gold.get("utterance_summaries", []) if isinstance(row, dict)]
    gold_totals = totals_by_utt(gold)
    pred_totals = totals_by_utt(pred)
    total_mae = mae([gold_totals.get(u, 0) for u in utt_ids], [pred_totals.get(u, 0) for u in utt_ids])

    phone_matches = 0
    phone_link_correct = 0
    boundary_errors = []
    for g, p in matched_event_pairs(gold, pred):
        if g.get("event_type") in PHONE_EVENTS:
            phone_matches += 1
            if p.get("word_index") == g.get("word_index"):
                phone_link_correct += 1
        try:
            boundary_errors.append((abs(float(p.get("start") or 0.0) - float(g.get("start") or 0.0)) + abs(float(p.get("end") or 0.0) - float(g.get("end") or 0.0))) / 2.0)
        except Exception:
            pass

    pred_phone_events = [e for e in pred.get("events", []) or [] if isinstance(e, dict) and e.get("event_type") in PHONE_EVENTS]
    invalid_links = [e for e in pred_phone_events if e.get("word_index") is None or e.get("word_index") == ""]

    return {
        "bundle_id": gold["bundle_id"],
        "valid": not errors,
        "validation_errors": errors,
        "event_count_precision": f1["precision"],
        "event_count_recall": f1["recall"],
        "event_count_f1": f1["f1"],
        "event_count_mae": total_mae,
        "word_link_accuracy": safe_div(phone_link_correct, phone_matches),
        "invalid_link_rate": safe_div(len(invalid_links), len(pred_phone_events)),
        "boundary_error_ms": (sum(boundary_errors) / len(boundary_errors) * 1000.0) if boundary_errors else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 3 predictions against gold.")
    parser.add_argument("--gold-dir", type=Path, default=Path("experiments/cross_task_generalization/gold/task3_timit"))
    parser.add_argument("--pred-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for gold_path in sorted(args.gold_dir.glob("*.gold.json")):
        bundle_id = gold_path.stem.replace(".gold", "")
        pred_path = args.pred_run / bundle_id / "final_output.json"
        if not pred_path.exists():
            continue
        rows.append(evaluate_pair(load_json(gold_path), load_json(pred_path)))

    out = args.out or (args.pred_run / "task3_metrics.json")
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
