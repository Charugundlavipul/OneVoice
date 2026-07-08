from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import load_json, write_json


def summarize_metric_file(path: Path) -> dict[str, Any]:
    rows = load_json(path)
    if not rows:
        return {"path": path.as_posix(), "rows": 0}
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in {"valid"}]
    out: dict[str, Any] = {"path": path.as_posix(), "rows": len(rows)}
    for key in numeric_keys:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            out[f"mean_{key}"] = mean(vals)
    precision = out.get("mean_event_count_precision")
    recall = out.get("mean_event_count_recall")
    if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
        out["event_count_f1_from_mean_precision_recall"] = (
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    out["valid_rate"] = mean([1.0 if r.get("valid") else 0.0 for r in rows])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize task metric JSON files.")
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/cross_task_generalization/runs"))
    parser.add_argument("--out", type=Path, default=Path("experiments/cross_task_generalization/reports/cross_task_summary.json"))
    args = parser.parse_args()

    summaries = []
    for path in sorted(args.runs_root.rglob("task*_metrics.json")):
        summaries.append(summarize_metric_file(path))
    legacy_runs_dir = args.runs_root.parents[1] / "legacy" / "uxssd_setup" / "runs"
    task1_csvs = [
        ("gpt-5-mini", legacy_runs_dir / "event_counts_gpt-5-mini.csv"),
        ("gpt-5.4-mini", legacy_runs_dir / "event_counts_gpt-5.4-mini.csv"),
        ("claude-haiku-4-5-20251001", legacy_runs_dir / "event_counts_claude-haiku.csv"),
    ]
    
    for model_name, csv_file in task1_csvs:
        if not csv_file.exists():
            continue
        rows = []
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        
        def to_int(val):
            return int(val) if val and str(val).strip() else 0
            
        KEYS = ["exp1", "exp1b", "exp2", "exp2b"]
        gold_tot_sum = sum(to_int(r.get("gold_mispronunciation")) + to_int(r.get("gold_behavioral_events")) for r in rows)
        
        for k in KEYS:
            pred_tot = tp_tot = mae_sum = 0
            for r in rows:
                g_mis = to_int(r.get("gold_mispronunciation"))
                g_beh = to_int(r.get("gold_behavioral_events"))
                g_tot = g_mis + g_beh
                
                mis = to_int(r.get(f"{k}_mispronunciation"))
                beh = to_int(r.get(f"{k}_behavioral_events"))
                tot = mis + beh
                
                pred_tot += tot
                tp_tot += min(g_tot, tot)
                mae_sum += abs(g_tot - tot)
                
            p = tp_tot / pred_tot if pred_tot > 0 else 0.0
            rc = tp_tot / gold_tot_sum if gold_tot_sum > 0 else 0.0
            f1 = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
            mae = mae_sum / len(rows) if rows else 0.0
            
            summaries.append({
                "event_count_f1_from_mean_precision_recall": f1,
                "path": f"task1_{k}_{model_name}",
                "rows": len(rows),
                "mean_event_count_precision": p,
                "mean_event_count_recall": rc,
                "mean_event_count_f1": f1,
                "mean_event_count_mae": mae,
                "valid_rate": 1.0
            })

    write_json(args.out, summaries)
    csv_path = args.out.with_suffix(".csv")
    if summaries:
        keys = sorted({k for row in summaries for k in row})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(summaries)
    print(f"Wrote {len(summaries)} summaries to {args.out}")


if __name__ == "__main__":
    main()
