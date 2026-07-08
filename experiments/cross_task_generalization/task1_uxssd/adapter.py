from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import write_json


LEGACY_ROOT = Path("experiments/legacy/uxssd_setup")


def parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8", errors="replace") if report_path.exists() else ""
    overall_text = text.split("### 6.3 Overall", 1)[1] if "### 6.3 Overall" in text else text
    out: dict[str, Any] = {
        "task": "task1_uxssd",
        "legacy_root": LEGACY_ROOT.as_posix(),
        "report_path": report_path.as_posix(),
        "source": "archived legacy UXSSD report",
    }
    m = re.search(r"Exp1: Precision ([0-9.]+), Recall ([0-9.]+), F1 ([0-9.]+).*?Exp2: Precision ([0-9.]+), Recall ([0-9.]+), F1 ([0-9.]+)", overall_text, re.S)
    if m:
        out["overall_proxy"] = {
            "exp1": {"precision": float(m.group(1)), "recall": float(m.group(2)), "f1": float(m.group(3))},
            "exp2": {"precision": float(m.group(4)), "recall": float(m.group(5)), "f1": float(m.group(6))},
        }
    m = re.search(r"Exp1 MAE: ([0-9.]+).*?Exp2 MAE: ([0-9.]+)", overall_text, re.S)
    if m:
        out.setdefault("overall_proxy", {}).setdefault("exp1", {})["mae"] = float(m.group(1))
        out.setdefault("overall_proxy", {}).setdefault("exp2", {})["mae"] = float(m.group(2))
    return out


def summarize_counts(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {"counts_csv": csv_path.as_posix(), "available": False}
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return {"counts_csv": csv_path.as_posix(), "available": True, "record_count": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Task 1 UXSSD legacy summary into the cross-task report folder.")
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT)
    parser.add_argument("--out", type=Path, default=Path("experiments/cross_task_generalization/reports/task1_uxssd_legacy_summary.json"))
    args = parser.parse_args()
    report = parse_report(args.legacy_root / "EXPERIMENT_REPORT_FOR_PAPER.md")
    report["counts"] = summarize_counts(args.legacy_root / "runs" / "event_counts_gold_exp1_exp2.csv")
    write_json(args.out, report)
    print(f"Wrote Task 1 summary to {args.out}")


if __name__ == "__main__":
    main()
