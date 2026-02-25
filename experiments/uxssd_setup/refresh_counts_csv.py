#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def count_final_outputs(run_dir: Path) -> int:
    return sum(1 for d in run_dir.iterdir() if d.is_dir() and (d / "final_output.json").exists())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_events(obj: dict[str, Any]) -> tuple[int, int]:
    mis = 0
    beh = 0

    utterances = obj.get("utterances")
    turns = obj.get("turns")

    if isinstance(utterances, list):
        for u in utterances:
            if isinstance(u, dict):
                mis += len(u.get("mispronunciations", []) or [])
                beh += len(u.get("behavioral_events", []) or [])
        return mis, beh

    if isinstance(turns, list):
        for t in turns:
            if isinstance(t, dict):
                mis += len(t.get("mispronunciations", []) or [])
                beh += len(t.get("behavioral_events", []) or [])
        return mis, beh

    return 0, 0


def count_from_run(run_dir: Path | None, record_id: str) -> tuple[int, int] | None:
    if run_dir is None:
        return None
    p = run_dir / record_id / "final_output.json"
    if not p.exists():
        return None
    try:
        return count_events(load_json(p))
    except Exception:
        return None


def pick_best_exp1_run(runs_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = runs_dir / explicit
        return p if p.exists() else None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("exp1_")]
    if not candidates:
        return None
    return max(candidates, key=count_final_outputs)


def pick_best_exp2_run(runs_dir: Path, explicit: str | None, required_manifest: str) -> Path | None:
    if explicit:
        p = runs_dir / explicit
        return p if p.exists() else None

    candidates: list[Path] = []
    for p in runs_dir.iterdir():
        if not (p.is_dir() and p.name.startswith("exp2_")):
            continue
        cfg = p / "run_config.json"
        if not cfg.exists():
            continue
        try:
            data = load_json(cfg)
        except Exception:
            continue
        manifest = str(data.get("manifest", "")).replace("\\", "/")
        if manifest == required_manifest:
            candidates.append(p)

    if not candidates:
        return None
    return max(candidates, key=count_final_outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh experiment event-count comparison CSV for gold vs Exp1 vs Exp2.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/uxssd_setup/runs"),
        help="Runs folder containing exp1_* and exp2_* folders.",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("experiments/uxssd_setup/gold_labels/templates"),
        help="Folder containing *.gold.json files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("experiments/uxssd_setup/runs/event_counts_gold_exp1_exp2.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--exp1-run",
        type=str,
        default="",
        help="Optional explicit exp1 run folder name.",
    )
    parser.add_argument(
        "--exp2-run",
        type=str,
        default="",
        help="Optional explicit exp2 run folder name.",
    )
    parser.add_argument(
        "--exp2-required-manifest",
        type=str,
        default="experiments/uxssd_setup/experiment2_onevoice_manifest.jsonl",
        help="Manifest path required when auto-selecting Exp2 run.",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    gold_dir = args.gold_dir
    out_csv = args.out_csv

    if not runs_dir.exists():
        raise SystemExit(f"Runs dir not found: {runs_dir}")
    if not gold_dir.exists():
        raise SystemExit(f"Gold dir not found: {gold_dir}")

    exp1_run = pick_best_exp1_run(runs_dir, args.exp1_run or None)
    exp2_run = pick_best_exp2_run(runs_dir, args.exp2_run or None, args.exp2_required_manifest)

    gold_files = sorted(gold_dir.glob("*.gold.json"))
    rows: list[dict[str, Any]] = []

    for gp in gold_files:
        record_id = gp.stem.replace(".gold", "")
        gold_mis, gold_beh = count_events(load_json(gp))
        exp1_counts = count_from_run(exp1_run, record_id)
        exp2_counts = count_from_run(exp2_run, record_id)

        rows.append(
            {
                "record_id": record_id,
                "gold_mispronunciation": gold_mis,
                "gold_behavioral_events": gold_beh,
                "exp1_mispronunciation": "" if exp1_counts is None else exp1_counts[0],
                "exp1_behavioral_events": "" if exp1_counts is None else exp1_counts[1],
                "exp2_mispronunciation": "" if exp2_counts is None else exp2_counts[0],
                "exp2_behavioral_events": "" if exp2_counts is None else exp2_counts[1],
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "record_id",
                "gold_mispronunciation",
                "gold_behavioral_events",
                "exp1_mispronunciation",
                "exp1_behavioral_events",
                "exp2_mispronunciation",
                "exp2_behavioral_events",
            ],
        )
        wr.writeheader()
        wr.writerows(rows)

    exp2_done = sum(1 for r in rows if str(r["exp2_mispronunciation"]).strip() != "")
    print(f"Wrote: {out_csv}")
    print(f"Using exp1 run: {exp1_run.name if exp1_run else '(none)'}")
    print(f"Using exp2 run: {exp2_run.name if exp2_run else '(none)'}")
    print(f"Exp2 filled records: {exp2_done}/{len(rows)}")


if __name__ == "__main__":
    main()
