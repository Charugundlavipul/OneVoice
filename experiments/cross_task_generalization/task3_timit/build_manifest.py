from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import rel, write_json, write_jsonl
from experiments.cross_task_generalization.task3_timit.parse_timit import find_timit_utterances, parse_utterance


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def compute_thresholds(root: Path, calibration_split: str, max_utts: int = 1200) -> dict[str, Any]:
    word_groups: dict[str, list[float]] = defaultdict(list)
    phone_groups: dict[str, list[float]] = defaultdict(list)
    all_word: list[float] = []
    all_phone: list[float] = []

    for txt_path in find_timit_utterances(root, calibration_split)[:max_utts]:
        utt = parse_utterance(txt_path)
        for word in utt["words"]:
            phone_count = max(1, len(word.get("phone_indices", [])))
            key = str(min(phone_count, 8))
            word_groups[key].append(word["duration"])
            all_word.append(word["duration"])
        for phone in utt["phones"]:
            phone_groups[phone["label"]].append(phone["duration"])
            all_phone.append(phone["duration"])

    thresholds = {
        "sample_rate": 16000,
        "calibration_split": calibration_split,
        "word_duration_global": {"p10": percentile(all_word, 0.10), "p90": percentile(all_word, 0.90)},
        "phone_duration_global": {"p10": percentile(all_phone, 0.10), "p90": percentile(all_phone, 0.90)},
        "word_duration_by_phone_count": {},
        "phone_duration_by_label": {},
    }
    for key, values in sorted(word_groups.items()):
        if len(values) >= 20:
            thresholds["word_duration_by_phone_count"][key] = {"p10": percentile(values, 0.10), "p90": percentile(values, 0.90), "n": len(values)}
    for key, values in sorted(phone_groups.items()):
        if len(values) >= 20:
            thresholds["phone_duration_by_label"][key] = {"p10": percentile(values, 0.10), "p90": percentile(values, 0.90), "n": len(values)}
    return thresholds


def utterance_payload(utt: dict[str, Any]) -> dict[str, Any]:
    return {
        "utt_id": utt["utt_id"],
        "speaker_id": utt["speaker_id"],
        "transcript": utt["transcript"],
        "duration": utt["duration"],
        "txt_path": rel(utt["txt_path"]),
        "wrd_path": rel(utt["wrd_path"]),
        "phn_path": rel(utt["phn_path"]),
        "word_intervals": [
            {
                "word_index": i,
                "word": w["label"],
                "start": w["start"],
                "end": w["end"],
                "duration": w["duration"],
                "phone_indices": w.get("phone_indices", []),
            }
            for i, w in enumerate(utt["words"])
        ],
        "phone_intervals": [
            {
                "phone_index": i,
                "phone": p["label"],
                "start": p["start"],
                "end": p["end"],
                "duration": p["duration"],
                "word_index": p.get("word_index"),
            }
            for i, p in enumerate(utt["phones"])
        ],
    }


def build_manifest(root: Path, out_path: Path, thresholds_path: Path, target_bundles: int, eval_split: str, calibration_split: str) -> list[dict[str, Any]]:
    thresholds = compute_thresholds(root, calibration_split)
    write_json(thresholds_path, thresholds)

    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for txt_path in find_timit_utterances(root, eval_split):
        by_speaker[txt_path.parent.name.upper()].append(txt_path)

    rows: list[dict[str, Any]] = []
    for speaker_id in sorted(by_speaker):
        paths = by_speaker[speaker_id]
        if len(paths) < 5:
            continue
        selected = paths[:5]
        utterances = [utterance_payload(parse_utterance(p)) for p in selected]
        rows.append(
            {
                "task": "task3_timit",
                "bundle_id": f"timit_{speaker_id}",
                "dataset": "TIMIT",
                "speaker_id": speaker_id,
                "eval_split": eval_split,
                "calibration_split": calibration_split,
                "thresholds_path": rel(thresholds_path),
                "utterance_count": len(utterances),
                "utterances": utterances,
            }
        )
        if len(rows) >= target_bundles:
            break
    write_jsonl(out_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task 3 TIMIT manifest.")
    parser.add_argument("--root", type=Path, default=Path("TIMIT"))
    parser.add_argument("--out", type=Path, default=Path("experiments/cross_task_generalization/manifests/task3_timit_manifest.jsonl"))
    parser.add_argument("--thresholds-out", type=Path, default=Path("experiments/cross_task_generalization/manifests/task3_timit_thresholds.json"))
    parser.add_argument("--target-bundles", type=int, default=40)
    parser.add_argument("--eval-split", type=str, default="TEST")
    parser.add_argument("--calibration-split", type=str, default="TRAIN")
    args = parser.parse_args()
    rows = build_manifest(args.root, args.out, args.thresholds_out, args.target_bundles, args.eval_split, args.calibration_split)
    print(f"Wrote {len(rows)} bundles to {args.out}")
    print(f"Wrote thresholds to {args.thresholds_out}")


if __name__ == "__main__":
    main()
