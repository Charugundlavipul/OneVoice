#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.uxssd_setup.build_experiment_helpers import (
    ROOT,
    load_behavioral_proxy,
    load_mispronunciation_proxy,
    load_utterances,
)


OUT_DIR = ROOT / "experiments" / "uxssd_setup"


def infer_cohort(mis: int, beh: int) -> str:
    if mis > beh:
        return "mispronunciation_heavy"
    if beh > mis:
        return "behavioral_heavy"
    return "balanced"


def build_rolling_window_candidates(
    utterances: dict[str, dict[str, Any]],
    mis_proxy: dict[str, int],
    beh_proxy: dict[str, int],
    *,
    min_duration_s: float,
    max_duration_s: float,
    min_utts: int,
    min_beh: int,
    max_beh: int,
    min_mis: int,
    max_mis: int,
) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for meta in utterances.values():
        if float(meta.get("duration_s", 0.0) or 0.0) <= 0:
            continue
        group_key = f"{meta['speaker_short']}-{meta['session']}"
        by_group[group_key].append(meta)

    for group_key in by_group:
        by_group[group_key].sort(key=lambda x: str(x.get("utt", "")))

    candidates: list[dict[str, Any]] = []
    for group_key, rows in by_group.items():
        if not rows:
            continue
        speaker_short = str(rows[0]["speaker_short"])
        speaker_id = str(rows[0]["speaker_id"])
        session_group = str(rows[0]["session"])
        n = len(rows)

        for start_idx in range(n):
            cur_dur = 0.0
            cur_mis = 0
            cur_beh = 0
            for end_idx in range(start_idx, n):
                row = rows[end_idx]
                cur_dur += float(row["duration_s"])
                cur_mis += int(mis_proxy.get(row["utt"], 0))
                cur_beh += int(beh_proxy.get(row["utt"], 0))
                cur_cnt = end_idx - start_idx + 1

                if cur_dur > max_duration_s:
                    break
                if cur_dur < min_duration_s or cur_cnt < min_utts:
                    continue
                if not (min_mis <= cur_mis <= max_mis and min_beh <= cur_beh <= max_beh):
                    continue

                cohort = infer_cohort(cur_mis, cur_beh)
                bundle_id = f"{speaker_short}_{session_group}_w{start_idx + 1:03d}_{end_idx + 1:03d}"
                candidates.append(
                    {
                        "bundle_id": bundle_id,
                        "speaker_id": speaker_id,
                        "speaker_short": speaker_short,
                        "session_group": session_group,
                        "duration_s": cur_dur,
                        "utterance_count": cur_cnt,
                        "mispronunciation_proxy_events": cur_mis,
                        "behavioral_event_proxy_events": cur_beh,
                        "cohort": cohort,
                        "utterances": rows[start_idx : end_idx + 1],
                        "_group_key": group_key,
                        "_start_idx": start_idx,
                        "_end_idx": end_idx,
                    }
                )
    return candidates


def overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    if str(a.get("_group_key", "")) != str(b.get("_group_key", "")):
        return 0.0
    a_start = int(a["_start_idx"])
    a_end = int(a["_end_idx"])
    b_start = int(b["_start_idx"])
    b_end = int(b["_end_idx"])
    inter = min(a_end, b_end) - max(a_start, b_start) + 1
    if inter <= 0:
        return 0.0
    a_len = a_end - a_start + 1
    b_len = b_end - b_start + 1
    return float(inter) / float(min(a_len, b_len))


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    target_records: int,
    max_overlap_ratio: float,
    max_records_per_speaker: int,
    min_mis_heavy: int,
    min_beh_heavy: int,
    min_mis: int,
    max_mis: int,
    min_beh: int,
    max_beh: int,
) -> list[dict[str, Any]]:
    mid_mis = (min_mis + max_mis) / 2.0
    mid_beh = (min_beh + max_beh) / 2.0

    for c in candidates:
        mis = int(c["mispronunciation_proxy_events"])
        beh = int(c["behavioral_event_proxy_events"])
        dur = float(c["duration_s"])
        c["_quality_score"] = abs(mis - mid_mis) + abs(beh - mid_beh) + 0.01 * abs(dur - 152.5)

    candidates = sorted(candidates, key=lambda c: (c["_quality_score"], c["bundle_id"]))

    selected: list[dict[str, Any]] = []
    selected_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_by_speaker: dict[str, int] = defaultdict(int)
    selected_ids: set[str] = set()

    def is_eligible(c: dict[str, Any], use_overlap: bool = True, use_speaker_cap: bool = True) -> bool:
        bid = str(c["bundle_id"])
        if bid in selected_ids:
            return False
        if use_speaker_cap and max_records_per_speaker > 0:
            speaker = str(c.get("speaker_id", ""))
            if selected_by_speaker[speaker] >= max_records_per_speaker:
                return False
        if not use_overlap:
            return True
        group_key = str(c["_group_key"])
        for prior in selected_by_group[group_key]:
            if overlap_ratio(c, prior) > max_overlap_ratio:
                return False
        return True

    def choose_best(pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not pool:
            return None
        def overlap_penalty(c: dict[str, Any]) -> float:
            group_key = str(c["_group_key"])
            priors = selected_by_group[group_key]
            if not priors:
                return 0.0
            return max(overlap_ratio(c, p) for p in priors)

        pool = sorted(
            pool,
            key=lambda c: (
                selected_by_speaker[str(c.get("speaker_id", ""))],
                len(selected_by_group[str(c["_group_key"])]),
                overlap_penalty(c),
                c["_quality_score"],
                abs(int(c["mispronunciation_proxy_events"]) - int(c["behavioral_event_proxy_events"])),
            ),
        )
        return pool[0]

    def add_one(c: dict[str, Any]) -> None:
        selected.append(c)
        selected_ids.add(str(c["bundle_id"]))
        selected_by_group[str(c["_group_key"])].append(c)
        selected_by_speaker[str(c.get("speaker_id", ""))] += 1

    def count_cohort(name: str) -> int:
        return sum(1 for x in selected if str(x.get("cohort", "")) == name)

    for cohort_name, quota in (
        ("mispronunciation_heavy", min_mis_heavy),
        ("behavioral_heavy", min_beh_heavy),
    ):
        while count_cohort(cohort_name) < quota and len(selected) < target_records:
            pool = [
                c
                for c in candidates
                if str(c.get("cohort", "")) == cohort_name and is_eligible(c, use_overlap=True, use_speaker_cap=True)
            ]
            pick = choose_best(pool)
            if pick is None:
                break
            add_one(pick)

    while len(selected) < target_records:
        pool = [c for c in candidates if is_eligible(c, use_overlap=True, use_speaker_cap=True)]
        pick = choose_best(pool)
        if pick is None:
            break
        add_one(pick)

    # Fallback 1: relax speaker cap, keep overlap.
    while len(selected) < target_records:
        pool = [c for c in candidates if is_eligible(c, use_overlap=True, use_speaker_cap=False)]
        pick = choose_best(pool)
        if pick is None:
            break
        add_one(pick)

    # Fallback 2: relax overlap and speaker cap to guarantee target fill if possible.
    while len(selected) < target_records:
        pool = [c for c in candidates if is_eligible(c, use_overlap=False, use_speaker_cap=False)]
        pick = choose_best(pool)
        if pick is None:
            break
        add_one(pick)

    for c in selected:
        c.pop("_quality_score", None)

    selected.sort(key=lambda b: (b["speaker_short"], b["session_group"], b["bundle_id"]))
    return selected


def write_outputs(
    selected: list[dict[str, Any]],
    mis_proxy: dict[str, int],
    mis_targets: dict[str, list[dict[str, Any]]],
    beh_proxy: dict[str, int],
    beh_details: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "selected_bundles.csv"
    utt_csv = out_dir / "selected_utterances.csv"
    exp1_jsonl = out_dir / "experiment1_agnostic_manifest.jsonl"
    exp2_jsonl = out_dir / "experiment2_onevoice_manifest.jsonl"
    mis_proxy_csv = out_dir / "mispronunciation_proxy_targets.csv"
    beh_proxy_csv = out_dir / "behavioral_proxy_targets.csv"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "bundle_id",
                "speaker_id",
                "session_group",
                "duration_s",
                "utterance_count",
                "mispronunciation_proxy_events",
                "behavioral_event_proxy_events",
                "cohort",
                "note",
            ]
        )
        for b in selected:
            wr.writerow(
                [
                    b["bundle_id"],
                    b["speaker_id"],
                    b["session_group"],
                    f"{b['duration_s']:.3f}",
                    b["utterance_count"],
                    b["mispronunciation_proxy_events"],
                    b["behavioral_event_proxy_events"],
                    b["cohort"],
                    "small-scale setup (behavioral 5-10, mispronunciation 5-10)",
                ]
            )

    with utt_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "bundle_id",
                "cohort",
                "utt",
                "speaker_id",
                "duration_s",
                "mispronunciation_proxy_events",
                "behavioral_event_proxy_events",
                "audio_wav",
                "transcript_txt",
                "param_file",
                "word_labels_textgrid",
                "phone_labels_textgrid",
                "speaker_labels_textgrid",
                "reference_word_textgrid",
                "reference_phone_textgrid",
                "reference_speaker_textgrid",
                "slt_labels_textgrid",
                "onevoice_json",
            ]
        )
        for b in selected:
            for u in b["utterances"]:
                p = u["paths"]
                wr.writerow(
                    [
                        b["bundle_id"],
                        b["cohort"],
                        u["utt"],
                        u["speaker_id"],
                        f"{u['duration_s']:.3f}",
                        int(mis_proxy.get(u["utt"], 0)),
                        int(beh_proxy.get(u["utt"], 0)),
                        p["audio_wav"],
                        p["transcript_txt"],
                        p["param_file"],
                        p["word_labels_textgrid"],
                        p["phone_labels_textgrid"],
                        p["speaker_labels_textgrid"],
                        p["reference_word_textgrid"],
                        p["reference_phone_textgrid"],
                        p["reference_speaker_textgrid"],
                        p["slt_labels_textgrid"],
                        p["onevoice_json"],
                    ]
                )

    with exp1_jsonl.open("w", encoding="utf-8") as f:
        for b in selected:
            entry = {
                "record_id": b["bundle_id"],
                "cohort": b["cohort"],
                "dataset": "uxssd",
                "mode": "experiment1_onevoice_agnostic",
                "input_policy": {
                    "use_onevoice_schema": False,
                    "validator_in_loop": False,
                    "agent_outputs_format": "free_or_minimal_json",
                },
                "bundle_duration_s": b["duration_s"],
                "utterances": [
                    {
                        "utt": u["utt"],
                        "speaker_id": u["speaker_id"],
                        "duration_s": u["duration_s"],
                        "files": {
                            "audio_wav": u["paths"].get("audio_wav", ""),
                            "transcript_txt": u["paths"].get("transcript_txt", ""),
                            "param_file": u["paths"].get("param_file", ""),
                        },
                    }
                    for u in b["utterances"]
                ],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with exp2_jsonl.open("w", encoding="utf-8") as f:
        for b in selected:
            entry = {
                "record_id": b["bundle_id"],
                "cohort": b["cohort"],
                "dataset": "uxssd",
                "mode": "experiment2_onevoice_pipeline",
                "input_policy": {
                    "use_onevoice_schema": True,
                    "validator_in_loop": True,
                    "validator_command": "python validate_onevoice.py <generated_onevoice_record.json> --mode full",
                    "max_repair_rounds_per_agent": 3,
                },
                "bundle_duration_s": b["duration_s"],
                "utterances": [
                    {
                        "utt": u["utt"],
                        "speaker_id": u["speaker_id"],
                        "duration_s": u["duration_s"],
                        "onevoice_json": u["paths"]["onevoice_json"],
                    }
                    for u in b["utterances"]
                ],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    selected_utts = sorted({u["utt"] for b in selected for u in b["utterances"]})
    with mis_proxy_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["utt", "mispronunciation_proxy_events", "target_details_json"])
        for utt in selected_utts:
            wr.writerow([utt, int(mis_proxy.get(utt, 0)), json.dumps(mis_targets.get(utt, []), ensure_ascii=False)])

    with beh_proxy_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["utt", "behavioral_event_proxy_events", "proxy_details_json"])
        for utt in selected_utts:
            wr.writerow([utt, int(beh_proxy.get(utt, 0)), json.dumps(beh_details.get(utt, {}), ensure_ascii=False)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UXSSD experiment manifests.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory for experiment setup.")
    parser.add_argument("--target-records", type=int, default=15, help="Target number of selected records.")
    parser.add_argument("--min-beh", type=int, default=5, help="Minimum behavioral proxy events per bundle.")
    parser.add_argument("--max-beh", type=int, default=10, help="Maximum behavioral proxy events per bundle.")
    parser.add_argument("--min-mis", type=int, default=5, help="Minimum mispronunciation proxy events per bundle.")
    parser.add_argument("--max-mis", type=int, default=10, help="Maximum mispronunciation proxy events per bundle.")
    parser.add_argument("--min-duration-s", type=float, default=120.0, help="Minimum record duration in seconds.")
    parser.add_argument("--max-duration-s", type=float, default=185.0, help="Maximum record duration in seconds.")
    parser.add_argument("--min-utts", type=int, default=8, help="Minimum utterances per record.")
    parser.add_argument(
        "--max-overlap-ratio",
        type=float,
        default=0.55,
        help="Maximum intra-session utterance overlap ratio allowed between selected windows.",
    )
    parser.add_argument(
        "--max-records-per-speaker",
        type=int,
        default=4,
        help="Max records per speaker before fallback relaxation.",
    )
    parser.add_argument(
        "--min-mis-heavy",
        type=int,
        default=5,
        help="Minimum number of mispronunciation_heavy records in final selection.",
    )
    parser.add_argument(
        "--min-beh-heavy",
        type=int,
        default=5,
        help="Minimum number of behavioral_heavy records in final selection.",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional hard cap after selection; 0 means no cap.")
    args = parser.parse_args()

    utterances = load_utterances()
    mis_proxy, mis_targets = load_mispronunciation_proxy()
    beh_proxy, beh_details = load_behavioral_proxy(utterances)

    candidates = build_rolling_window_candidates(
        utterances=utterances,
        mis_proxy=mis_proxy,
        beh_proxy=beh_proxy,
        min_duration_s=float(args.min_duration_s),
        max_duration_s=float(args.max_duration_s),
        min_utts=int(args.min_utts),
        min_beh=int(args.min_beh),
        max_beh=int(args.max_beh),
        min_mis=int(args.min_mis),
        max_mis=int(args.max_mis),
    )

    selected = select_candidates(
        candidates=candidates,
        target_records=max(0, int(args.target_records)),
        max_overlap_ratio=float(args.max_overlap_ratio),
        max_records_per_speaker=max(0, int(args.max_records_per_speaker)),
        min_mis_heavy=max(0, int(args.min_mis_heavy)),
        min_beh_heavy=max(0, int(args.min_beh_heavy)),
        min_mis=int(args.min_mis),
        max_mis=int(args.max_mis),
        min_beh=int(args.min_beh),
        max_beh=int(args.max_beh),
    )

    if args.max_records > 0:
        selected = selected[: args.max_records]

    write_outputs(selected, mis_proxy, mis_targets, beh_proxy, beh_details, args.out_dir)

    cohort_counts = {
        "mispronunciation_heavy": sum(1 for b in selected if b.get("cohort") == "mispronunciation_heavy"),
        "behavioral_heavy": sum(1 for b in selected if b.get("cohort") == "behavioral_heavy"),
        "balanced": sum(1 for b in selected if b.get("cohort") == "balanced"),
    }

    print(f"Output dir: {args.out_dir}")
    print(f"Candidate rolling windows (all): {len(candidates)}")
    print(
        "Selected records: "
        f"{len(selected)} using ranges "
        f"beh=[{args.min_beh},{args.max_beh}], mis=[{args.min_mis},{args.max_mis}]"
    )
    print(
        "Cohorts: "
        f"mispronunciation_heavy={cohort_counts['mispronunciation_heavy']}, "
        f"behavioral_heavy={cohort_counts['behavioral_heavy']}, "
        f"balanced={cohort_counts['balanced']}"
    )
    for b in selected:
        print(
            f" - {b['bundle_id']}: mis={b['mispronunciation_proxy_events']}, "
            f"beh={b['behavioral_event_proxy_events']}, dur={b['duration_s']:.3f}s, utts={b['utterance_count']}"
        )


if __name__ == "__main__":
    main()
