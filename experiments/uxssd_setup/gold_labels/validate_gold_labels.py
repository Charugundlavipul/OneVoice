#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


TIME_FMT = "%H:%M:%S:%f"


def parse_ts(ts: Any) -> float | None:
    if not isinstance(ts, str) or not ts:
        return None
    parts = ts.split(":")
    if len(parts) != 4:
        return None
    ms = parts[3]
    if len(ms) == 3:
        ts = f"{parts[0]}:{parts[1]}:{parts[2]}:{ms}000"
    try:
        dt = datetime.strptime(ts, TIME_FMT)
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    except Exception:
        return None


def is_non_empty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def validate_file(path: Path, final_mode: bool, min_total_events: int) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}: Failed to parse JSON ({exc})"]

    if not isinstance(data, dict):
        return [f"{path}: Root must be a JSON object"]

    required_root = ["schema_version", "record_id", "annotator", "timebase", "summary", "utterances", "qa"]
    for k in required_root:
        if k not in data:
            errors.append(f"{path}: Missing root field '{k}'")

    annotator = data.get("annotator", {})
    if not isinstance(annotator, dict):
        errors.append(f"{path}: 'annotator' must be object")
    else:
        if "annotation_status" not in annotator:
            errors.append(f"{path}: annotator.annotation_status is required")
        elif final_mode and annotator.get("annotation_status") != "complete":
            errors.append(f"{path}: annotation_status must be 'complete' in --final mode")

    if data.get("timebase") != "utterance_local":
        errors.append(f"{path}: timebase must be 'utterance_local'")

    utterances = data.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        errors.append(f"{path}: utterances must be a non-empty array")
        return errors

    total_events = 0
    seen_event_keys: set[str] = set()

    for i, utt in enumerate(utterances):
        up = f"{path}: utterances[{i}]"
        if not isinstance(utt, dict):
            errors.append(f"{up} must be object")
            continue

        for key in ["utt", "speaker_id", "duration_s", "mispronunciations", "behavioral_events"]:
            if key not in utt:
                errors.append(f"{up}: Missing '{key}'")

        dur = utt.get("duration_s")
        if not isinstance(dur, (int, float)) or float(dur) <= 0:
            errors.append(f"{up}: duration_s must be > 0")
            dur = 0.0
        else:
            dur = float(dur)

        mis = utt.get("mispronunciations", [])
        beh = utt.get("behavioral_events", [])
        if not isinstance(mis, list):
            errors.append(f"{up}: mispronunciations must be array")
            mis = []
        if not isinstance(beh, list):
            errors.append(f"{up}: behavioral_events must be array")
            beh = []

        for j, item in enumerate(mis):
            ep = f"{up}.mispronunciations[{j}]"
            if not isinstance(item, dict):
                errors.append(f"{ep} must be object")
                continue

            if not is_non_empty_str(item.get("type")):
                errors.append(f"{ep}: non-empty 'type' is required")

            has_target = is_non_empty_str(item.get("target_phone"))
            has_observed = is_non_empty_str(item.get("observed_phone"))
            if not (has_target or has_observed):
                errors.append(f"{ep}: at least one of 'target_phone' or 'observed_phone' is required")

            s = parse_ts(item.get("start"))
            e = parse_ts(item.get("end"))
            if s is None:
                errors.append(f"{ep}: invalid start time (expected HH:MM:SS:mmm)")
            if e is None:
                errors.append(f"{ep}: invalid end time (expected HH:MM:SS:mmm)")
            if s is not None and e is not None:
                if s >= e:
                    errors.append(f"{ep}: start must be < end")
                if s < 0 or e > dur:
                    errors.append(f"{ep}: event must lie within utterance duration [0,{dur:.3f}]")

            key = json.dumps(["mis", utt.get("utt", ""), item.get("type", ""), item.get("start", ""), item.get("end", "")], ensure_ascii=False)
            if key in seen_event_keys:
                errors.append(f"{ep}: duplicate event signature detected")
            else:
                seen_event_keys.add(key)

        for j, item in enumerate(beh):
            ep = f"{up}.behavioral_events[{j}]"
            if not isinstance(item, dict):
                errors.append(f"{ep} must be object")
                continue

            if not is_non_empty_str(item.get("type")):
                errors.append(f"{ep}: non-empty 'type' is required")

            s = parse_ts(item.get("start"))
            e = parse_ts(item.get("end"))
            if s is None:
                errors.append(f"{ep}: invalid start time (expected HH:MM:SS:mmm)")
            if e is None:
                errors.append(f"{ep}: invalid end time (expected HH:MM:SS:mmm)")
            if s is not None and e is not None:
                if s >= e:
                    errors.append(f"{ep}: start must be < end")
                if s < 0 or e > dur:
                    errors.append(f"{ep}: event must lie within utterance duration [0,{dur:.3f}]")

            key = json.dumps(["beh", utt.get("utt", ""), item.get("type", ""), item.get("start", ""), item.get("end", "")], ensure_ascii=False)
            if key in seen_event_keys:
                errors.append(f"{ep}: duplicate event signature detected")
            else:
                seen_event_keys.add(key)

        total_events += len(mis) + len(beh)

    qa = data.get("qa", {})
    if not isinstance(qa, dict):
        errors.append(f"{path}: qa must be object")
    elif final_mode:
        qa_keys = [
            "all_events_have_type",
            "all_events_have_valid_time_range",
            "all_events_within_utterance",
            "mispronunciations_have_target_or_observed_phone",
            "no_duplicate_events",
        ]
        for key in qa_keys:
            if qa.get(key) is not True:
                errors.append(f"{path}: qa.{key} must be true in --final mode")

    if total_events < min_total_events:
        errors.append(f"{path}: total events ({total_events}) is below required minimum ({min_total_events})")

    return errors


def gather_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(target.glob("*.gold.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict validation for human gold-label files.")
    parser.add_argument("target", type=Path, help="A .gold.json file or a directory containing them.")
    parser.add_argument("--final", action="store_true", help="Apply final-submission checks (annotation_status and QA flags).")
    parser.add_argument(
        "--min-total-events",
        type=int,
        default=0,
        help="Minimum number of total events (mispronunciations + behavioral_events) required per file.",
    )
    args = parser.parse_args()

    files = gather_files(args.target)
    if not files:
        print(f"No .gold.json files found in {args.target}")
        raise SystemExit(1)

    all_errors: list[str] = []
    for file in files:
        all_errors.extend(validate_file(file, final_mode=args.final, min_total_events=args.min_total_events))

    if all_errors:
        print("Gold-label validation FAILED:")
        for e in all_errors:
            print(f" - {e}")
        raise SystemExit(1)

    print(f"Gold-label validation PASSED for {len(files)} file(s).")


if __name__ == "__main__":
    main()
