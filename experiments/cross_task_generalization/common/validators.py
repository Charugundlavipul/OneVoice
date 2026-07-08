from __future__ import annotations

from collections import Counter
from typing import Any


TASK2_EVENT_TYPES = {"repetition", "repair", "pause", "non_speech", "overlap"}
TASK3_EVENT_TYPES = {
    "long_word",
    "short_word",
    "inter_word_gap",
    "boundary_gap",
    "long_phone",
    "short_phone",
    "closure_segment",
    "pause_silence_segment",
    "glottal_stop",
}


def _err(errors: list[str], path: str, msg: str) -> None:
    errors.append(f"{path}: {msg}")


def validate_task2_final(obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj.get("window_id"), str) or not obj.get("window_id"):
        _err(errors, "window_id", "required non-empty string")
    if not isinstance(obj.get("speakers"), list):
        _err(errors, "speakers", "required list")
    if not isinstance(obj.get("utterance_speaker_assignments"), list):
        _err(errors, "utterance_speaker_assignments", "required list")
    if not isinstance(obj.get("utterance_event_counts"), list):
        _err(errors, "utterance_event_counts", "required list")
    if not isinstance(obj.get("events"), list):
        _err(errors, "events", "required list")
        return errors

    utt_indices = set()
    for i, row in enumerate(obj.get("utterance_speaker_assignments", []) or []):
        if not isinstance(row, dict):
            _err(errors, f"utterance_speaker_assignments[{i}]", "must be object")
            continue
        if not isinstance(row.get("utt_index"), int):
            _err(errors, f"utterance_speaker_assignments[{i}].utt_index", "must be int")
        else:
            utt_indices.add(row["utt_index"])
        if not isinstance(row.get("pred_speaker_id"), str) or not row.get("pred_speaker_id"):
            _err(errors, f"utterance_speaker_assignments[{i}].pred_speaker_id", "required non-empty string")

    for i, event in enumerate(obj.get("events", []) or []):
        if not isinstance(event, dict):
            _err(errors, f"events[{i}]", "must be object")
            continue
        if event.get("event_type") not in TASK2_EVENT_TYPES:
            _err(errors, f"events[{i}].event_type", f"must be one of {sorted(TASK2_EVENT_TYPES)}")
        if not isinstance(event.get("utt_index"), int):
            _err(errors, f"events[{i}].utt_index", "must be int")
        elif utt_indices and event["utt_index"] not in utt_indices:
            _err(errors, f"events[{i}].utt_index", "not present in utterance assignments")
        if not isinstance(event.get("pred_speaker_id"), str) or not event.get("pred_speaker_id"):
            _err(errors, f"events[{i}].pred_speaker_id", "required non-empty string")
    return errors


def validate_task3_final(obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj.get("bundle_id"), str) or not obj.get("bundle_id"):
        _err(errors, "bundle_id", "required non-empty string")
    if not isinstance(obj.get("utterance_summaries"), list):
        _err(errors, "utterance_summaries", "required list")
    if not isinstance(obj.get("events"), list):
        _err(errors, "events", "required list")
        return errors
    if not isinstance(obj.get("unlinked_events"), list):
        _err(errors, "unlinked_events", "required list")

    utt_ids = {row.get("utt_id") for row in obj.get("utterance_summaries", []) or [] if isinstance(row, dict)}
    for i, event in enumerate(obj.get("events", []) or []):
        if not isinstance(event, dict):
            _err(errors, f"events[{i}]", "must be object")
            continue
        if event.get("event_type") not in TASK3_EVENT_TYPES:
            _err(errors, f"events[{i}].event_type", f"must be one of {sorted(TASK3_EVENT_TYPES)}")
        if not isinstance(event.get("utt_id"), str) or not event.get("utt_id"):
            _err(errors, f"events[{i}].utt_id", "required non-empty string")
        elif utt_ids and event["utt_id"] not in utt_ids:
            _err(errors, f"events[{i}].utt_id", "not present in utterance summaries")
        for key in ("start", "end"):
            if key in event and not isinstance(event[key], (int, float)):
                _err(errors, f"events[{i}].{key}", "must be numeric seconds")
    return errors


def task2_event_count_rows(events: list[dict[str, Any]], assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    speakers = {int(a["utt_index"]): a.get("pred_speaker_id", "") for a in assignments if isinstance(a, dict) and isinstance(a.get("utt_index"), int)}
    by_utt: Counter[tuple[int, str]] = Counter()
    for event in events:
        if not isinstance(event, dict):
            continue
        idx = event.get("utt_index")
        typ = event.get("event_type")
        if isinstance(idx, int) and isinstance(typ, str):
            by_utt[(idx, typ)] += 1
    rows: list[dict[str, Any]] = []
    for idx in sorted(speakers):
        row = {
            "utt_index": idx,
            "pred_speaker_id": speakers[idx],
            "repetition_count": by_utt[(idx, "repetition")],
            "repair_count": by_utt[(idx, "repair")],
            "pause_count": by_utt[(idx, "pause")],
            "non_speech_count": by_utt[(idx, "non_speech")],
            "overlap_count": by_utt[(idx, "overlap")],
        }
        row["total_event_count"] = sum(v for k, v in row.items() if k.endswith("_count") and k != "total_event_count")
        rows.append(row)
    return rows
