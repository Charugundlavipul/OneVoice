from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiments.cross_task_generalization.common.io_utils import ROOT, write_json
from experiments.cross_task_generalization.common.time_utils import seconds_to_ts


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


ONEVOICE_REPAIR_PROMPT = """You repair OneVoice JSON so it passes the validator.
Return one complete corrected OneVoice JSON object only. Preserve all turns and useful annotations.
Do not add keys outside the OneVoice schema."""


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def onevoice_with_turn_subset(onevoice: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, Any]:
    subset = dict(onevoice)
    subset["speakers"] = list(onevoice.get("speakers", []))
    subset["metadata"] = dict(onevoice.get("metadata", {}))
    subset["turns"] = turns
    if turns:
        subset["session_id"] = f"{onevoice.get('session_id', '')}_chunk_{turns[0].get('turn_index')}_{turns[-1].get('turn_index')}"
        subset["session_duration"] = str(turns[-1].get("end") or onevoice.get("session_duration") or "")
    return subset


def merge_onevoice_chunks(base: dict[str, Any], chunk_docs: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(base)
    turn_by_index: dict[int, dict[str, Any]] = {
        int(turn["turn_index"]): dict(turn)
        for turn in base.get("turns", [])
        if isinstance(turn, dict) and str(turn.get("turn_index", "")).isdigit()
    }
    speaker_by_id: dict[str, dict[str, Any]] = {
        str(speaker.get("speaker_id")): dict(speaker)
        for speaker in base.get("speakers", [])
        if isinstance(speaker, dict) and speaker.get("speaker_id")
    }
    for doc in chunk_docs:
        for speaker in doc.get("speakers", []) or []:
            if isinstance(speaker, dict) and speaker.get("speaker_id"):
                speaker_by_id[str(speaker["speaker_id"])] = dict(speaker)
        for turn in doc.get("turns", []) or []:
            if not isinstance(turn, dict):
                continue
            idx = turn.get("turn_index")
            try:
                idx_int = int(idx)
            except Exception:
                continue
            turn_by_index[idx_int] = dict(turn)
    merged["speakers"] = [speaker_by_id[k] for k in sorted(speaker_by_id)]
    merged["turns"] = [turn_by_index[k] for k in sorted(turn_by_index)]
    return merged


def validate_onevoice(candidate: dict[str, Any], work_dir: Path, validator_path: Path = Path("validate_onevoice.py")) -> tuple[bool, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = work_dir / "candidate.json"
    write_json(candidate_path, candidate)
    validator = validator_path if validator_path.is_absolute() else ROOT / validator_path
    cp = subprocess.run(
        [sys.executable, str(validator), str(candidate_path), "--mode", "full"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return cp.returncode == 0, (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")


def repair_with_validator(
    candidate: dict[str, Any],
    record_dir: Path,
    agent_name: str,
    call_json: Callable[[str, dict[str, Any]], tuple[dict[str, Any], str]],
    repair_prompt: str,
    max_rounds: int,
) -> tuple[dict[str, Any], bool, int]:
    current = candidate
    for round_idx in range(max_rounds + 1):
        ok, log = validate_onevoice(current, record_dir / f"{agent_name}_validation_round_{round_idx}")
        (record_dir / f"{agent_name}_validator_round_{round_idx}.log").write_text(log, encoding="utf-8")
        if ok:
            return current, True, round_idx
        if round_idx >= max_rounds:
            return current, False, round_idx
        current, raw = call_json(
            repair_prompt,
            {
                "task": "Repair this OneVoice JSON so the validator passes. Return corrected full JSON only.",
                "validator_errors": log[-6000:],
                "current_json": current,
            },
        )
        (record_dir / f"{agent_name}_repair_round_{round_idx + 1}.txt").write_text(raw, encoding="utf-8")
    return current, False, max_rounds


def task2_empty_final(window_id: str, utterances: list[dict[str, Any]]) -> dict[str, Any]:
    speakers = sorted({str(u.get("speaker_id", "UNK")) for u in utterances})
    assignments = [{"utt_index": int(u.get("utt_index", i)), "pred_speaker_id": str(u.get("speaker_id", "UNK"))} for i, u in enumerate(utterances)]
    return {
        "window_id": window_id,
        "speakers": [{"pred_speaker_id": sid, "role": "child" if sid == "CHI" else "adult"} for sid in speakers],
        "utterance_speaker_assignments": assignments,
        "utterance_event_counts": [
            {
                "utt_index": a["utt_index"],
                "pred_speaker_id": a["pred_speaker_id"],
                "repetition_count": 0,
                "repair_count": 0,
                "pause_count": 0,
                "non_speech_count": 0,
                "overlap_count": 0,
                "total_event_count": 0,
            }
            for a in assignments
        ],
        "events": [],
    }


def task3_empty_final(bundle_id: str, utterances: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bundle_id": bundle_id,
        "utterance_summaries": [
            {
                "utt_id": u["utt_id"],
                "long_word_count": 0,
                "short_word_count": 0,
                "inter_word_gap_count": 0,
                "boundary_gap_count": 0,
                "long_phone_count": 0,
                "short_phone_count": 0,
                "closure_count": 0,
                "pause_silence_count": 0,
                "glottal_stop_count": 0,
                "total_event_count": 0,
            }
            for u in utterances
        ],
        "events": [],
        "unlinked_events": [],
    }


def onevoice_from_childes(row: dict[str, Any]) -> dict[str, Any]:
    speakers = []
    for speaker in row.get("speakers", []):
        sid = str(speaker.get("speaker_id", ""))
        if not sid:
            continue
        speakers.append(
            {
                "speaker_id": sid,
                "gender": "",
                "language": "en",
                "native_language": "english",
                "accent": "",
                "age": "",
                "name": speaker.get("description", ""),
                "age_group": "child" if sid == "CHI" else "adult",
            }
        )
    turns = []
    for utt in row.get("utterances", []):
        turns.append(
            {
                "turn_index": int(utt["utt_index"]),
                "utt_id": f"{row['window_id']}_{utt['utt_index']}",
                "speaker_id": utt["speaker_id"],
                "reference_transcript": "",
                "phoneme_reference": "",
                "orthographic_transcript": utt.get("text", ""),
                "phonetic_transcript": "",
                "ipa_transcript": "",
                "raw_transcript": utt.get("raw", ""),
                "start": "",
                "end": "",
                "word_alignments": [],
                "mispronunciations": [],
                "behavioral_events": [],
            }
        )
    return {
        "session_id": row["window_id"],
        "session_date": "",
        "sample_rate": 0,
        "audio_file_path": "",
        "audio_format": "none",
        "session_duration": "",
        "speakers": speakers,
        "turns": turns,
        "metadata": {
            "dataset_name": "CHILDES",
            "dataset_split": "selected_windows",
            "text_annotation_source": "human CHAT",
            "text_annotation_details": "Main speaker tiers from CHAT window.",
            "text_coverage": "full",
            "phoneme_annotation_source": "none",
            "phoneme_annotation_details": "",
            "phoneme_coverage": "none",
        },
    }


def onevoice_from_timit(row: dict[str, Any]) -> dict[str, Any]:
    offset = 0.0
    turns = []
    for idx, utt in enumerate(row.get("utterances", [])):
        turn_start = offset
        turn_end = offset + float(utt.get("duration", 0.0))
        words = []
        for word in utt.get("word_intervals", []):
            phones = []
            for phone in utt.get("phone_intervals", []):
                if phone.get("word_index") != word.get("word_index"):
                    continue
                phones.append(
                    {
                        "phone": phone["phone"],
                        "start": seconds_to_ts(turn_start + float(phone["start"])),
                        "end": seconds_to_ts(turn_start + float(phone["end"])),
                        "phone_index": int(phone["phone_index"]),
                    }
                )
            words.append(
                {
                    "word_index": int(word["word_index"]),
                    "word": word["word"],
                    "start": seconds_to_ts(turn_start + float(word["start"])),
                    "end": seconds_to_ts(turn_start + float(word["end"])),
                    "phonemes": phones,
                }
            )
        turns.append(
            {
                "turn_index": idx,
                "utt_id": utt["utt_id"],
                "speaker_id": row["speaker_id"],
                "reference_transcript": utt.get("transcript", ""),
                "phoneme_reference": "",
                "orthographic_transcript": utt.get("transcript", ""),
                "phonetic_transcript": " ".join(p["phone"] for p in utt.get("phone_intervals", [])),
                "ipa_transcript": "",
                "raw_transcript": utt.get("transcript", ""),
                "start": seconds_to_ts(turn_start),
                "end": seconds_to_ts(turn_end),
                "word_alignments": words,
                "mispronunciations": [],
                "behavioral_events": [],
            }
        )
        offset = turn_end
    return {
        "session_id": row["bundle_id"],
        "session_date": "",
        "sample_rate": 16000,
        "audio_file_path": "",
        "audio_format": "wav",
        "session_duration": seconds_to_ts(offset),
        "speakers": [
            {
                "speaker_id": row["speaker_id"],
                "gender": "other",
                "language": "en",
                "native_language": "english",
                "accent": "",
                "age": "",
                "name": "",
                "age_group": "adult",
            }
        ],
        "turns": turns,
        "metadata": {
            "dataset_name": "TIMIT",
            "dataset_split": row.get("eval_split", "TEST"),
            "text_annotation_source": "TIMIT .TXT/.WRD",
            "text_annotation_details": "Human-aligned word intervals.",
            "text_coverage": "full",
            "phoneme_annotation_source": "TIMIT .PHN",
            "phoneme_annotation_details": "Human-aligned phone intervals.",
            "phoneme_coverage": "full",
        },
    }
