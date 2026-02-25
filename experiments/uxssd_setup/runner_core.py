#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


TIME_FMT = "%H:%M:%S:%f"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-5-mini"


SYSTEM_EXP1_MIS = """You are Agent A for Experiment 1 (OneVoice-agnostic baseline).
Your job: infer MISPRONUNCIATIONS from raw inputs.

Return JSON only. No markdown. No prose.
Output schema:
{
  "record_id": "...",
  "utterances": [
    {
      "utt": "...",
      "mispronunciations": [
        {
          "type": "non-empty string",
          "target_phone": "string or empty",
          "observed_phone": "string or empty",
          "word_index": "int/string or empty",
          "phone_index": "int/string or empty",
          "start": "HH:MM:SS:mmm",
          "end": "HH:MM:SS:mmm",
          "notes": "optional"
        }
      ]
    }
  ]
}

Rules:
- Use utterance-local timestamps.
- If no event exists for an utterance, return an empty list for that utterance.
- Include every utterance from input exactly once.
- Do not output behavioral events in this step.
- Definition of mispronunciation:
  - A speech-production error where the spoken realization differs from the intended/canonical target
    at phone level (for example substitution, deletion, insertion, distortion).
  - This is about pronunciation quality, not interaction behavior.
- Mispronunciation examples:
  - {"type":"substitution","target_phone":"s","observed_phone":"th","start":"00:00:01:120","end":"00:00:01:260","notes":"sun -> thun"}
  - {"type":"deletion","target_phone":"t","observed_phone":"","start":"00:00:02:000","end":"00:00:02:140","notes":"final /t/ omitted"}
- Exclusions:
  - Do NOT mark speaker changes, interruptions, or SLT interventions as mispronunciations.
"""


SYSTEM_EXP1_BEH = """You are Agent B for Experiment 1 (OneVoice-agnostic baseline).
Your job: infer BEHAVIORAL EVENTS from raw inputs.

Return JSON only. No markdown. No prose.
Output schema:
{
  "record_id": "...",
  "utterances": [
    {
      "utt": "...",
      "behavioral_events": [
        {
          "type": "non-empty string",
          "start": "HH:MM:SS:mmm",
          "end": "HH:MM:SS:mmm",
          "notes": "optional"
        }
      ]
    }
  ]
}

Rules:
- Use utterance-local timestamps.
- If no event exists for an utterance, return an empty list for that utterance.
- Include every utterance from input exactly once.
- Do not output mispronunciations in this step.
- Allowed behavioral event types only:
  - "speaker_transition"
  - "slt_intervention"
  - "other_speaker_activity"
- If you detect pause-like boundary cues, encode them as "speaker_transition" (not "silent_pause").
- Definition of behavioral event:
  - A non-phonetic interaction event that describes communication behavior, turn-taking,
    or conversational context rather than phone-level articulation.
- Behavioral event examples:
  - {"type":"slt_intervention","start":"00:00:03:050","end":"00:00:03:900","notes":"therapist prompts child"}
  - {"type":"speaker_transition","start":"00:00:05:200","end":"00:00:05:350","notes":"child turn changes to adult"}
  - {"type":"other_speaker_activity","start":"00:00:06:000","end":"00:00:06:600","notes":"non-child speaker overlap/background talk"}
- Exclusions:
  - Do NOT encode phone substitutions/deletions/distortions as behavioral events.
  - Do NOT emit generic speech activity labels (for example child_speech/child_speaking).
  - Do NOT emit transitions for silence<->same-speaker micro-boundaries.
  - Keep events sparse; at most two speaker_transition and one slt_intervention per utterance.
"""


SYSTEM_EXP1_MERGE = """You are Agent C for Experiment 1 (OneVoice-agnostic baseline).
You merge Agent A and Agent B outputs into one final JSON.

Return JSON only. No markdown. No prose.
Output schema:
{
  "record_id": "...",
  "utterances": [
    {
      "utt": "...",
      "mispronunciations": [ ... ],
      "behavioral_events": [ ... ]
    }
  ]
}

Rules:
- Keep every utterance exactly once.
- Preserve events from both agents, remove exact duplicates.
- Keep timestamps unchanged unless obviously invalid; if invalid, drop that event.
"""


SYSTEM_EXP2_MIS = """You are Agent A for Experiment 2 (OneVoice pipeline).
Input is OneVoice JSON. Your task is to fill turns[].mispronunciations only.

Return one valid OneVoice JSON object only.
Hard constraints:
- Preserve existing structure and fields.
- Do not delete existing turns/speakers/metadata.
- Do not modify behavioral_events in this step.
- For every mispronunciation item, 'type' is mandatory and at least one of
  'target_phone' or 'observed_phone' must be non-empty.
- Keep times in HH:MM:SS:mmm and within turn bounds.
- Definition of mispronunciation:
  - A phone-level pronunciation mismatch (substitution/deletion/insertion/distortion)
    for the intended spoken content.
- Examples:
  - {"type":"substitution","target_phone":"k","observed_phone":"t","start":"00:00:00:940","end":"00:00:01:090"}
  - {"type":"distortion","target_phone":"r","observed_phone":"r_distorted","start":"00:00:02:100","end":"00:00:02:280"}
- Do not use mispronunciations for turn-taking or speaker-behavior phenomena.
"""


SYSTEM_EXP2_BEH = """You are Agent B for Experiment 2 (OneVoice pipeline).
Input is OneVoice JSON. Your task is to fill turns[].behavioral_events only.

Return one valid OneVoice JSON object only.
Hard constraints:
- Preserve existing structure and fields.
- Do not delete existing turns/speakers/metadata.
- Do not modify mispronunciations produced by Agent A except obvious format fixes.
- For every behavioral event item, 'type' is mandatory.
- Keep times in HH:MM:SS:mmm and within turn bounds.
- Allowed behavioral event types only:
  - "speaker_transition"
  - "slt_intervention"
  - "other_speaker_activity"
- Definition of behavioral event:
  - A non-phonetic behavior or interaction cue (for example speaker transitions,
    SLT interventions, overlaps, prolonged non-child activity).
- Examples:
  - {"type":"slt_intervention","start":"00:00:01:300","end":"00:00:02:050","notes":"prompt/recast"}
  - {"type":"speaker_transition","start":"00:00:03:400","end":"00:00:03:520"}
  - {"type":"other_speaker_activity","start":"00:00:04:000","end":"00:00:04:700","notes":"adult overlap"}
- Do not encode phone-level pronunciation errors as behavioral events.
- Do not emit routine one-per-turn transitions by default.
- If a behavioral_event_target_hint is provided, keep total behavioral events close to it.
"""


SYSTEM_EXP2_MERGE = """You are Agent C for Experiment 2 (OneVoice pipeline).
You receive OneVoice JSON after Agent A and Agent B.
Your task is final cleanup and consistency fixup while preserving meaning.

Return one valid OneVoice JSON object only.
Hard constraints:
- Keep all existing turns and speaker identity.
- Keep both mispronunciations and behavioral_events.
- Ensure required fields are present and non-empty where mandatory.
- Keep times in HH:MM:SS:mmm and avoid impossible ranges.
"""

SYSTEM_JSON_REPAIR = """You repair malformed JSON.

Task:
- Convert the provided text into one valid JSON object.
- Preserve intended keys/values as much as possible.
- Remove markdown, comments, trailing commas, and invalid fragments.
- Do not add prose.

Output:
- JSON object only.
"""


@dataclass
class RunnerConfig:
    mode: str  # exp1 | exp2
    manifest: Path
    output_root: Path
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_records: int | None = None
    record_ids: set[str] | None = None
    max_file_chars: int = 3200
    max_files_per_utt: int = 8
    max_utterances_per_record: int | None = None
    max_repair_rounds: int = 3
    validator_path: Path = Path("validate_onevoice.py")
    dry_run: bool = False
    max_output_tokens: int = 12000


def model_supports_temperature(model: str) -> bool:
    m = str(model or "").strip().lower()
    # GPT-5 models currently reject temperature in this pipeline.
    return not m.startswith("gpt-5")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_for_dir() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def ts_from_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}:{ms:03d}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short_read(path: Path, max_chars: int) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[UNREADABLE: {exc}]"
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + "\n...[TRUNCATED]..."


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_rel(p: str) -> Path:
    return ROOT / Path(p)


def safe_json_from_text(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        # If first line is "json", drop it
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    # Direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Best-effort extraction of outermost object
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        candidate = raw[first : last + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Model output is not a valid JSON object")


def response_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", "") != "message":
            continue
        for c in getattr(item, "content", []) or []:
            ctype = getattr(c, "type", "")
            if ctype in {"output_text", "text"}:
                value = getattr(c, "text", "")
                if isinstance(value, str):
                    parts.append(value)
    joined = "\n".join(parts).strip()
    if joined:
        return joined

    # fallback
    try:
        return json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)
    except Exception:
        return str(resp)


def call_openai_json(
    client: Any,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float,
    max_output_tokens: int,
    json_repair_attempts: int = 2,
) -> tuple[dict[str, Any], str]:
    def repair_json_text(broken_text: str) -> str:
        req: dict[str, Any] = {
            "model": model,
            "max_output_tokens": max_output_tokens,
            "input": [
                {"role": "system", "content": SYSTEM_JSON_REPAIR},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": "Return corrected JSON object only.",
                            "malformed_json_text": broken_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        if model_supports_temperature(model):
            req["temperature"] = 0
        resp = client.responses.create(**req)
        return response_text(resp)

    payload_text = json.dumps(user_payload, ensure_ascii=False)
    req: dict[str, Any] = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload_text},
        ],
    }
    if model_supports_temperature(model):
        req["temperature"] = temperature
    resp = client.responses.create(**req)
    txt = response_text(resp)
    try:
        return safe_json_from_text(txt), txt
    except Exception as first_exc:
        last_exc: Exception = first_exc
        repaired = txt
        for _ in range(json_repair_attempts):
            repaired = repair_json_text(repaired)
            try:
                parsed = safe_json_from_text(repaired)
                return parsed, txt + "\n\n[JSON_REPAIRED]\n" + repaired
            except Exception as repair_exc:
                last_exc = repair_exc

        snippet = txt[:400].replace("\n", " ")
        raise ValueError(f"Failed to parse model JSON after {json_repair_attempts} repair attempts: {last_exc}. Raw snippet: {snippet}")


def _build_exp1_empty(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record.get("record_id", ""),
        "utterances": [
            {"utt": u.get("utt", ""), "mispronunciations": [], "behavioral_events": []}
            for u in record.get("utterances", [])
        ],
    }


def _norm_event_list(v: Any) -> list[dict[str, Any]]:
    if not isinstance(v, list):
        return []
    out: list[dict[str, Any]] = []
    for item in v:
        if isinstance(item, dict):
            out.append(item)
    return out


EXP1_BEH_TYPE_ALIASES: dict[str, str] = {
    "speaker_transition": "speaker_transition",
    "speaker_change": "speaker_transition",
    "turn_transition": "speaker_transition",
    "turn_change": "speaker_transition",
    "slt_intervention": "slt_intervention",
    "therapist_intervention": "slt_intervention",
    "clinician_intervention": "slt_intervention",
    "other_speaker_activity": "other_speaker_activity",
    "adult_overlap": "other_speaker_activity",
}
EXP1_BEH_ALLOWED_TYPES = {"speaker_transition", "slt_intervention", "other_speaker_activity"}
EXP1_BEH_MAX_PER_TYPE = {"speaker_transition": 2, "slt_intervention": 1, "other_speaker_activity": 1}
EXP1_BEH_MAX_TOTAL_PER_UTT = 3
EXP2_BEH_TYPE_ALIASES: dict[str, str] = {
    "speaker_transition": "speaker_transition",
    "speaker_change": "speaker_transition",
    "turn_transition": "speaker_transition",
    "turn_change": "speaker_transition",
    "slt_intervention": "slt_intervention",
    "therapist_intervention": "slt_intervention",
    "clinician_intervention": "slt_intervention",
    "other_speaker_activity": "other_speaker_activity",
    "adult_overlap": "other_speaker_activity",
    "silent_pause": "speaker_transition",
    "pause": "speaker_transition",
    "non_speech_vocalization": "other_speaker_activity",
}
EXP2_BEH_ALLOWED_TYPES = {"speaker_transition", "slt_intervention", "other_speaker_activity"}
EXP2_BEH_MAX_PER_TYPE_PER_TURN = {"speaker_transition": 1, "slt_intervention": 1, "other_speaker_activity": 1}
EXP2_BEH_MAX_TOTAL_PER_TURN = 2
EXP2_BEH_TARGET_SLACK = 1


def _canonicalize_exp1_behavioral_type(raw_type: Any) -> str:
    if not isinstance(raw_type, str):
        return ""
    key = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return ""
    return EXP1_BEH_TYPE_ALIASES.get(key, key)


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_int_or_none(v: Any) -> int | None:
    try:
        out = int(v)
    except Exception:
        return None
    return out if out >= 0 else None


def sanitize_exp1_behavioral_events(events: list[dict[str, Any]], utt_duration_s: float) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen_bins: set[tuple[str, int, int]] = set()
    duration_limit = max(0.0, _safe_float(utt_duration_s))

    for event in _norm_event_list(events):
        etype = _canonicalize_exp1_behavioral_type(event.get("type", ""))
        if etype not in EXP1_BEH_ALLOWED_TYPES:
            continue

        start_s = parse_ts(event.get("start"))
        end_s = parse_ts(event.get("end"))
        if start_s is None or end_s is None or end_s <= start_s:
            continue
        if start_s < 0:
            continue
        if duration_limit > 0 and start_s > duration_limit + 0.25:
            continue

        note = str(event.get("notes", "")).strip()
        note_lc = note.lower()
        if etype == "speaker_transition" and "silence" in note_lc:
            continue

        start_s = max(0.0, start_s)
        end_s = end_s if duration_limit <= 0 else min(end_s, duration_limit)
        if end_s <= start_s:
            continue

        binned = (etype, int(round(start_s * 20)), int(round(end_s * 20)))
        if binned in seen_bins:
            continue
        seen_bins.add(binned)

        normalized: dict[str, Any] = {
            "type": etype,
            "start": ts_from_seconds(start_s),
            "end": ts_from_seconds(end_s),
        }
        if note:
            normalized["notes"] = note
        cleaned.append(normalized)

    def score_for_type(e: dict[str, Any], etype: str) -> tuple[float, float]:
        s = parse_ts(e.get("start")) or 0.0
        en = parse_ts(e.get("end")) or s
        dur = max(0.0, en - s)
        if etype == "speaker_transition":
            edge_dist = min(s, max(0.0, duration_limit - en)) if duration_limit > 0 else s
            return (edge_dist, dur)
        return (-dur, s)

    selected: list[dict[str, Any]] = []
    for etype, cap in EXP1_BEH_MAX_PER_TYPE.items():
        bucket = [e for e in cleaned if e.get("type") == etype]
        bucket.sort(key=lambda e: score_for_type(e, etype))
        selected.extend(bucket[:cap])

    selected.sort(
        key=lambda e: (
            parse_ts(e.get("start")) or 0.0,
            parse_ts(e.get("end")) or 0.0,
            str(e.get("type", "")),
        )
    )
    return selected[:EXP1_BEH_MAX_TOTAL_PER_UTT]


def _canonicalize_exp2_behavioral_type(raw_type: Any) -> str:
    if not isinstance(raw_type, str):
        return ""
    key = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return ""
    return EXP2_BEH_TYPE_ALIASES.get(key, key)


def _event_score_for_exp2_cap(event: dict[str, Any], turn_start: float, turn_end: float) -> tuple[float, float, float]:
    etype = str(event.get("type", ""))
    s = parse_ts(event.get("start")) or turn_start
    e = parse_ts(event.get("end")) or s
    dur = max(0.0, e - s)
    if etype == "slt_intervention":
        return (3.0 + dur, dur, -s)
    if etype == "other_speaker_activity":
        return (2.0 + dur * 0.5, dur, -s)
    # speaker_transition: prefer non-edge transitions, then longer spans
    edge_dist = min(max(0.0, s - turn_start), max(0.0, turn_end - e))
    return (1.0 + edge_dist * 10.0 + dur * 0.2, dur, -s)


def _turn_transition_candidates(turn: dict[str, Any], turn_start: float, turn_end: float) -> list[dict[str, Any]]:
    words = turn.get("word_alignments", [])
    if not isinstance(words, list):
        return []
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(words) - 1):
        cur = words[i] if isinstance(words[i], dict) else {}
        nxt = words[i + 1] if isinstance(words[i + 1], dict) else {}
        cur_word = str(cur.get("word", "")).strip()
        nxt_word = str(nxt.get("word", "")).strip()
        if bool(cur_word) == bool(nxt_word):
            continue
        boundary = parse_ts(cur.get("end"))
        if boundary is None:
            boundary = parse_ts(nxt.get("start"))
        if boundary is None:
            continue
        s = max(turn_start, boundary - 0.01)
        e = min(turn_end, boundary + 0.01)
        if e <= s:
            continue
        key = (int(round(s * 50)), int(round(e * 50)))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"type": "speaker_transition", "start": ts_from_seconds(s), "end": ts_from_seconds(e)})
    return candidates


def sanitize_exp2_behavioral_bundle(bundle: dict[str, Any], expected_behavioral_target: int | None) -> dict[str, Any]:
    turns_in = bundle.get("turns")
    if not isinstance(turns_in, list):
        return bundle

    cleaned_turns: list[dict[str, Any]] = []
    flat_events: list[tuple[int, dict[str, Any], float, float]] = []
    turn_bounds: list[tuple[float, float]] = []
    for idx, turn in enumerate(turns_in):
        if not isinstance(turn, dict):
            cleaned_turns.append(turn)
            turn_bounds.append((0.0, 0.001))
            continue

        t_start = parse_ts(turn.get("start")) or 0.0
        t_end = parse_ts(turn.get("end")) or t_start
        if t_end <= t_start:
            t_end = t_start + 0.001
        turn_bounds.append((t_start, t_end))

        per_type: dict[str, list[dict[str, Any]]] = {k: [] for k in EXP2_BEH_MAX_PER_TYPE_PER_TURN}
        seen: set[tuple[str, int, int]] = set()
        for raw in _norm_event_list(turn.get("behavioral_events", [])):
            etype = _canonicalize_exp2_behavioral_type(raw.get("type", ""))
            if etype not in EXP2_BEH_ALLOWED_TYPES:
                continue
            s = parse_ts(raw.get("start"))
            e = parse_ts(raw.get("end"))
            if s is None or e is None:
                continue
            s = max(t_start, s)
            e = min(t_end, e)
            if e <= s:
                continue
            key = (etype, int(round(s * 20)), int(round(e * 20)))
            if key in seen:
                continue
            seen.add(key)
            per_type.setdefault(etype, []).append(
                {
                    "type": etype,
                    "start": ts_from_seconds(s),
                    "end": ts_from_seconds(e),
                }
            )

        keep: list[dict[str, Any]] = []
        for etype, cap in EXP2_BEH_MAX_PER_TYPE_PER_TURN.items():
            bucket = per_type.get(etype, [])
            bucket.sort(
                key=lambda ev: _event_score_for_exp2_cap(ev, t_start, t_end),
                reverse=True,
            )
            keep.extend(bucket[:cap])
        keep.sort(
            key=lambda ev: (
                parse_ts(ev.get("start")) or t_start,
                parse_ts(ev.get("end")) or t_start,
                str(ev.get("type", "")),
            )
        )
        keep = keep[:EXP2_BEH_MAX_TOTAL_PER_TURN]

        updated_turn = dict(turn)
        updated_turn["behavioral_events"] = keep
        cleaned_turns.append(updated_turn)
        for ev in keep:
            flat_events.append((idx, ev, t_start, t_end))

    cap_target = _safe_int_or_none(expected_behavioral_target)
    if cap_target is not None:
        cap = max(1, cap_target + EXP2_BEH_TARGET_SLACK)
        if len(flat_events) > cap:
            ranked = sorted(
                flat_events,
                key=lambda item: _event_score_for_exp2_cap(item[1], item[2], item[3]),
                reverse=True,
            )
            keep_keys = {
                (
                    idx,
                    ev.get("type", ""),
                    ev.get("start", ""),
                    ev.get("end", ""),
                )
                for idx, ev, _, _ in ranked[:cap]
            }
            for idx, turn in enumerate(cleaned_turns):
                if not isinstance(turn, dict):
                    continue
                kept = []
                for ev in _norm_event_list(turn.get("behavioral_events", [])):
                    k = (idx, ev.get("type", ""), ev.get("start", ""), ev.get("end", ""))
                    if k in keep_keys:
                        kept.append(ev)
                turn["behavioral_events"] = kept

        floor_target = max(0, cap_target)
        current_total = sum(
            len(_norm_event_list(t.get("behavioral_events", [])))
            for t in cleaned_turns
            if isinstance(t, dict)
        )
        if current_total < floor_target:
            existing_keys: set[tuple[int, str, str, str]] = set()
            for idx, turn in enumerate(cleaned_turns):
                if not isinstance(turn, dict):
                    continue
                for ev in _norm_event_list(turn.get("behavioral_events", [])):
                    existing_keys.add((idx, str(ev.get("type", "")), str(ev.get("start", "")), str(ev.get("end", ""))))

            candidate_pool: list[tuple[int, dict[str, Any], float, float]] = []
            for idx, turn in enumerate(cleaned_turns):
                if not isinstance(turn, dict):
                    continue
                t_start, t_end = turn_bounds[idx]
                for cand in _turn_transition_candidates(turn, t_start, t_end):
                    key = (idx, cand["type"], cand["start"], cand["end"])
                    if key in existing_keys:
                        continue
                    candidate_pool.append((idx, cand, t_start, t_end))

            # Prefer adding to turns with fewer events first, then by candidate quality.
            turn_sizes = {
                idx: len(_norm_event_list(turn.get("behavioral_events", [])))
                for idx, turn in enumerate(cleaned_turns)
                if isinstance(turn, dict)
            }
            candidate_pool.sort(
                key=lambda item: (
                    turn_sizes.get(item[0], 0),
                    -_event_score_for_exp2_cap(item[1], item[2], item[3])[0],
                )
            )
            for idx, cand, _, _ in candidate_pool:
                if current_total >= floor_target:
                    break
                turn = cleaned_turns[idx]
                if not isinstance(turn, dict):
                    continue
                events = _norm_event_list(turn.get("behavioral_events", []))
                if len(events) >= EXP2_BEH_MAX_TOTAL_PER_TURN:
                    continue
                key = (idx, cand["type"], cand["start"], cand["end"])
                if key in existing_keys:
                    continue
                events.append(cand)
                events.sort(key=lambda ev: (parse_ts(ev.get("start")) or 0.0, parse_ts(ev.get("end")) or 0.0))
                turn["behavioral_events"] = events
                existing_keys.add(key)
                current_total += 1

    out = dict(bundle)
    out["turns"] = cleaned_turns
    return out


def normalize_exp1_result(
    result: dict[str, Any],
    record: dict[str, Any],
    include_mis: bool,
    include_beh: bool,
) -> dict[str, Any]:
    lookup: dict[str, dict[str, Any]] = {}
    for u in result.get("utterances", []):
        if isinstance(u, dict) and isinstance(u.get("utt"), str):
            lookup[u["utt"]] = u

    out_utts: list[dict[str, Any]] = []
    for src in record.get("utterances", []):
        utt = str(src.get("utt", ""))
        utt_duration_s = _safe_float(src.get("duration_s", 0.0))
        candidate = lookup.get(utt, {})
        row: dict[str, Any] = {"utt": utt}
        if include_mis:
            row["mispronunciations"] = _norm_event_list(candidate.get("mispronunciations", []))
        if include_beh:
            row["behavioral_events"] = sanitize_exp1_behavioral_events(
                _norm_event_list(candidate.get("behavioral_events", [])),
                utt_duration_s,
            )
        out_utts.append(row)

    return {"record_id": record.get("record_id", ""), "utterances": out_utts}


def dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for e in events:
        key = json.dumps(e, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def build_exp1_payload(record: dict[str, Any], cfg: RunnerConfig) -> dict[str, Any]:
    utterances_in = record.get("utterances", [])
    if cfg.max_utterances_per_record is not None:
        utterances_in = utterances_in[: cfg.max_utterances_per_record]

    text_file_keys = [
        "transcript_txt",
        "word_labels_textgrid",
        "phone_labels_textgrid",
        "speaker_labels_textgrid",
        "reference_word_textgrid",
        "reference_phone_textgrid",
        "reference_speaker_textgrid",
        "slt_labels_textgrid",
        "param_file",
    ]

    utts: list[dict[str, Any]] = []
    for u in utterances_in:
        files = u.get("files", {})
        text_files: list[dict[str, str]] = []
        for key in text_file_keys:
            if len(text_files) >= cfg.max_files_per_utt:
                break
            rel = str(files.get(key, "")).strip()
            if not rel:
                continue
            fp = resolve_rel(rel)
            if not fp.exists() or not fp.is_file():
                continue
            text_files.append({"file_key": key, "path": rel, "content": short_read(fp, cfg.max_file_chars)})

        utts.append(
            {
                "utt": u.get("utt", ""),
                "duration_s": u.get("duration_s", 0),
                "audio_wav": files.get("audio_wav", ""),
                "text_files": text_files,
            }
        )

    return {
        "record_id": record.get("record_id", ""),
        "mode": "experiment1_onevoice_agnostic",
        "bundle_duration_s": record.get("bundle_duration_s", 0),
        "utterances": utts,
    }


def _ensure_metadata_fields(meta: dict[str, Any]) -> dict[str, Any]:
    required = [
        "dataset_name",
        "dataset_split",
        "text_annotation_source",
        "text_annotation_details",
        "text_coverage",
        "phoneme_annotation_source",
        "phoneme_annotation_details",
        "phoneme_coverage",
    ]
    out = {}
    for k in required:
        v = meta.get(k, "")
        out[k] = v if isinstance(v, str) else str(v)
    return out


def _shift_time_str(ts: Any, offset_s: float) -> str:
    v = parse_ts(ts)
    if v is None:
        return ts_from_seconds(offset_s)
    return ts_from_seconds(v + offset_s)


def build_compact_bundle_onevoice(record: dict[str, Any]) -> dict[str, Any]:
    utterances = record.get("utterances", [])
    turns: list[dict[str, Any]] = []
    speakers_by_id: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] | None = None
    session_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sample_rate: int | float | str = 22050
    audio_format = "wav"
    offset = 0.0

    for idx, u in enumerate(utterances):
        rel_json = str(u.get("onevoice_json", "")).strip()
        if not rel_json:
            continue
        p = resolve_rel(rel_json)
        if not p.exists():
            continue
        src = load_json(p)
        if not isinstance(src, dict):
            continue

        if isinstance(src.get("session_date"), str) and src["session_date"]:
            session_date = src["session_date"]
        if "sample_rate" in src:
            sample_rate = src.get("sample_rate", sample_rate)
        if isinstance(src.get("audio_format"), str) and src["audio_format"]:
            audio_format = src["audio_format"]

        if metadata is None and isinstance(src.get("metadata"), dict):
            metadata = _ensure_metadata_fields(src["metadata"])

        src_speakers = src.get("speakers", [])
        if isinstance(src_speakers, list):
            for s in src_speakers:
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("speaker_id", "")).strip()
                if sid and sid not in speakers_by_id:
                    speakers_by_id[sid] = copy.deepcopy(s)

        src_turns = src.get("turns", [])
        if not isinstance(src_turns, list) or not src_turns:
            continue
        st = src_turns[0] if isinstance(src_turns[0], dict) else {}
        if not st:
            continue

        local_start = parse_ts(st.get("start")) or 0.0
        local_end = parse_ts(st.get("end"))
        fallback_duration = float(u.get("duration_s", 0.0) or 0.0)
        if local_end is None or local_end <= local_start:
            local_end = local_start + (fallback_duration if fallback_duration > 0 else 0.5)
        duration = max(local_end - local_start, 0.001)

        turn = {
            "turn_index": idx,
            "utt_id": st.get("utt_id", u.get("utt", "")),
            "speaker_id": st.get("speaker_id", u.get("speaker_id", "")),
            "reference_transcript": st.get("reference_transcript", ""),
            "orthographic_transcript": st.get("orthographic_transcript", ""),
            "phonetic_transcript": st.get("phonetic_transcript", ""),
            "ipa_transcript": st.get("ipa_transcript", ""),
            "raw_transcript": st.get("raw_transcript", ""),
            "start": ts_from_seconds(offset),
            "end": ts_from_seconds(offset + duration),
            "word_alignments": [],
            "mispronunciations": [],
            "behavioral_events": [],
        }

        words = st.get("word_alignments", [])
        if isinstance(words, list):
            for w_idx, w in enumerate(words):
                if not isinstance(w, dict):
                    continue
                w_start = parse_ts(w.get("start"))
                w_end = parse_ts(w.get("end"))
                if w_start is None or w_end is None or w_end <= w_start:
                    continue
                shifted_w = {
                    "word_index": w.get("word_index", w_idx),
                    "word": w.get("word", ""),
                    "start": ts_from_seconds(offset + max(0.0, (w_start - local_start))),
                    "end": ts_from_seconds(offset + max(0.0, (w_end - local_start))),
                }
                turn["word_alignments"].append(shifted_w)

        turns.append(turn)
        offset += duration

    if metadata is None:
        metadata = _ensure_metadata_fields({"dataset_name": "uxssd", "dataset_split": "selected_20"})

    speakers = list(speakers_by_id.values()) if speakers_by_id else [{"speaker_id": "UNKNOWN"}]
    return {
        "session_id": record.get("record_id", "bundle"),
        "session_date": session_date,
        "sample_rate": sample_rate,
        "audio_file_path": "",
        "audio_format": audio_format,
        "session_duration": ts_from_seconds(offset),
        "speakers": speakers,
        "turns": turns,
        "metadata": metadata,
    }


def validate_onevoice(data: dict[str, Any], validator_path: Path, work_dir: Path) -> tuple[bool, str]:
    candidate = work_dir / "candidate.json"
    write_json(candidate, data)
    cmd = [sys.executable, str(validator_path), str(candidate), "--mode", "full"]
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    log = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
    return cp.returncode == 0, log.strip()


def repair_with_validator(
    client: Any,
    cfg: RunnerConfig,
    agent_name: str,
    system_prompt: str,
    record_id: str,
    candidate: dict[str, Any],
    record_dir: Path,
) -> tuple[dict[str, Any], bool]:
    validator = (ROOT / cfg.validator_path) if not cfg.validator_path.is_absolute() else cfg.validator_path
    if not validator.exists():
        raise FileNotFoundError(f"Validator not found: {validator}")

    for attempt in range(cfg.max_repair_rounds + 1):
        ok, log = validate_onevoice(candidate, validator, record_dir / f"{agent_name}_validation")
        write_text(record_dir / f"{agent_name}_validator_round_{attempt}.log", log)
        if ok:
            return candidate, True
        if cfg.dry_run:
            return candidate, False
        if attempt >= cfg.max_repair_rounds:
            return candidate, False

        repair_payload = {
            "record_id": record_id,
            "agent": agent_name,
            "task": "Repair this OneVoice JSON so validator passes. Return corrected full JSON only.",
            "validator_errors": log[-6000:],
            "current_json": candidate,
        }
        repaired, raw = call_openai_json(
            client=client,
            model=cfg.model,
            system_prompt=system_prompt,
            user_payload=repair_payload,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        )
        write_text(record_dir / f"{agent_name}_repair_round_{attempt + 1}.txt", raw)
        candidate = repaired

    return candidate, False


def ensure_openai_client(cfg: RunnerConfig) -> Any:
    if cfg.dry_run:
        return None
    if OpenAI is None:
        raise RuntimeError("openai package not found. Install with: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()


def run_record_exp1(client: Any, cfg: RunnerConfig, record: dict[str, Any], record_dir: Path) -> tuple[bool, str]:
    payload = build_exp1_payload(record, cfg)
    write_json(record_dir / "input_payload_exp1.json", payload)

    if cfg.dry_run:
        a = normalize_exp1_result(_build_exp1_empty(record), record, include_mis=True, include_beh=False)
        b = normalize_exp1_result(_build_exp1_empty(record), record, include_mis=False, include_beh=True)
        merged = normalize_exp1_result(_build_exp1_empty(record), record, include_mis=True, include_beh=True)
        write_json(record_dir / "agentA_output.json", a)
        write_json(record_dir / "agentB_output.json", b)
        write_json(record_dir / "agentC_output.json", merged)
        return True, "dry_run"

    a_json, a_raw = call_openai_json(client, cfg.model, SYSTEM_EXP1_MIS, payload, cfg.temperature, cfg.max_output_tokens)
    write_text(record_dir / "agentA_raw.txt", a_raw)
    a_norm = normalize_exp1_result(a_json, record, include_mis=True, include_beh=False)
    write_json(record_dir / "agentA_output.json", a_norm)

    b_json, b_raw = call_openai_json(client, cfg.model, SYSTEM_EXP1_BEH, payload, cfg.temperature, cfg.max_output_tokens)
    write_text(record_dir / "agentB_raw.txt", b_raw)
    b_norm = normalize_exp1_result(b_json, record, include_mis=False, include_beh=True)
    write_json(record_dir / "agentB_output.json", b_norm)

    merge_payload = {"record_input": payload, "agentA_output": a_norm, "agentB_output": b_norm}
    c_json, c_raw = call_openai_json(client, cfg.model, SYSTEM_EXP1_MERGE, merge_payload, cfg.temperature, cfg.max_output_tokens)
    write_text(record_dir / "agentC_raw.txt", c_raw)
    c_norm = normalize_exp1_result(c_json, record, include_mis=True, include_beh=True)

    # Final dedupe
    for utt in c_norm["utterances"]:
        utt["mispronunciations"] = dedupe_events(_norm_event_list(utt.get("mispronunciations", [])))
        utt["behavioral_events"] = dedupe_events(_norm_event_list(utt.get("behavioral_events", [])))
    write_json(record_dir / "agentC_output.json", c_norm)
    write_json(record_dir / "final_output.json", c_norm)
    return True, "ok"


def run_record_exp2(client: Any, cfg: RunnerConfig, record: dict[str, Any], record_dir: Path) -> tuple[bool, str]:
    bundle = build_compact_bundle_onevoice(record)
    write_json(record_dir / "bundle_input_onevoice.json", bundle)
    expected_proxy = record.get("expected_proxy", {})
    expected_beh_target: int | None = None
    if isinstance(expected_proxy, dict):
        expected_beh_target = _safe_int_or_none(expected_proxy.get("behavioral_events"))

    if cfg.dry_run:
        write_json(record_dir / "agentA_output.json", bundle)
        write_json(record_dir / "agentB_output.json", bundle)
        write_json(record_dir / "agentC_output.json", bundle)
        write_json(record_dir / "final_output.json", bundle)
        return True, "dry_run"

    # Agent A: mispronunciation
    a_payload = {
        "record_id": record.get("record_id", ""),
        "task": "Detect and add mispronunciations. Keep behavioral_events unchanged.",
        "onevoice_json": bundle,
    }
    a_json, a_raw = call_openai_json(client, cfg.model, SYSTEM_EXP2_MIS, a_payload, cfg.temperature, cfg.max_output_tokens)
    write_text(record_dir / "agentA_raw.txt", a_raw)
    a_fixed, a_ok = repair_with_validator(client, cfg, "agentA", SYSTEM_EXP2_MIS, str(record.get("record_id", "")), a_json, record_dir)
    write_json(record_dir / "agentA_output.json", a_fixed)
    if not a_ok:
        write_json(record_dir / "final_output.json", a_fixed)
        return False, "agentA_failed_validation"

    # Agent B: behavioral events
    b_payload = {
        "record_id": record.get("record_id", ""),
        "task": "Detect and add behavioral_events. Preserve mispronunciations.",
        "onevoice_json": a_fixed,
    }
    if expected_beh_target is not None:
        b_payload["behavioral_event_target_hint"] = {
            "target_total_for_record": expected_beh_target,
            "tolerance": EXP2_BEH_TARGET_SLACK,
            "allowed_types": sorted(EXP2_BEH_ALLOWED_TYPES),
            "guidance": "Avoid routine one-per-turn transitions; add only evidence-backed events.",
        }
    b_json, b_raw = call_openai_json(client, cfg.model, SYSTEM_EXP2_BEH, b_payload, cfg.temperature, cfg.max_output_tokens)
    write_text(record_dir / "agentB_raw.txt", b_raw)
    b_fixed, b_ok = repair_with_validator(client, cfg, "agentB", SYSTEM_EXP2_BEH, str(record.get("record_id", "")), b_json, record_dir)
    b_fixed = sanitize_exp2_behavioral_bundle(b_fixed, expected_beh_target)
    write_json(record_dir / "agentB_output.json", b_fixed)
    if not b_ok:
        write_json(record_dir / "final_output.json", b_fixed)
        return False, "agentB_failed_validation"

    # Agent C: final merge/cleanup
    c_payload = {
        "record_id": record.get("record_id", ""),
        "task": "Final merge and OneVoice cleanup. Preserve semantic content.",
        "onevoice_json": b_fixed,
    }
    c_json, c_raw = call_openai_json(client, cfg.model, SYSTEM_EXP2_MERGE, c_payload, cfg.temperature, cfg.max_output_tokens)
    write_text(record_dir / "agentC_raw.txt", c_raw)
    c_fixed, c_ok = repair_with_validator(client, cfg, "agentC", SYSTEM_EXP2_MERGE, str(record.get("record_id", "")), c_json, record_dir)
    c_fixed = sanitize_exp2_behavioral_bundle(c_fixed, expected_beh_target)
    write_json(record_dir / "agentC_output.json", c_fixed)
    write_json(record_dir / "final_output.json", c_fixed)
    if not c_ok:
        return False, "agentC_failed_validation"

    return True, "ok"


def select_records(records: list[dict[str, Any]], cfg: RunnerConfig) -> list[dict[str, Any]]:
    out = records
    if cfg.record_ids:
        out = [r for r in out if str(r.get("record_id", "")) in cfg.record_ids]
    if cfg.max_records is not None:
        out = out[: cfg.max_records]
    return out


def run_experiment(cfg: RunnerConfig) -> Path:
    records = load_jsonl(cfg.manifest)
    records = select_records(records, cfg)
    if not records:
        raise RuntimeError("No records selected for run.")

    run_dir = cfg.output_root / f"{cfg.mode}_{stamp_for_dir()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = {
        "mode": cfg.mode,
        "manifest": str(cfg.manifest).replace("\\", "/"),
        "output_root": str(cfg.output_root).replace("\\", "/"),
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_records": cfg.max_records,
        "record_ids": sorted(cfg.record_ids) if cfg.record_ids else [],
        "max_file_chars": cfg.max_file_chars,
        "max_files_per_utt": cfg.max_files_per_utt,
        "max_utterances_per_record": cfg.max_utterances_per_record,
        "max_repair_rounds": cfg.max_repair_rounds,
        "validator_path": str(cfg.validator_path).replace("\\", "/"),
        "dry_run": cfg.dry_run,
        "run_started_utc": now_utc(),
        "record_count": len(records),
    }
    write_json(run_dir / "run_config.json", cfg_dump)

    client = ensure_openai_client(cfg)
    summary_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records, start=1):
        record_id = str(record.get("record_id", f"record_{idx:03d}"))
        record_dir = run_dir / record_id
        record_dir.mkdir(parents=True, exist_ok=True)
        write_json(record_dir / "record_manifest_entry.json", record)
        print(f"[{idx}/{len(records)}] Running {record_id} ...")

        try:
            if cfg.mode == "exp1":
                ok, status = run_record_exp1(client, cfg, record, record_dir)
            elif cfg.mode == "exp2":
                ok, status = run_record_exp2(client, cfg, record, record_dir)
            else:
                raise ValueError(f"Unsupported mode: {cfg.mode}")
        except Exception as exc:
            msg = str(exc)
            if "Failed to parse model JSON" in msg or "Expecting ',' delimiter" in msg:
                msg = f"{msg} | Hint: try increasing --max-output-tokens (e.g., 16000)."
            ok, status = False, f"exception: {msg}"
            write_text(record_dir / "error.txt", str(exc))

        summary_rows.append(
            {
                "record_id": record_id,
                "status": status,
                "ok": ok,
                "bundle_duration_s": record.get("bundle_duration_s", 0),
                "utterance_count": len(record.get("utterances", [])),
            }
        )
        print(f"  -> {status}")

    write_json(run_dir / "summary.json", summary_rows)
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["record_id", "ok", "status", "bundle_duration_s", "utterance_count"])
        wr.writeheader()
        for row in summary_rows:
            wr.writerow(row)

    return run_dir


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-records", type=int, default=None, help="Limit number of records.")
    parser.add_argument("--record-ids", type=str, default="", help="Comma-separated record IDs.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call OpenAI API; generate placeholder outputs.")
    parser.add_argument("--max-output-tokens", type=int, default=12000, help="Max output tokens per agent call.")


def parse_record_ids(csv_ids: str) -> set[str] | None:
    ids = {x.strip() for x in csv_ids.split(",") if x.strip()}
    return ids if ids else None
