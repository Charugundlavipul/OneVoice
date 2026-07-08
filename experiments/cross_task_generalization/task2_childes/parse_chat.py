from __future__ import annotations

import re
from pathlib import Path
from typing import Any


MAIN_TIER_RE = re.compile(r"^\*([A-Z0-9_]+):\s*(.*)$")
DEPENDENT_RE = re.compile(r"^%([A-Za-z0-9_]+):\s*(.*)$")
PARTICIPANTS_RE = re.compile(r"([A-Z0-9_]{2,})\s+([^,]+)")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_chat_text(raw: str) -> str:
    text = re.sub(r"\x15\s*\d+\s*_\s*\d+\s*\x15", " ", raw)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"&=[A-Za-z0-9_:.-]+", " ", text)
    text = re.sub(r"[<>]", " ", text)
    return normalize_space(text)


def parse_participants(value: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for match in PARTICIPANTS_RE.finditer(value):
        code = match.group(1).strip()
        desc = match.group(2).strip()
        role = "child" if code == "CHI" or "child" in desc.lower() else "adult"
        out[code] = {"speaker_id": code, "description": desc, "role": role}
    return out


def detect_events(raw: str, dependent_tiers: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    text = raw
    tier_text = " ".join(" ".join(v) for v in (dependent_tiers or {}).values())
    combined = f"{text} {tier_text}"
    events: list[dict[str, Any]] = []

    for m in re.finditer(r"\[/\]|\[x\s+\d+\]", text):
        events.append({"event_type": "repetition", "source": m.group(0)})
    for m in re.finditer(r"\[//\]|\[///\]|\+//\.|\+/\.", text):
        events.append({"event_type": "repair", "source": m.group(0)})
    for m in re.finditer(r"\(\.{1,3}\)|\(\d+(?:\.\d+)?\.\)", text):
        events.append({"event_type": "pause", "source": m.group(0)})
    for m in re.finditer(r"&=[A-Za-z0-9_:.-]+|\[=!\s*[^\]]+\]|\bxxx\b|\byyy\b", combined, flags=re.I):
        events.append({"event_type": "non_speech", "source": m.group(0)})
    for m in re.finditer(r"\[<\]|\[>\]|\+<", text):
        events.append({"event_type": "overlap", "source": m.group(0)})

    return events


def parse_chat_file(path: Path) -> dict[str, Any]:
    participants: dict[str, dict[str, str]] = {}
    utterances: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("@Participants:"):
            participants.update(parse_participants(line.split(":", 1)[1]))
            continue
        main = MAIN_TIER_RE.match(line)
        if main:
            if current is not None:
                current["events"] = detect_events(current["raw"], current["dependent_tiers"])
                utterances.append(current)
            speaker = main.group(1)
            raw = main.group(2).strip()
            if speaker not in participants:
                participants[speaker] = {
                    "speaker_id": speaker,
                    "description": "",
                    "role": "child" if speaker == "CHI" else "adult",
                }
            current = {
                "utt_index": len(utterances),
                "speaker_id": speaker,
                "raw": raw,
                "text": clean_chat_text(raw),
                "dependent_tiers": {},
            }
            continue
        dep = DEPENDENT_RE.match(line)
        if dep and current is not None:
            current["dependent_tiers"].setdefault(dep.group(1), []).append(dep.group(2).strip())

    if current is not None:
        current["events"] = detect_events(current["raw"], current["dependent_tiers"])
        utterances.append(current)

    return {
        "source_file": path,
        "participants": participants,
        "utterances": utterances,
    }


def window_to_chat_text(window: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for utt in window:
        lines.append(f"*{utt['speaker_id']}:\t{utt['raw']}")
        for tier, values in sorted((utt.get("dependent_tiers") or {}).items()):
            for value in values:
                lines.append(f"%{tier}:\t{value}")
    return "\n".join(lines)
