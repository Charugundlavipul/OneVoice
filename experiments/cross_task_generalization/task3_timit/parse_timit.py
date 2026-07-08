from __future__ import annotations

from pathlib import Path
from typing import Any


SAMPLE_RATE = 16000
CLOSURE_LABELS = {"bcl", "dcl", "gcl", "pcl", "tcl", "kcl"}
PAUSE_LABELS = {"h#", "pau", "epi"}


def read_txt(path: Path) -> dict[str, Any]:
    line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    parts = line.strip().split(maxsplit=2)
    if len(parts) < 3:
        return {"start_sample": 0, "end_sample": 0, "transcript": line}
    return {"start_sample": int(parts[0]), "end_sample": int(parts[1]), "transcript": parts[2]}


def read_intervals(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        start_sample = int(parts[0])
        end_sample = int(parts[1])
        rows.append(
            {
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start": start_sample / SAMPLE_RATE,
                "end": end_sample / SAMPLE_RATE,
                "label": parts[2],
                "duration": (end_sample - start_sample) / SAMPLE_RATE,
            }
        )
    return rows


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def parent_word_index(phone: dict[str, Any], words: list[dict[str, Any]]) -> int | None:
    best_idx: int | None = None
    best_overlap = 0.0
    for idx, word in enumerate(words):
        ov = overlap(phone["start"], phone["end"], word["start"], word["end"])
        if ov > best_overlap:
            best_overlap = ov
            best_idx = idx
    return best_idx


def attach_phone_links(words: list[dict[str, Any]], phones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    linked_words = []
    for idx, word in enumerate(words):
        copy = dict(word)
        copy["word_index"] = idx
        copy["phone_indices"] = []
        linked_words.append(copy)
    for pidx, phone in enumerate(phones):
        widx = parent_word_index(phone, words)
        phone["phone_index"] = pidx
        phone["word_index"] = widx
        if widx is not None:
            linked_words[widx]["phone_indices"].append(pidx)
    return linked_words


def parse_utterance(txt_path: Path) -> dict[str, Any]:
    base = txt_path.with_suffix("")
    wrd_path = base.with_suffix(".WRD")
    phn_path = base.with_suffix(".PHN")
    txt = read_txt(txt_path)
    words = read_intervals(wrd_path)
    phones = read_intervals(phn_path)
    words = attach_phone_links(words, phones)
    speaker_id = txt_path.parent.name.upper()
    utt_id = f"{speaker_id}_{txt_path.stem.upper()}"
    return {
        "utt_id": utt_id,
        "speaker_id": speaker_id,
        "txt_path": txt_path,
        "wrd_path": wrd_path,
        "phn_path": phn_path,
        "transcript": txt["transcript"],
        "duration": txt["end_sample"] / SAMPLE_RATE if txt["end_sample"] else (phones[-1]["end"] if phones else 0.0),
        "words": words,
        "phones": phones,
    }


def find_timit_utterances(root: Path, split: str | None = None) -> list[Path]:
    base = root / split if split else root
    paths = []
    for txt in base.rglob("*.TXT"):
        stem = txt.with_suffix("")
        if stem.with_suffix(".WRD").exists() and stem.with_suffix(".PHN").exists():
            paths.append(txt)
    return sorted(paths, key=lambda p: p.as_posix().lower())
