#!/usr/bin/env python3
"""
Reference toolkit converter (folder-discovery first).

Design goals:
- One speaker-details CSV
- One dataset-metadata JSON
- Fixed input folders for audio/textgrid/diarization/transcripts/prompts/cha
- No path columns required in CSV
- Auto-discover files by session key (audio stem), similar to Ultrasuite converters
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import wave
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


METADATA_KEYS = [
    "dataset_name",
    "dataset_split",
    "text_annotation_source",
    "text_annotation_details",
    "text_coverage",
    "phoneme_annotation_source",
    "phoneme_annotation_details",
    "phoneme_coverage",
]

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm"}


def clean(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return str(value).strip()


def maybe_set(target: Dict[str, Any], key: str, value: Optional[Any]) -> None:
    v = clean(value)
    if v != "":
        target[key] = v


def parse_num_or_str(value: Optional[str]) -> Any:
    v = clean(value)
    if v == "":
        return ""
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def parse_int_or_str(value: Optional[str]) -> Any:
    v = clean(value)
    if v == "":
        return ""
    try:
        return int(v)
    except Exception:
        return v


def seconds_to_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{ms:03d}"


def timestamp_to_seconds(ts: Optional[str]) -> Optional[float]:
    s = clean(ts)
    if s == "":
        return None
    parts = s.split(":")
    if len(parts) != 4:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        sec = int(parts[2])
        ms = int(parts[3])
        return float(h * 3600 + m * 60 + sec) + float(ms) / 1000.0
    except Exception:
        return None


def wav_info(path: Path) -> Tuple[Optional[int], Optional[float]]:
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            nframes = wf.getnframes()
            dur = float(nframes) / float(sr) if sr > 0 else None
            return sr, dur
    except Exception:
        return None, None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def split_nonempty_lines(text: str) -> List[str]:
    return [clean(x) for x in text.splitlines() if clean(x) != ""]


def assign_lines_to_turns(turns: List[Dict[str, Any]], lines: List[str], field: str) -> None:
    if not turns or not lines:
        return

    if len(lines) == 1:
        turns[0][field] = lines[0]
        return

    if len(lines) == len(turns):
        for i, line in enumerate(lines):
            turns[i][field] = line
        return

    n = min(len(lines), len(turns))
    for i in range(n):
        turns[i][field] = lines[i]

    if len(lines) > len(turns):
        extra = " ".join(lines[len(turns) :]).strip()
        if extra:
            prev = clean(turns[-1].get(field, ""))
            turns[-1][field] = f"{prev} {extra}".strip() if prev else extra


def parse_cha_utterances(cha_path: Path) -> List[Tuple[str, str]]:
    if not cha_path.exists():
        return []

    out: List[Tuple[str, str]] = []
    lines = cha_path.read_text(encoding="utf-8", errors="replace").splitlines()

    cur_spk = ""
    cur_text: List[str] = []

    def flush() -> None:
        nonlocal cur_spk, cur_text
        if cur_spk and cur_text:
            joined = " ".join(cur_text).strip()
            if joined:
                out.append((cur_spk, joined))
        cur_spk = ""
        cur_text = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("*"):
            flush()
            if ":" in line:
                spk, txt = line[1:].split(":", 1)
                cur_spk = clean(spk)
                cur_text = [clean(txt)]
        elif cur_spk and line and not line.startswith("%"):
            cur_text.append(clean(line))

    flush()
    return out


def default_metadata() -> Dict[str, str]:
    return {k: "" for k in METADATA_KEYS}


def read_dataset_metadata(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing dataset metadata file: {path}")

    cfg = json.loads(path.read_text(encoding="utf-8"))

    meta = default_metadata()
    nested_meta = cfg.get("metadata", {}) if isinstance(cfg.get("metadata", {}), dict) else {}
    for k in METADATA_KEYS:
        v = cfg.get(k)
        if clean(v) == "":
            v = nested_meta.get(k)
        meta[k] = clean(v)

    defaults = {
        "default_language": clean(cfg.get("default_language") or cfg.get("defaults", {}).get("language", "")),
        "default_native_language": clean(
            cfg.get("default_native_language") or cfg.get("defaults", {}).get("native_language", "")
        ),
        "default_age_group": clean(cfg.get("default_age_group") or cfg.get("defaults", {}).get("age_group", "")),
        "default_accent": clean(cfg.get("default_accent") or cfg.get("defaults", {}).get("accent", "")),
        "default_sample_rate": clean(cfg.get("default_sample_rate") or cfg.get("defaults", {}).get("sample_rate", "")),
    }

    return meta, defaults

def load_speaker_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing speaker details CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if "speaker_key" not in cols:
            raise ValueError(f"{path}: required column missing: speaker_key")
        rows = []
        for row in reader:
            rows.append({k: clean(v) for k, v in row.items()})
        return rows


def speaker_indices(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    by_key: Dict[str, Dict[str, str]] = {}
    by_id: Dict[str, Dict[str, str]] = {}

    for row in rows:
        key = clean(row.get("speaker_key", "")).lower()
        sid = clean(row.get("speaker_id", "")).lower()
        if key:
            by_key[key] = row
        if sid:
            by_id[sid] = row

    return by_key, by_id


def session_prefix(session_id: str) -> str:
    sid = clean(session_id)
    if "-" in sid:
        return sid.split("-", 1)[0]
    if "_" in sid:
        return sid.split("_", 1)[0]
    return sid


def match_speaker_row(
    speaker_label: str,
    speaker_hint: str,
    session_id: str,
    by_key: Dict[str, Dict[str, str]],
    by_id: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, str]]:
    candidates = []
    sl = clean(speaker_label)
    sh = clean(speaker_hint)
    sp = session_prefix(session_id)

    if sl:
        candidates.extend([sl, sl.replace("_CHILD", ""), sl.replace("_ADULT", "")])
    if sh:
        candidates.append(sh)
    if sp:
        candidates.append(sp)

    for c in candidates:
        key = c.lower()
        if key in by_id:
            return by_id[key]
        if key in by_key:
            return by_key[key]
    return None


def build_speaker_obj(
    speaker_label: str,
    speaker_hint: str,
    session_id: str,
    by_key: Dict[str, Dict[str, str]],
    by_id: Dict[str, Dict[str, str]],
    defaults: Dict[str, str],
) -> Dict[str, Any]:
    row = match_speaker_row(speaker_label, speaker_hint, session_id, by_key, by_id)
    base_id = clean(speaker_label) or clean(speaker_hint) or "SPK_001"

    if row is None:
        obj: Dict[str, Any] = {"speaker_id": base_id}
        maybe_set(obj, "language", defaults.get("default_language", ""))
        maybe_set(obj, "native_language", defaults.get("default_native_language", ""))
        maybe_set(obj, "age_group", defaults.get("default_age_group", ""))
        maybe_set(obj, "accent", defaults.get("default_accent", ""))
        return obj

    obj = {"speaker_id": clean(row.get("speaker_id", "")) or base_id}
    maybe_set(obj, "gender", row.get("gender", ""))
    maybe_set(obj, "language", row.get("language", "") or defaults.get("default_language", ""))
    maybe_set(
        obj,
        "native_language",
        row.get("native_language", "") or defaults.get("default_native_language", ""),
    )
    maybe_set(obj, "accent", row.get("accent", "") or defaults.get("default_accent", ""))
    maybe_set(obj, "name", row.get("name", ""))
    maybe_set(obj, "age_group", row.get("age_group", "") or defaults.get("default_age_group", ""))

    age = parse_num_or_str(row.get("age", ""))
    if age != "":
        obj["age"] = age

    return obj


def discover_audio_files(audio_root: Path, max_files: int = 0) -> List[Path]:
    if not audio_root.exists():
        return []
    files = [
        p
        for p in audio_root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    files.sort(key=lambda p: p.as_posix().lower())
    if max_files > 0:
        files = files[:max_files]
    return files


def index_files_by_stem(root: Path, exts: Optional[List[str]] = None) -> Dict[str, List[Path]]:
    idx: DefaultDict[str, List[Path]] = defaultdict(list)
    if not root.exists():
        return dict(idx)

    extset = {e.lower() for e in exts} if exts else None
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if extset and p.suffix.lower() not in extset:
            continue
        idx[p.stem.lower()].append(p)

    for k in list(idx.keys()):
        idx[k] = sorted(idx[k], key=lambda x: x.as_posix().lower())
    return dict(idx)


def choose_stem_match(stem: str, idx: Dict[str, List[Path]], speaker_hint: str = "") -> Optional[Path]:
    cands = idx.get(stem.lower(), [])
    if not cands:
        return None
    if speaker_hint:
        sh = speaker_hint.lower()
        for p in cands:
            if sh in [x.lower() for x in p.parts]:
                return p
    return cands[0]


def parse_rttm_file(path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    out: DefaultDict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0].upper() != "SPEAKER":
            continue
        file_id = clean(parts[1])
        if file_id == "":
            continue
        try:
            start = float(parts[3])
            dur = float(parts[4])
            end = start + max(0.0, dur)
        except Exception:
            continue
        speaker = clean(parts[7])
        if speaker == "" or speaker == "<NA>":
            speaker = "SPK_001"
        out[file_id].append((start, end, speaker))

    for file_id in list(out.keys()):
        out[file_id].sort(key=lambda x: (x[0], x[1], x[2]))
    return dict(out)


def build_rttm_index(diar_root: Path) -> Tuple[Dict[str, List[Tuple[float, float, str]]], Dict[str, List[Tuple[float, float, str]]]]:
    by_fileid: DefaultDict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    by_stem: DefaultDict[str, List[Tuple[float, float, str]]] = defaultdict(list)

    if not diar_root.exists():
        return dict(by_fileid), dict(by_stem)

    for rttm in sorted(diar_root.rglob("*.rttm"), key=lambda p: p.as_posix().lower()):
        parsed = parse_rttm_file(rttm)
        merged_for_this_file: List[Tuple[float, float, str]] = []
        for file_id, segs in parsed.items():
            by_fileid[file_id.lower()].extend(segs)
            merged_for_this_file.extend(segs)
        merged_for_this_file.sort(key=lambda x: (x[0], x[1], x[2]))
        if merged_for_this_file:
            by_stem[rttm.stem.lower()].extend(merged_for_this_file)

    for k in list(by_fileid.keys()):
        by_fileid[k].sort(key=lambda x: (x[0], x[1], x[2]))
    for k in list(by_stem.keys()):
        by_stem[k].sort(key=lambda x: (x[0], x[1], x[2]))

    return dict(by_fileid), dict(by_stem)


def build_turns_from_rttm(session_id: str, segs: List[Tuple[float, float, str]], fallback_spk: str) -> List[Dict[str, Any]]:
    turns = []
    for i, (start, end, spk) in enumerate(segs):
        turns.append(
            {
                "turn_index": i,
                "utt_id": f"{session_id}_utt_{i:04d}",
                "speaker_id": clean(spk) or fallback_spk,
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
                "word_alignments": [],
                "mispronunciations": [],
                "behavioral_events": [],
            }
        )
    return turns

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1]
    return s


def parse_textgrid_any(content: str) -> Dict[str, List[Dict[str, Any]]]:
    if "item [" in content and "intervals [" in content:
        # Robust long-format parsing via per-item regex blocks.
        tiers: Dict[str, List[Dict[str, Any]]] = {}
        item_blocks = re.finditer(r"item \[\d+\]:(.*?)(?=item \[\d+\]:|\Z)", content, flags=re.DOTALL)
        for item in item_blocks:
            block = item.group(1)
            cls = re.search(r'class\s*=\s*"([^"]+)"', block)
            if not cls or cls.group(1).strip() != "IntervalTier":
                continue
            name_m = re.search(r'name\s*=\s*"([^"]+)"', block)
            if not name_m:
                continue
            tier_name = name_m.group(1).strip()

            intervals: List[Dict[str, Any]] = []
            int_blocks = re.finditer(r"intervals \[\d+\]:(.*?)(?=intervals \[\d+\]:|\Z)", block, flags=re.DOTALL)
            for ib in int_blocks:
                b = ib.group(1)
                xmin_m = re.search(r"xmin\s*=\s*([0-9eE+\-.]+)", b)
                xmax_m = re.search(r"xmax\s*=\s*([0-9eE+\-.]+)", b)
                text_m = re.search(r'text\s*=\s*"([^"]*)"', b)
                if not xmin_m or not xmax_m:
                    continue
                try:
                    xmin = float(xmin_m.group(1))
                    xmax = float(xmax_m.group(1))
                except Exception:
                    continue
                if xmax < xmin:
                    xmin, xmax = xmax, xmin
                intervals.append({"xmin": xmin, "xmax": xmax, "text": text_m.group(1).strip() if text_m else ""})
            if intervals:
                tiers[tier_name] = intervals
        if tiers:
            return tiers
    return _parse_textgrid_short(content)


def _parse_textgrid_short(text: str) -> Dict[str, List[Dict[str, Any]]]:
    tiers: Dict[str, List[Dict[str, Any]]] = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    n = len(lines)

    while i < n and lines[i].strip('"') not in ("IntervalTier", "TextTier"):
        i += 1

    while i < n:
        if lines[i].strip('"') not in ("IntervalTier", "TextTier"):
            i += 1
            continue

        tier_type = lines[i].strip('"')
        i += 1
        if i >= n:
            break

        name = _strip_quotes(lines[i])
        i += 1
        if i + 2 >= n:
            break

        try:
            float(lines[i])
            float(lines[i + 1])
            i += 2
        except Exception:
            i += 1
            continue

        try:
            size = int(float(lines[i]))
            i += 1
        except Exception:
            size = 0
            i += 1

        entries: List[Dict[str, Any]] = []
        if tier_type == "IntervalTier":
            for _ in range(size):
                if i + 2 >= n:
                    break
                try:
                    xmin = float(lines[i])
                    xmax = float(lines[i + 1])
                    txt = _strip_quotes(lines[i + 2])
                    if xmax < xmin:
                        xmin, xmax = xmax, xmin
                    entries.append({"xmin": xmin, "xmax": xmax, "text": txt})
                except Exception:
                    pass
                i += 3

        tiers[name] = entries

    return tiers


def choose_tier(tiers: Dict[str, List[Dict[str, Any]]], keywords: List[str]) -> List[Dict[str, Any]]:
    if not tiers:
        return []

    names = list(tiers.keys())
    lowered = {name: name.lower() for name in names}

    for kw in keywords:
        for name in names:
            if lowered[name] == kw:
                return tiers[name]

    for kw in keywords:
        for name in names:
            if kw in lowered[name]:
                return tiers[name]

    return tiers[names[0]]


def textgrid_alignments(textgrid_path: Path) -> Dict[str, Any]:
    tiers = parse_textgrid_any(read_text(textgrid_path))
    word_tier = choose_tier(tiers, ["word", "words", "transcript", "orthography"]) if tiers else []
    phone_tier = choose_tier(tiers, ["phone", "phones", "phoneme", "phonemes", "segment", "segments"]) if tiers else []

    words = []
    for it in sorted(word_tier, key=lambda x: float(x.get("xmin", 0.0))):
        a = float(it.get("xmin", 0.0))
        b = float(it.get("xmax", 0.0))
        if b < a:
            a, b = b, a
        words.append({"xmin": a, "xmax": b, "text": clean(it.get("text", ""))})

    phones = []
    for it in sorted(phone_tier, key=lambda x: float(x.get("xmin", 0.0))):
        a = float(it.get("xmin", 0.0))
        b = float(it.get("xmax", 0.0))
        if b < a:
            a, b = b, a
        phones.append({"xmin": a, "xmax": b, "text": clean(it.get("text", ""))})

    word_objs: List[Dict[str, Any]] = []
    for wi, w in enumerate(words):
        word_objs.append(
            {
                "word_index": wi,
                "word": w["text"],
                "start": seconds_to_timestamp(w["xmin"]),
                "end": seconds_to_timestamp(w["xmax"]),
                "phonemes": [],
                "_xmin": w["xmin"],
                "_xmax": w["xmax"],
            }
        )

    for ph in phones:
        best_i = -1
        best_ov = 0.0
        for i, w in enumerate(word_objs):
            ov = min(ph["xmax"], w["_xmax"]) - max(ph["xmin"], w["_xmin"])
            if ov > best_ov:
                best_ov = ov
                best_i = i
        if best_i >= 0 and best_ov > 0:
            word_objs[best_i]["phonemes"].append(
                {
                    "phone": ph["text"],
                    "start": seconds_to_timestamp(ph["xmin"]),
                    "end": seconds_to_timestamp(ph["xmax"]),
                }
            )

    for w in word_objs:
        for i, p in enumerate(w["phonemes"]):
            p["phone_index"] = i
        del w["_xmin"]
        del w["_xmax"]

    orth = " ".join([w["word"] for w in word_objs if clean(w["word"]) != ""]).strip()
    phn_tokens: List[str] = []
    for w in word_objs:
        for p in w["phonemes"]:
            if clean(p.get("phone", "")):
                phn_tokens.append(clean(p["phone"]))
    phn = " ".join(phn_tokens).strip()

    starts = [w["xmin"] for w in words] + [p["xmin"] for p in phones]
    ends = [w["xmax"] for w in words] + [p["xmax"] for p in phones]

    return {
        "word_alignments": word_objs,
        "orthographic_transcript": orth,
        "phonetic_transcript": phn,
        "start_sec": min(starts) if starts else None,
        "end_sec": max(ends) if ends else None,
    }


def has_text(turn: Dict[str, Any]) -> bool:
    for key in ("raw_transcript", "orthographic_transcript", "reference_transcript"):
        if clean(turn.get(key, "")) != "":
            return True
    return False


def has_phonetics(turn: Dict[str, Any]) -> bool:
    if clean(turn.get("phonetic_transcript", "")) != "" or clean(turn.get("ipa_transcript", "")) != "":
        return True
    for w in turn.get("word_alignments", []):
        if w.get("phonemes"):
            return True
    return False


def infer_coverage(metadata: Dict[str, str], turns: List[Dict[str, Any]]) -> Dict[str, str]:
    out = dict(metadata)
    n = len(turns)
    txt = sum(1 for t in turns if has_text(t))
    phn = sum(1 for t in turns if has_phonetics(t))

    def cov(count: int) -> str:
        if count == 0:
            return "none"
        if count == n:
            return "full"
        return "partial"

    text_cov = clean(out.get("text_coverage", ""))
    if text_cov in ("", "auto"):
        out["text_coverage"] = cov(txt)

    phn_cov = clean(out.get("phoneme_coverage", ""))
    if phn_cov in ("", "auto"):
        out["phoneme_coverage"] = cov(phn)

    text_src = clean(out.get("text_annotation_source", ""))
    if text_src in ("", "auto"):
        out["text_annotation_source"] = "human" if out["text_coverage"] != "none" else "none"

    phn_src = clean(out.get("phoneme_annotation_source", ""))
    if phn_src in ("", "auto"):
        out["phoneme_annotation_source"] = (
            "machine_with_partial_human_review" if out["phoneme_coverage"] != "none" else "none"
        )

    return out

def build_records(root: Path, speaker_csv: Path, metadata_json: Path, max_files: int = 0) -> List[Dict[str, Any]]:
    meta_base, defaults = read_dataset_metadata(metadata_json)

    speaker_rows = load_speaker_rows(speaker_csv)
    by_key, by_id = speaker_indices(speaker_rows)

    audio_root = root / "input" / "audio"
    textgrid_root = root / "input" / "textgrid"
    diar_root = root / "input" / "diarization"
    transcripts_root = root / "input" / "transcripts"
    prompts_root = root / "input" / "prompts"
    cha_root = root / "input" / "cha"

    audios = discover_audio_files(audio_root, max_files=max_files)
    if not audios:
        return []

    tg_index = index_files_by_stem(textgrid_root, [".textgrid"])
    tr_index = index_files_by_stem(transcripts_root, [".txt"])
    pr_index = index_files_by_stem(prompts_root, [".txt"])
    cha_index = index_files_by_stem(cha_root, [".cha"])

    rttm_by_fileid, rttm_by_stem = build_rttm_index(diar_root)

    records: List[Dict[str, Any]] = []

    for audio in audios:
        session_id = audio.stem
        rel_parts = audio.relative_to(audio_root).parts
        path_hint = rel_parts[0] if len(rel_parts) >= 2 else ""
        speaker_lookup_hint = session_prefix(session_id)

        audio_rel = audio.relative_to(root).as_posix()
        audio_format = clean(audio.suffix).lstrip(".").lower()

        wav_sr, wav_dur = wav_info(audio)
        sr = wav_sr
        if sr is None:
            d_sr = parse_num_or_str(defaults.get("default_sample_rate", ""))
            sr = d_sr if d_sr != "" else None

        segs = list(rttm_by_fileid.get(session_id.lower(), []))
        if not segs:
            segs = list(rttm_by_stem.get(session_id.lower(), []))

        turns: List[Dict[str, Any]] = []
        if segs:
            base_spk = clean(segs[0][2]) or "SPK_001"
            turns = build_turns_from_rttm(session_id, segs, fallback_spk=base_spk)
        else:
            end_ts = seconds_to_timestamp(wav_dur) if wav_dur is not None else ""
            turns = [
                {
                    "turn_index": 0,
                    "utt_id": f"{session_id}_utt_0000",
                    "speaker_id": "SPK_001",
                    "start": "00:00:00:000" if end_ts else "",
                    "end": end_ts,
                    "word_alignments": [],
                    "mispronunciations": [],
                    "behavioral_events": [],
                }
            ]

        cha_file = choose_stem_match(session_id, cha_index, speaker_hint=path_hint)
        if cha_file:
            cha_utts = parse_cha_utterances(cha_file)
            if cha_utts and (not segs) and len(cha_utts) > 1:
                turns = []
                for i, (spk_code, utt_text) in enumerate(cha_utts):
                    turns.append(
                        {
                            "turn_index": i,
                            "utt_id": f"{session_id}_utt_{i:04d}",
                            "speaker_id": clean(spk_code) or "SPK_001",
                            "word_alignments": [],
                            "mispronunciations": [],
                            "behavioral_events": [],
                            "raw_transcript": utt_text,
                            "orthographic_transcript": utt_text,
                        }
                    )

        transcript_lines: List[str] = []
        tr_file = choose_stem_match(session_id, tr_index, speaker_hint=path_hint)
        if tr_file:
            transcript_lines = split_nonempty_lines(read_text(tr_file))

        if (not transcript_lines) and cha_file:
            transcript_lines = [txt for (_, txt) in parse_cha_utterances(cha_file)]

        assign_lines_to_turns(turns, transcript_lines, "raw_transcript")
        assign_lines_to_turns(turns, transcript_lines, "orthographic_transcript")

        pr_file = choose_stem_match(session_id, pr_index, speaker_hint=path_hint)
        if pr_file:
            prompt_lines = split_nonempty_lines(read_text(pr_file))
            assign_lines_to_turns(turns, prompt_lines, "reference_transcript")

        tg_file = choose_stem_match(session_id, tg_index, speaker_hint=path_hint)
        if tg_file:
            tg = textgrid_alignments(tg_file)
            if len(turns) == 1:
                turns[0]["word_alignments"] = tg["word_alignments"]
                if clean(turns[0].get("orthographic_transcript", "")) == "" and clean(tg["orthographic_transcript"]) != "":
                    turns[0]["orthographic_transcript"] = tg["orthographic_transcript"]
                if clean(turns[0].get("phonetic_transcript", "")) == "" and clean(tg["phonetic_transcript"]) != "":
                    turns[0]["phonetic_transcript"] = tg["phonetic_transcript"]
                if clean(turns[0].get("start", "")) == "" and tg["start_sec"] is not None:
                    turns[0]["start"] = seconds_to_timestamp(tg["start_sec"])
                if clean(turns[0].get("end", "")) == "" and tg["end_sec"] is not None:
                    turns[0]["end"] = seconds_to_timestamp(tg["end_sec"])
            else:
                for turn in turns:
                    ts = timestamp_to_seconds(turn.get("start", ""))
                    te = timestamp_to_seconds(turn.get("end", ""))
                    if ts is None or te is None:
                        continue
                    selected = []
                    for w in tg["word_alignments"]:
                        ws = timestamp_to_seconds(w.get("start", ""))
                        we = timestamp_to_seconds(w.get("end", ""))
                        if ws is None or we is None:
                            continue
                        mid = (ws + we) / 2.0
                        if ts - 1e-6 <= mid <= te + 1e-6:
                            selected.append(w)
                    rebuilt = []
                    for wi, w in enumerate(selected):
                        nw = {
                            "word_index": wi,
                            "word": clean(w.get("word", "")),
                            "start": clean(w.get("start", "")),
                            "end": clean(w.get("end", "")),
                            "phonemes": [],
                        }
                        for pi, p in enumerate(w.get("phonemes", [])):
                            nw["phonemes"].append(
                                {
                                    "phone": clean(p.get("phone", "")),
                                    "start": clean(p.get("start", "")),
                                    "end": clean(p.get("end", "")),
                                    "phone_index": pi,
                                }
                            )
                        rebuilt.append(nw)
                    if rebuilt:
                        turn["word_alignments"] = rebuilt

        for i, turn in enumerate(turns):
            if clean(str(turn.get("turn_index", ""))) == "":
                turn["turn_index"] = i
            turn.setdefault("word_alignments", [])
            turn.setdefault("mispronunciations", [])
            turn.setdefault("behavioral_events", [])
            turn.setdefault("raw_transcript", "")
            turn.setdefault("orthographic_transcript", "")
            turn.setdefault("phonetic_transcript", "")
            if clean(turn.get("speaker_id", "")) == "":
                turn["speaker_id"] = "SPK_001"
            turn["word_alignments"].sort(key=lambda w: parse_int_or_str(w.get("word_index", "")))
            for w in turn["word_alignments"]:
                w["phonemes"].sort(key=lambda p: parse_int_or_str(p.get("phone_index", "")))

        turns.sort(key=lambda t: parse_int_or_str(t.get("turn_index", "")))

        unique_turn_speakers = []
        seen = set()
        for t in turns:
            sid = clean(t.get("speaker_id", ""))
            if sid and sid.lower() not in seen:
                seen.add(sid.lower())
                unique_turn_speakers.append(sid)

        speakers = [
            build_speaker_obj(sid, speaker_lookup_hint, session_id, by_key, by_id, defaults)
            for sid in unique_turn_speakers
        ]
        if not speakers:
            speakers = [build_speaker_obj("", speaker_lookup_hint, session_id, by_key, by_id, defaults)]

        session_date = datetime.fromtimestamp(audio.stat().st_mtime).strftime("%Y-%m-%d")

        derived_end = 0.0
        for t in turns:
            te = timestamp_to_seconds(t.get("end", ""))
            if te is not None:
                derived_end = max(derived_end, te)

        duration_sec: Optional[float] = wav_dur
        if duration_sec is None and derived_end > 0:
            duration_sec = derived_end
        if duration_sec is not None and derived_end > duration_sec:
            duration_sec = derived_end

        metadata = infer_coverage(meta_base, turns)

        record: Dict[str, Any] = {
            "session_id": session_id,
            "session_date": session_date,
            "audio_file_path": audio_rel,
            "audio_format": audio_format,
            "speakers": speakers,
            "turns": turns,
            "metadata": metadata,
        }
        if sr is not None:
            record["sample_rate"] = sr
        if duration_sec is not None:
            record["session_duration"] = seconds_to_timestamp(duration_sec)

        records.append(record)

    return records

def write_outputs(records: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = out_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for rec in records:
        fp = sessions_dir / f"{rec['session_id']}.json"
        fp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(fp)

    jsonl = out_dir / "all.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    written.append(jsonl)

    return written


def run_validation(records: List[Dict[str, Any]], out_dir: Path, validator: Path, mode: str) -> int:
    failures = 0
    for rec in records:
        session_file = out_dir / "sessions" / f"{rec['session_id']}.json"
        cmd = [sys.executable, str(validator), str(session_file), "--mode", mode]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            failures += 1
            print(f"[validate] FAILED: {session_file}")
            if proc.stdout.strip():
                print(proc.stdout.strip())
            if proc.stderr.strip():
                print(proc.stderr.strip())
        else:
            print(f"[validate] OK: {session_file}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-discover audio/textgrid/diarization/transcript/prompt files from fixed folders "
            "and convert to OneVoice records."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("reference_toolkit"), help="Toolkit root folder")
    parser.add_argument(
        "--speaker-details",
        type=Path,
        default=Path("reference_toolkit/templates/speaker_details.csv"),
        help="Single speaker details list CSV",
    )
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path("reference_toolkit/templates/dataset_metadata.json"),
        help="Single dataset metadata JSON",
    )
    parser.add_argument("--out", type=Path, default=Path("reference_toolkit/output"), help="Output folder")
    parser.add_argument("--max-files", type=int, default=0, help="If >0, limit number of discovered audio files")
    parser.add_argument("--validate", action="store_true", help="Run validate_onevoice.py on output sessions")
    parser.add_argument("--validator", type=Path, default=Path("validate_onevoice.py"), help="Validator script path")
    parser.add_argument("--validation-mode", choices=("structure", "full"), default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = args.root.resolve()
    speaker_csv = args.speaker_details.resolve()
    metadata_json = args.dataset_metadata.resolve()
    out_dir = args.out.resolve()

    try:
        records = build_records(root, speaker_csv, metadata_json, max_files=args.max_files)
    except Exception as exc:
        print(f"error: conversion failed: {exc}", file=sys.stderr)
        return 1

    if not records:
        print("error: no session records produced (check input/audio folder and templates)", file=sys.stderr)
        return 1

    written = write_outputs(records, out_dir)
    print(f"Wrote {len(records)} session record(s).")
    for p in written:
        print(f" - {p}")

    if args.validate:
        validator = args.validator.resolve()
        if not validator.exists():
            print(f"error: validator not found: {validator}", file=sys.stderr)
            return 1
        failures = run_validation(records, out_dir, validator, args.validation_mode)
        if failures > 0:
            print(f"Validation finished with {failures} failure(s).", file=sys.stderr)
            return 2
        print("Validation successful for all generated records.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
