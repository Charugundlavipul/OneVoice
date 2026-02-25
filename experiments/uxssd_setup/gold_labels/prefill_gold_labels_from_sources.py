#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[3]
DEFAULT_TEMPLATE_DIR = ROOT / "experiments" / "uxssd_setup" / "gold_labels" / "templates"
DEFAULT_UTT_CSV = ROOT / "experiments" / "uxssd_setup" / "selected_utterances_20.csv"
DEFAULT_MIS_PROXY_CSV = ROOT / "experiments" / "uxssd_setup" / "mispronunciation_proxy_targets.csv"

TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}):(\d{3})$")
NUM_LINE_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)$")
QUOTED_LINE_RE = re.compile(r'^"(.*)"$')
XMIN_LINE_RE = re.compile(r"^\s*xmin\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$", re.IGNORECASE)
XMAX_LINE_RE = re.compile(r"^\s*xmax\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$", re.IGNORECASE)
TEXT_LINE_RE = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$', re.IGNORECASE)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts_to_seconds(ts: Any) -> float | None:
    if not isinstance(ts, str):
        return None
    m = TS_RE.match(ts.strip())
    if not m:
        return None
    hh, mm, ss, ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def to_hhmmss_mmm(seconds: float) -> str:
    ms_total = max(0, int(round(float(seconds) * 1000.0)))
    hh = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    mm = rem // 60_000
    rem = rem % 60_000
    ss = rem // 1000
    ms = rem % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ms:03d}"


def clamp_interval(start_s: float, end_s: float, duration_s: float) -> tuple[float, float] | None:
    s = max(0.0, min(float(start_s), float(duration_s)))
    e = max(0.0, min(float(end_s), float(duration_s)))
    if e <= s:
        e = min(float(duration_s), s + 0.001)
    if e <= s:
        return None
    return s, e


def normalize_word(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9']+", "", str(text or "")).upper()


def parse_textgrid_intervals(path: Path) -> list[tuple[float, float, str]]:
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    lines = txt.split("\n")

    # Long TextGrid format (xmin/xmax/text).
    long_out: list[tuple[float, float, str]] = []
    for i in range(0, len(lines) - 2):
        m1 = XMIN_LINE_RE.match(lines[i])
        m2 = XMAX_LINE_RE.match(lines[i + 1])
        m3 = TEXT_LINE_RE.match(lines[i + 2])
        if not (m1 and m2 and m3):
            continue
        s = float(m1.group(1))
        e = float(m2.group(1))
        if e <= s:
            continue
        long_out.append((s, e, m3.group(1)))
    if long_out:
        long_out.sort(key=lambda x: (x[0], x[1], x[2]))
        return long_out

    # Short TextGrid format (num / num / "label").
    short_out: list[tuple[float, float, str]] = []
    for i in range(0, len(lines) - 2):
        a = lines[i].strip()
        b = lines[i + 1].strip()
        c = lines[i + 2].strip()
        if not NUM_LINE_RE.match(a):
            continue
        if not NUM_LINE_RE.match(b):
            continue
        m3 = QUOTED_LINE_RE.match(c)
        if not m3:
            continue
        s = float(a)
        e = float(b)
        if e <= s:
            continue
        short_out.append((s, e, m3.group(1)))
    short_out.sort(key=lambda x: (x[0], x[1], x[2]))
    return short_out


def load_selected_utterances(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            utt = str(row.get("utt", "")).strip()
            if utt:
                out[utt] = row
    return out


def load_mis_target_details(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            utt = str(row.get("utt", "")).strip()
            if not utt:
                continue
            raw = str(row.get("target_details_json", "") or "[]")
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = []
            if isinstance(parsed, list):
                out[utt] = [x for x in parsed if isinstance(x, dict)]
            else:
                out[utt] = []
    return out


def parse_positive_int(v: Any, default: int = 1) -> int:
    try:
        x = int(float(str(v).strip()))
    except Exception:
        return default
    return x if x > 0 else default


def get_intervals_for_relpath(
    relpath: str,
    cache: dict[str, list[tuple[float, float, str]]],
) -> list[tuple[float, float, str]]:
    rp = str(relpath or "").strip()
    if not rp:
        return []
    if rp in cache:
        return cache[rp]
    abs_path = ROOT / rp
    intervals = parse_textgrid_intervals(abs_path)
    cache[rp] = intervals
    return intervals


def pick_word_interval(
    intervals: list[tuple[float, float, str]],
    target_word: str,
    target_word_idx: Any,
) -> tuple[tuple[float, float, str] | None, int]:
    non_empty = [x for x in intervals if str(x[2]).strip()]
    idx = parse_positive_int(target_word_idx, default=1)
    want = normalize_word(target_word)

    exact = [x for x in non_empty if normalize_word(x[2]) == want]
    if exact:
        if 1 <= idx <= len(exact):
            return exact[idx - 1], idx
        return exact[0], 1

    if non_empty:
        if 1 <= idx <= len(non_empty):
            return non_empty[idx - 1], idx
        return non_empty[0], 1

    return None, idx


def dedupe_events(
    utt: str,
    events: list[dict[str, Any]],
    event_kind: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ev in events:
        key = json.dumps(
            [
                event_kind,
                utt,
                str(ev.get("type", "")),
                str(ev.get("start", "")),
                str(ev.get("end", "")),
            ],
            ensure_ascii=False,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def build_mispronunciations_for_utt(
    utt: str,
    utt_row: dict[str, str] | None,
    duration_s: float,
    target_details: list[dict[str, Any]],
    tg_cache: dict[str, list[tuple[float, float, str]]],
) -> tuple[list[dict[str, Any]], int]:
    if not target_details:
        return [], 0
    if not utt_row:
        return [], len(target_details)

    ref_word_tg = str(utt_row.get("reference_word_textgrid", "")).strip()
    word_tg = str(utt_row.get("word_labels_textgrid", "")).strip()
    candidate_sources = []
    if ref_word_tg:
        candidate_sources.append(("reference_word_textgrid", ref_word_tg))
    if word_tg and word_tg != ref_word_tg:
        candidate_sources.append(("word_labels_textgrid", word_tg))

    events: list[dict[str, Any]] = []
    missing = 0

    for item in target_details:
        word = str(item.get("word", "")).strip()
        word_idx_raw = item.get("word_idx", "")
        phone = str(item.get("phone", "")).strip()
        phone_class = str(item.get("phone_class", "")).strip()
        score = item.get("median_primary_score", "")

        chosen: tuple[float, float, str] | None = None
        chosen_word_idx = parse_positive_int(word_idx_raw, default=1)
        chosen_src = ""

        for src_name, relpath in candidate_sources:
            intervals = get_intervals_for_relpath(relpath, tg_cache)
            interval, widx = pick_word_interval(intervals, word, word_idx_raw)
            if interval is None:
                continue
            chosen = interval
            chosen_word_idx = widx
            chosen_src = src_name
            break

        if chosen is None:
            missing += 1
            continue

        clamped = clamp_interval(chosen[0], chosen[1], duration_s)
        if clamped is None:
            missing += 1
            continue

        target_phone = phone or phone_class or "unknown"
        notes = f"auto_prefill source={chosen_src}; word={word}; phone_class={phone_class}; median_primary_score={score}"

        events.append(
            {
                "type": "mispronunciation",
                "target_phone": target_phone,
                "word_index": chosen_word_idx,
                "start": to_hhmmss_mmm(clamped[0]),
                "end": to_hhmmss_mmm(clamped[1]),
                "notes": notes,
            }
        )

    events = dedupe_events(utt=utt, events=events, event_kind="mis")
    return events, missing


def build_behavioral_events_for_utt(
    utt: str,
    utt_row: dict[str, str] | None,
    duration_s: float,
    tg_cache: dict[str, list[tuple[float, float, str]]],
) -> tuple[list[dict[str, Any]], int]:
    if not utt_row:
        return [], 1

    speaker_tg = str(utt_row.get("speaker_labels_textgrid", "")).strip()
    if not speaker_tg:
        return [], 1

    intervals = get_intervals_for_relpath(speaker_tg, tg_cache)
    if not intervals:
        return [], 1

    events: list[dict[str, Any]] = []
    non_empty: list[tuple[float, float, str]] = []

    for s, e, raw_label in intervals:
        label = str(raw_label or "").strip()
        if not label:
            continue
        clamped = clamp_interval(s, e, duration_s)
        if clamped is None:
            continue

        non_empty.append((clamped[0], clamped[1], label))
        up = label.upper()

        if up == "SLT":
            events.append(
                {
                    "type": "slt_intervention",
                    "start": to_hhmmss_mmm(clamped[0]),
                    "end": to_hhmmss_mmm(clamped[1]),
                    "notes": "auto_prefill source=speaker_labels_textgrid label=SLT",
                }
            )
        elif up != "CHILD":
            events.append(
                {
                    "type": "other_speaker_activity",
                    "start": to_hhmmss_mmm(clamped[0]),
                    "end": to_hhmmss_mmm(clamped[1]),
                    "notes": f"auto_prefill source=speaker_labels_textgrid label={label}",
                }
            )

    # Proxy-compatible transition events among non-empty labels.
    for i in range(1, len(non_empty)):
        prev = non_empty[i - 1]
        curr = non_empty[i]
        prev_label = prev[2]
        curr_label = curr[2]
        if prev_label == curr_label:
            continue
        boundary = curr[0]
        clamped = clamp_interval(boundary - 0.01, boundary + 0.01, duration_s)
        if clamped is None:
            continue
        events.append(
            {
                "type": "speaker_transition",
                "start": to_hhmmss_mmm(clamped[0]),
                "end": to_hhmmss_mmm(clamped[1]),
                "notes": f"auto_prefill source=speaker_labels_textgrid transition={prev_label}->{curr_label}",
            }
        )

    events = dedupe_events(utt=utt, events=events, event_kind="beh")
    return events, 0


def compute_qa_flags(doc: dict[str, Any]) -> dict[str, bool]:
    all_have_type = True
    all_valid_time = True
    within_utt = True
    mis_have_target_or_observed = True
    no_dupes = True
    seen: set[str] = set()

    utterances = doc.get("utterances", [])
    if not isinstance(utterances, list):
        utterances = []

    for utt_obj in utterances:
        if not isinstance(utt_obj, dict):
            continue
        utt = str(utt_obj.get("utt", "")).strip()
        duration_s = float(utt_obj.get("duration_s", 0.0) or 0.0)

        mis = utt_obj.get("mispronunciations", [])
        beh = utt_obj.get("behavioral_events", [])
        if not isinstance(mis, list):
            mis = []
        if not isinstance(beh, list):
            beh = []

        for ev in mis:
            if not isinstance(ev, dict):
                continue
            etype = str(ev.get("type", "")).strip()
            if not etype:
                all_have_type = False
            s = parse_ts_to_seconds(ev.get("start"))
            e = parse_ts_to_seconds(ev.get("end"))
            if s is None or e is None or e <= s:
                all_valid_time = False
            else:
                if s < -1e-9 or e > duration_s + 1e-9:
                    within_utt = False
            if not str(ev.get("target_phone", "")).strip() and not str(ev.get("observed_phone", "")).strip():
                mis_have_target_or_observed = False

            sig = json.dumps(["mis", utt, etype, ev.get("start", ""), ev.get("end", "")], ensure_ascii=False)
            if sig in seen:
                no_dupes = False
            else:
                seen.add(sig)

        for ev in beh:
            if not isinstance(ev, dict):
                continue
            etype = str(ev.get("type", "")).strip()
            if not etype:
                all_have_type = False
            s = parse_ts_to_seconds(ev.get("start"))
            e = parse_ts_to_seconds(ev.get("end"))
            if s is None or e is None or e <= s:
                all_valid_time = False
            else:
                if s < -1e-9 or e > duration_s + 1e-9:
                    within_utt = False

            sig = json.dumps(["beh", utt, etype, ev.get("start", ""), ev.get("end", "")], ensure_ascii=False)
            if sig in seen:
                no_dupes = False
            else:
                seen.add(sig)

    return {
        "all_events_have_type": all_have_type,
        "all_events_have_valid_time_range": all_valid_time,
        "all_events_within_utterance": within_utt,
        "mispronunciations_have_target_or_observed_phone": mis_have_target_or_observed,
        "no_duplicate_events": no_dupes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-prefill UXSSD gold label templates from source TextGrids and proxy target CSVs."
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        help="Directory containing *.gold.json files.",
    )
    parser.add_argument(
        "--selected-utterances-csv",
        type=Path,
        default=DEFAULT_UTT_CSV,
        help="selected_utterances_20.csv path.",
    )
    parser.add_argument(
        "--mis-target-csv",
        type=Path,
        default=DEFAULT_MIS_PROXY_CSV,
        help="mispronunciation_proxy_targets.csv path.",
    )
    parser.add_argument(
        "--record-ids",
        type=str,
        default="",
        help="Optional comma-separated record_id subset.",
    )
    args = parser.parse_args()

    if not args.template_dir.exists():
        raise SystemExit(f"Template dir not found: {args.template_dir}")
    if not args.selected_utterances_csv.exists():
        raise SystemExit(f"selected_utterances CSV not found: {args.selected_utterances_csv}")
    if not args.mis_target_csv.exists():
        raise SystemExit(f"mis target CSV not found: {args.mis_target_csv}")

    selected_rows = load_selected_utterances(args.selected_utterances_csv)
    mis_targets = load_mis_target_details(args.mis_target_csv)

    chosen_record_ids = {x.strip() for x in args.record_ids.split(",") if x.strip()}
    gold_files = sorted(args.template_dir.glob("*.gold.json"))
    if chosen_record_ids:
        gold_files = [p for p in gold_files if p.stem.replace(".gold", "") in chosen_record_ids]

    tg_cache: dict[str, list[tuple[float, float, str]]] = {}
    summary_rows: list[dict[str, Any]] = []
    stamp = now_utc_iso()

    for path in gold_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        record_id = str(data.get("record_id", path.stem.replace(".gold", "")))
        utterances = data.get("utterances", [])
        if not isinstance(utterances, list):
            utterances = []

        record_mis = 0
        record_beh = 0
        missing_utt_rows = 0
        missing_mis_alignments = 0
        missing_speaker_labels = 0

        for utt_obj in utterances:
            if not isinstance(utt_obj, dict):
                continue
            utt = str(utt_obj.get("utt", "")).strip()
            duration_s = float(utt_obj.get("duration_s", 0.0) or 0.0)
            row = selected_rows.get(utt)
            if row is None:
                missing_utt_rows += 1

            mis_events, mis_missing = build_mispronunciations_for_utt(
                utt=utt,
                utt_row=row,
                duration_s=duration_s,
                target_details=mis_targets.get(utt, []),
                tg_cache=tg_cache,
            )
            beh_events, beh_missing = build_behavioral_events_for_utt(
                utt=utt,
                utt_row=row,
                duration_s=duration_s,
                tg_cache=tg_cache,
            )

            utt_obj["mispronunciations"] = mis_events
            utt_obj["behavioral_events"] = beh_events

            record_mis += len(mis_events)
            record_beh += len(beh_events)
            missing_mis_alignments += mis_missing
            missing_speaker_labels += beh_missing

        annotator = data.get("annotator")
        if not isinstance(annotator, dict):
            annotator = {}
        if not str(annotator.get("annotator_id", "")).strip():
            annotator["annotator_id"] = "auto_prefill_from_sources"
        annotator["annotation_status"] = "in_progress"
        note = f"Auto-prefilled from UXSSD source files on {stamp}"
        prev_notes = str(annotator.get("notes", "")).strip()
        annotator["notes"] = note if not prev_notes else f"{prev_notes} | {note}"
        data["annotator"] = annotator

        summary = data.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        summary["auto_prefill_event_counts"] = {
            "mispronunciations": record_mis,
            "behavioral_events": record_beh,
            "total_events": record_mis + record_beh,
        }
        data["summary"] = summary

        data["qa"] = compute_qa_flags(data)

        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_rows.append(
            {
                "record_id": record_id,
                "utterance_count": len(utterances),
                "mispronunciation_events": record_mis,
                "behavioral_events": record_beh,
                "total_events": record_mis + record_beh,
                "missing_selected_utterance_rows": missing_utt_rows,
                "missing_mispronunciation_alignments": missing_mis_alignments,
                "missing_speaker_label_files": missing_speaker_labels,
            }
        )

    summary_csv = args.template_dir / "auto_prefill_counts.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "record_id",
            "utterance_count",
            "mispronunciation_events",
            "behavioral_events",
            "total_events",
            "missing_selected_utterance_rows",
            "missing_mispronunciation_alignments",
            "missing_speaker_label_files",
        ]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in summary_rows:
            wr.writerow(row)

    summary_json = args.template_dir / "auto_prefill_counts.json"
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    total_mis = sum(int(r["mispronunciation_events"]) for r in summary_rows)
    total_beh = sum(int(r["behavioral_events"]) for r in summary_rows)
    print(f"Prefilled {len(summary_rows)} gold files.")
    print(f"Total mispronunciation events: {total_mis}")
    print(f"Total behavioral events: {total_beh}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
