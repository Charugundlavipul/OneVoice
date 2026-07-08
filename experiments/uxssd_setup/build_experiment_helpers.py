#!/usr/bin/env python3
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "uxssd_setup"

JSONL_PATH = ROOT / "ultrasuite" / "output" / "uxssd" / "all_utterances.jsonl"
SCORES_CSV = (
    ROOT
    / "ultrasuite"
    / "uxssd_additional"
    / "uxssd"
    / "pronunciation_scores"
    / "uxssd-pronunciation-scores.csv"
)
SPEAKER_LABELS_DIR = ROOT / "ultrasuite" / "uxssd_additional" / "uxssd" / "speaker_labels" / "TG"


def parse_duration_s(ts: str) -> float:
    parts = str(ts).split(":")
    if len(parts) != 4:
        return 0.0
    try:
        h, m, s, ms = map(int, parts)
        return h * 3600 + m * 60 + s + ms / 1000.0
    except Exception:
        return 0.0


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load_utterances() -> dict:
    utterances = {}
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            utt = d.get("session_id", "")
            if not utt:
                continue

            speakers = d.get("speakers", [])
            speaker_id = ""
            if isinstance(speakers, list) and speakers and isinstance(speakers[0], dict):
                speaker_id = str(speakers[0].get("speaker_id", ""))

            audio_rel = str(d.get("audio_file_path", ""))
            audio_parts = Path(audio_rel).parts
            session = audio_parts[-2] if len(audio_parts) >= 2 else ""

            duration_s = parse_duration_s(str(d.get("session_duration", "")))
            turns = d.get("turns", [])
            if duration_s <= 0 and isinstance(turns, list) and turns and isinstance(turns[0], dict):
                duration_s = parse_duration_s(str(turns[0].get("end", "")))

            if not speaker_id and "-" in utt:
                speaker_id = utt.split("-", 1)[0] + "_CHILD"
            speaker_short = speaker_id.replace("_CHILD", "") if speaker_id else (utt.split("-", 1)[0] if "-" in utt else "")

            core_audio = ROOT / "ultrasuite" / audio_rel
            core_txt = core_audio.with_suffix(".txt")
            core_param = core_audio.with_suffix(".param")

            add_base = ROOT / "ultrasuite" / "uxssd_additional" / "uxssd"
            utt_tg = f"{utt}.TextGrid"
            paths = {
                "audio_wav": rel(core_audio) if core_audio.exists() else "",
                "transcript_txt": rel(core_txt) if core_txt.exists() else "",
                "param_file": rel(core_param) if core_param.exists() else "",
                "onevoice_json": rel(ROOT / "ultrasuite" / "output" / "uxssd" / speaker_short / f"{utt}.json")
                if (ROOT / "ultrasuite" / "output" / "uxssd" / speaker_short / f"{utt}.json").exists()
                else "",
            }

            utterances[utt] = {
                "utt": utt,
                "speaker_id": speaker_id,
                "speaker_short": speaker_short,
                "session": session,
                "duration_s": duration_s,
                "paths": paths,
            }
    return utterances


def load_mispronunciation_proxy() -> tuple[dict, dict]:
    # Per-utterance target-wise score aggregation across annotators.
    target_scores = defaultdict(lambda: defaultdict(list))
    with SCORES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            utt = str(row.get("utt", "")).strip()
            if not utt:
                continue
            try:
                primary = float(str(row.get("primary_score", "")).strip())
            except Exception:
                continue
            key = (
                str(row.get("phone_class", "")).strip(),
                str(row.get("phone", "")).strip(),
                str(row.get("word", "")).strip(),
                str(row.get("wordIdx", "")).strip(),
            )
            target_scores[utt][key].append(primary)

    utt_proxy = {}
    utt_targets = {}
    for utt, target_map in target_scores.items():
        bad_targets = []
        for key, vals in target_map.items():
            if not vals:
                continue
            med = float(median(vals))
            if med <= 3.0:
                bad_targets.append({"phone_class": key[0], "phone": key[1], "word": key[2], "word_idx": key[3], "median_primary_score": med})
        utt_proxy[utt] = len(bad_targets)
        utt_targets[utt] = bad_targets
    return utt_proxy, utt_targets


def load_behavioral_proxy(utterances: dict) -> tuple[dict, dict]:
    """
    Behavioral proxy from speaker labels TextGrid:
    - +1 for each SLT interval
    - +1 for each non-empty speaker-label transition
    - +1 for each non-(CHILD/SLT) label
    """
    proxy = {}
    details = {}
    triplet_re = re.compile(r'\n\s*([0-9]+(?:\.[0-9]+)?)\s*\n\s*([0-9]+(?:\.[0-9]+)?)\s*\n\s*"(.*?)"', re.S)

    for utt in utterances.keys():
        p = SPEAKER_LABELS_DIR / f"{utt}.TextGrid"
        if not p.exists():
            proxy[utt] = 0
            details[utt] = {"slt_intervals": 0, "speaker_transitions": 0, "other_speaker_labels": 0, "non_empty_labels": 0}
            continue

        txt = p.read_text(encoding="utf-8", errors="replace")
        triples = triplet_re.findall("\n" + txt)
        non_empty = []
        slt_intervals = 0
        other_labels = 0

        for _, _, label in triples:
            label = label.strip()
            if not label:
                continue
            non_empty.append(label)
            u = label.upper()
            if u == "SLT":
                slt_intervals += 1
            elif u != "CHILD":
                other_labels += 1

        transitions = 0
        for i in range(1, len(non_empty)):
            if non_empty[i] != non_empty[i - 1]:
                transitions += 1

        score = slt_intervals + transitions + other_labels
        proxy[utt] = score
        details[utt] = {
            "slt_intervals": slt_intervals,
            "speaker_transitions": transitions,
            "other_speaker_labels": other_labels,
            "non_empty_labels": len(non_empty),
        }

    return proxy, details


def build_bundles(utterances: dict, mis_proxy: dict, beh_proxy: dict) -> list:
    by_group = defaultdict(list)
    for _, meta in utterances.items():
        if meta["duration_s"] <= 0:
            continue
        group = f"{meta['speaker_short']}-{meta['session']}"
        by_group[group].append(meta)

    for group in by_group:
        by_group[group].sort(key=lambda x: x["utt"])

    bundles = []
    for group, rows in by_group.items():
        speaker_short = rows[0]["speaker_short"] if rows else ""
        speaker_id = rows[0]["speaker_id"] if rows else ""
        idx = 1
        cur = []
        cur_dur = 0.0
        cur_mis = 0
        cur_beh = 0

        for row in rows:
            next_dur = cur_dur + row["duration_s"]
            if cur and next_dur > 185:
                if 120 <= cur_dur <= 185 and len(cur) >= 8:
                    bundles.append(
                        {
                            "bundle_id": f"{speaker_short}_{group}_b{idx:02d}",
                            "speaker_id": speaker_id,
                            "speaker_short": speaker_short,
                            "session_group": group,
                            "duration_s": round(cur_dur, 3),
                            "mispronunciation_proxy_events": int(cur_mis),
                            "behavioral_event_proxy_events": int(cur_beh),
                            "utterance_count": len(cur),
                            "utterances": list(cur),
                        }
                    )
                    idx += 1
                cur = []
                cur_dur = 0.0
                cur_mis = 0
                cur_beh = 0

            cur.append(row)
            cur_dur += row["duration_s"]
            cur_mis += int(mis_proxy.get(row["utt"], 0))
            cur_beh += int(beh_proxy.get(row["utt"], 0))

        if 120 <= cur_dur <= 185 and len(cur) >= 8:
            bundles.append(
                {
                    "bundle_id": f"{speaker_short}_{group}_b{idx:02d}",
                    "speaker_id": speaker_id,
                    "speaker_short": speaker_short,
                    "session_group": group,
                    "duration_s": round(cur_dur, 3),
                    "mispronunciation_proxy_events": int(cur_mis),
                    "behavioral_event_proxy_events": int(cur_beh),
                    "utterance_count": len(cur),
                    "utterances": list(cur),
                }
            )

    return bundles


def _density(events: int, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return 60.0 * float(events) / float(duration_s)


def select_split_10_10(bundles: list) -> list:
    behavioral_pool = [b for b in bundles if b["behavioral_event_proxy_events"] >= 6]
    mispron_pool = [b for b in bundles if b["mispronunciation_proxy_events"] >= 6]

    behavioral_pool.sort(
        key=lambda b: (
            -_density(b["behavioral_event_proxy_events"], b["duration_s"]),
            _density(b["mispronunciation_proxy_events"], b["duration_s"]),
            abs(b["duration_s"] - 150.0),
            -b["utterance_count"],
        )
    )
    mispron_pool.sort(
        key=lambda b: (
            -_density(b["mispronunciation_proxy_events"], b["duration_s"]),
            _density(b["behavioral_event_proxy_events"], b["duration_s"]),
            abs(b["duration_s"] - 150.0),
            -b["utterance_count"],
        )
    )

    selected = []
    selected_ids = set()

    # Behavioral-heavy: strongest behavioral density first.
    for b in behavioral_pool:
        if len([x for x in selected if x["cohort"] == "behavioral_heavy"]) >= 10:
            break
        if b["bundle_id"] in selected_ids:
            continue
        x = dict(b)
        x["cohort"] = "behavioral_heavy"
        selected.append(x)
        selected_ids.add(x["bundle_id"])

    # Mispronunciation-heavy:
    # prefer strong mispronunciation density while avoiding very behavior-dense bundles.
    mis_focus = [b for b in mispron_pool if _density(b["behavioral_event_proxy_events"], b["duration_s"]) <= 25.0]
    for pool in (mis_focus, mispron_pool):
        for b in pool:
            if len([x for x in selected if x["cohort"] == "mispronunciation_heavy"]) >= 10:
                break
            if b["bundle_id"] in selected_ids:
                continue
            x = dict(b)
            x["cohort"] = "mispronunciation_heavy"
            selected.append(x)
            selected_ids.add(x["bundle_id"])
        if len([x for x in selected if x["cohort"] == "mispronunciation_heavy"]) >= 10:
            break

    return selected


def write_outputs(selected: list, mis_proxy: dict, mis_targets: dict, beh_proxy: dict, beh_details: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_csv = OUT_DIR / "selected_bundles_20.csv"
    utt_csv = OUT_DIR / "selected_utterances_20.csv"
    exp1_jsonl = OUT_DIR / "experiment1_agnostic_manifest.jsonl"
    exp2_jsonl = OUT_DIR / "experiment2_onevoice_manifest.jsonl"
    mis_proxy_csv = OUT_DIR / "mispronunciation_proxy_targets.csv"
    beh_proxy_csv = OUT_DIR / "behavioral_proxy_targets.csv"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "bundle_id",
                "cohort",
                "speaker_id",
                "session_group",
                "duration_s",
                "utterance_count",
                "mispronunciation_proxy_events",
                "behavioral_event_proxy_events",
                "note",
            ]
        )
        for b in selected:
            wr.writerow(
                [
                    b["bundle_id"],
                    b["cohort"],
                    b["speaker_id"],
                    b["session_group"],
                    f"{b['duration_s']:.3f}",
                    b["utterance_count"],
                    b["mispronunciation_proxy_events"],
                    b["behavioral_event_proxy_events"],
                    "behavioral proxy from speaker-label transitions/SLT activity; manual annotation still required",
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
    utterances = load_utterances()
    mis_proxy, mis_targets = load_mispronunciation_proxy()
    beh_proxy, beh_details = load_behavioral_proxy(utterances)
    bundles = build_bundles(utterances, mis_proxy, beh_proxy)
    selected = select_split_10_10(bundles)
    write_outputs(selected, mis_proxy, mis_targets, beh_proxy, beh_details)

    n_beh = sum(1 for b in selected if b["cohort"] == "behavioral_heavy")
    n_mis = sum(1 for b in selected if b["cohort"] == "mispronunciation_heavy")
    print(f"Generated manifests in: {rel(OUT_DIR)}")
    print(f"Candidate bundles: {len(bundles)}")
    print(f"Selected bundles: {len(selected)} (behavioral_heavy={n_beh}, mispronunciation_heavy={n_mis})")


if __name__ == "__main__":
    main()
