#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_template(record: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    utterances_out = []
    for utt in record.get("utterances", []):
        files = utt.get("files", {})
        utterances_out.append(
            {
                "utt": utt.get("utt", ""),
                "speaker_id": utt.get("speaker_id", ""),
                "duration_s": float(utt.get("duration_s", 0.0) or 0.0),
                "audio_wav": files.get("audio_wav", ""),
                "transcript_txt": files.get("transcript_txt", ""),
                "mispronunciations": [],
                "behavioral_events": [],
                "review": {"is_reviewed": False, "comments": ""},
            }
        )

    return {
        "schema_version": "1.0",
        "record_id": record.get("record_id", ""),
        "annotator": {"annotator_id": "", "annotation_status": "not_started", "notes": ""},
        "timebase": "utterance_local",
        "source": {
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "summary": {
            "bundle_duration_s": float(record.get("bundle_duration_s", 0.0) or 0.0),
            "annotation_min_total_events": 6,
        },
        "utterances": utterances_out,
        "qa": {
            "all_events_have_type": False,
            "all_events_have_valid_time_range": False,
            "all_events_within_utterance": False,
            "mispronunciations_have_target_or_observed_phone": False,
            "no_duplicate_events": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold empty per-record annotation files from an experiment manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/uxssd_setup/experiment1_agnostic_manifest.jsonl"),
        help="JSONL manifest to scaffold annotation files from.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/uxssd_setup/gold_labels/templates"),
        help="Directory for scaffolded annotation files.",
    )
    parser.add_argument(
        "--record-ids",
        type=str,
        default="",
        help="Comma-separated subset of record IDs. Default: all records in manifest.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.manifest)
    selected_ids = {x.strip() for x in args.record_ids.split(",") if x.strip()}
    if selected_ids:
        records = [r for r in records if str(r.get("record_id", "")) in selected_ids]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []

    for record in records:
        template = build_template(record, args.manifest)
        record_id = str(template["record_id"])
        out_path = args.out_dir / f"{record_id}.gold.json"
        out_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append(
            {
                "record_id": record_id,
                "file": str(out_path).replace("\\", "/"),
                "bundle_duration_s": template["summary"]["bundle_duration_s"],
                "utterance_count": len(template["utterances"]),
            }
        )

    index_path = args.out_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Scaffolded {len(index)} annotation files at {args.out_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
