from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import load_json, load_jsonl, write_json
from experiments.cross_task_generalization.common.openai_runner import JsonModelClient
from experiments.cross_task_generalization.common.claude_runner import ClaudeJsonModelClient


def make_model_client(model: str, **kwargs) -> JsonModelClient | ClaudeJsonModelClient:
    """Factory: pick OpenAI or Claude client based on model name."""
    if "claude" in model.lower() or "haiku" in model.lower():
        return ClaudeJsonModelClient(model, **kwargs)
    return JsonModelClient(model, **kwargs)
from experiments.cross_task_generalization.common.run_utils import (
    ONEVOICE_REPAIR_PROMPT,
    chunks,
    merge_onevoice_chunks,
    onevoice_from_timit,
    onevoice_with_turn_subset,
    repair_with_validator,
    stamp,
    task3_empty_final,
    write_summary_csv,
)
from experiments.cross_task_generalization.common.validators import validate_task3_final
from experiments.cross_task_generalization.task3_timit import prompts


def payload_from_row(row: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": row["bundle_id"],
        "speaker_id": row["speaker_id"],
        "thresholds": thresholds,
        "utterances": row.get("utterances", []),
    }


def project_timit_events(onevoice: dict[str, Any], events_res: dict[str, Any], allowed_types: set[str], default_type: str) -> dict[str, Any]:
    events = events_res.get("events", [])
    if not isinstance(events, list):
        events = []
    
    from experiments.cross_task_generalization.common.time_utils import seconds_to_ts
    for item in events:
        if not isinstance(item, dict):
            continue
        utt_id = item.get("utt_id")
        event_type = str(item.get("event_type") or "").strip().lower()
        if event_type not in allowed_types:
            event_type = default_type
        
        start_val = item.get("start")
        end_val = item.get("end")
        
        try:
            start_ts = seconds_to_ts(float(start_val)) if start_val is not None else ""
            end_ts = seconds_to_ts(float(end_val)) if end_val is not None else ""
        except Exception:
            start_ts = ""
            end_ts = ""
            
        for turn in onevoice.get("turns", []):
            if turn.get("utt_id") == utt_id:
                ev_obj = {
                    "type": event_type,
                    "start": start_ts,
                    "end": end_ts,
                }
                if "word_index" in item:
                    ev_obj["word_index"] = item["word_index"]
                if "phone_index" in item:
                    ev_obj["phone_index"] = item["phone_index"]
                turn.setdefault("behavioral_events", []).append(ev_obj)
    return onevoice


def normalize_utt_event(item: dict[str, Any], allowed_types: set[str], default_type: str) -> dict[str, Any] | None:
    utt_id = item.get("utt_id")
    if not isinstance(utt_id, str) or not utt_id:
        return None
    event_type = str(item.get("event_type") or "").strip().lower()
    if event_type not in allowed_types:
        event_type = default_type
    try:
        start = float(item.get("start"))
        end = float(item.get("end"))
    except Exception:
        return None
    if end < start:
        start, end = end, start
    out: dict[str, Any] = {"utt_id": utt_id, "event_type": event_type, "start": start, "end": end}
    for key in ("word_index", "phone_index"):
        if item.get(key) is None:
            continue
        try:
            out[key] = int(item[key])
        except Exception:
            pass
    return out


def turn_subset_for_utterances(base_onevoice: dict[str, Any], utterances: list[dict[str, Any]]) -> dict[str, Any]:
    wanted = {str(u.get("utt_id")) for u in utterances}
    turns = [dict(t) for t in base_onevoice.get("turns", []) if str(t.get("utt_id")) in wanted]
    return onevoice_with_turn_subset(base_onevoice, turns)


def task3_final_from_events(row: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    valid_utt_ids = {u["utt_id"] for u in row.get("utterances", [])}
    phone_to_word: dict[tuple[str, int], int] = {}
    for utt in row.get("utterances", []):
        utt_id = utt["utt_id"]
        for phone in utt.get("phone_intervals", []) or []:
            try:
                phone_to_word[(utt_id, int(phone["phone_index"]))] = int(phone["word_index"])
            except Exception:
                continue
    filtered = []
    for event in events:
        if event.get("utt_id") not in valid_utt_ids:
            continue
        if event.get("word_index") is None and event.get("phone_index") is not None:
            try:
                inferred_word_index = phone_to_word.get((str(event["utt_id"]), int(event["phone_index"])))
            except Exception:
                inferred_word_index = None
            if inferred_word_index is not None:
                event = dict(event)
                event["word_index"] = inferred_word_index
        filtered.append(event)
    filtered.sort(
        key=lambda e: (
            e["utt_id"],
            float(e.get("start", 0.0)),
            float(e.get("end", 0.0)),
            e["event_type"],
            int(e.get("phone_index", -1)),
            int(e.get("word_index", -1)),
        )
    )
    counts = Counter((e["utt_id"], e["event_type"]) for e in filtered)
    summaries = []
    for utt in row.get("utterances", []):
        utt_id = utt["utt_id"]
        summary = {
            "utt_id": utt_id,
            "long_word_count": counts[(utt_id, "long_word")],
            "short_word_count": counts[(utt_id, "short_word")],
            "inter_word_gap_count": counts[(utt_id, "inter_word_gap")],
            "boundary_gap_count": counts[(utt_id, "boundary_gap")],
            "long_phone_count": counts[(utt_id, "long_phone")],
            "short_phone_count": counts[(utt_id, "short_phone")],
            "closure_count": counts[(utt_id, "closure_segment")],
            "pause_silence_count": counts[(utt_id, "pause_silence_segment")],
            "glottal_stop_count": counts[(utt_id, "glottal_stop")],
        }
        summary["total_event_count"] = sum(v for k, v in summary.items() if k.endswith("_count") and k != "total_event_count")
        summaries.append(summary)
    unlinked = [
        {
            "utt_id": e["utt_id"],
            "event_type": e["event_type"],
            "reason": "no_parent_word",
            **({"phone_index": e["phone_index"]} if "phone_index" in e else {}),
        }
        for e in filtered
        if e["event_type"] in {"long_phone", "short_phone", "closure_segment", "pause_silence_segment", "glottal_stop"}
        and e.get("word_index") is None
    ]
    return {
        "bundle_id": row["bundle_id"],
        "utterance_summaries": summaries,
        "events": filtered,
        "unlinked_events": unlinked,
    }


def run_chunked_c2(
    row: dict[str, Any],
    thresholds: dict[str, Any],
    record_dir: Path,
    model_client: JsonModelClient | ClaudeJsonModelClient,
    max_repair_rounds: int,
    chunk_size: int,
    parallel_chunks: int,
) -> dict[str, Any]:
    base_onevoice = onevoice_from_timit(row)
    utterance_chunks = chunks(row.get("utterances", []), chunk_size)
    word_types = {"long_word", "short_word", "inter_word_gap", "boundary_gap"}
    phone_types = {"long_phone", "short_phone", "closure_segment", "pause_silence_segment", "glottal_stop"}

    def process_chunk(chunk_index: int, utterances: list[dict[str, Any]]) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
        chunk_dir = record_dir / "chunks" / f"chunk_{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle_id": row["bundle_id"],
            "speaker_id": row["speaker_id"],
            "chunk_index": chunk_index,
            "utterances": utterances,
            "thresholds": thresholds,
        }
        write_json(chunk_dir / "input_payload.json", payload)
        onevoice = turn_subset_for_utterances(base_onevoice, utterances)

        a_res, a_raw = model_client.call_json(
            prompts.TASK3_C1C2_AGENT_A,
            {"bundle_id": row["bundle_id"], "word_intervals_input": utterances, "thresholds": thresholds},
        )
        write_json(chunk_dir / "agentA_output_flat.json", a_res)
        (chunk_dir / "agentA_raw.txt").write_text(a_raw, encoding="utf-8")
        word_events = [
            event
            for item in (a_res.get("events", []) if isinstance(a_res.get("events"), list) else [])
            if isinstance(item, dict)
            for event in [normalize_utt_event(item, word_types, "long_word")]
            if event is not None
        ]
        onevoice = project_timit_events(onevoice, {"events": word_events}, word_types, "long_word")

        b_res, b_raw = model_client.call_json(
            prompts.TASK3_C1C2_AGENT_B,
            {
                "bundle_id": row["bundle_id"],
                "phone_intervals_input": utterances,
                "word_events": word_events,
                "thresholds": thresholds,
            },
        )
        write_json(chunk_dir / "agentB_output_flat.json", b_res)
        (chunk_dir / "agentB_raw.txt").write_text(b_raw, encoding="utf-8")
        phone_events = [
            event
            for item in (b_res.get("events", []) if isinstance(b_res.get("events"), list) else [])
            if isinstance(item, dict)
            for event in [normalize_utt_event(item, phone_types, "long_phone")]
            if event is not None
        ]
        onevoice = project_timit_events(onevoice, {"events": phone_events}, phone_types, "long_phone")

        write_json(chunk_dir / "onevoice_before_validation.json", onevoice)
        onevoice, ok, rounds = repair_with_validator(
            onevoice,
            chunk_dir,
            "onevoice",
            model_client.call_json,
            ONEVOICE_REPAIR_PROMPT,
            max_repair_rounds,
        )
        write_json(chunk_dir / "onevoice_validated.json", onevoice)
        write_json(chunk_dir / "validation_summary.json", {"ok": ok, "repair_rounds": rounds})
        return chunk_index, onevoice, word_events + phone_events

    chunk_docs: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    if parallel_chunks > 1 and len(utterance_chunks) > 1:
        with ThreadPoolExecutor(max_workers=parallel_chunks) as executor:
            futures = {executor.submit(process_chunk, idx, utterances): idx for idx, utterances in enumerate(utterance_chunks)}
            for future in as_completed(futures):
                _idx, doc, events = future.result()
                chunk_docs.append(doc)
                all_events.extend(events)
    else:
        for idx, utterances in enumerate(utterance_chunks):
            _idx, doc, events = process_chunk(idx, utterances)
            chunk_docs.append(doc)
            all_events.extend(events)

    merged = merge_onevoice_chunks(base_onevoice, chunk_docs)
    write_json(record_dir / "agentB_output.json", merged)
    return task3_final_from_events(row, all_events)


def run_row(row: dict[str, Any], thresholds: dict[str, Any], record_dir: Path, condition: str, model_client: JsonModelClient | None, max_repair_rounds: int) -> tuple[bool, str]:
    payload = payload_from_row(row, thresholds)
    write_json(record_dir / "input_payload.json", payload)
    if condition in {"c1", "c2"}:
        write_json(record_dir / "input_onevoice.json", onevoice_from_timit(row))

    if model_client is None:
        final = task3_empty_final(row["bundle_id"], row.get("utterances", []))
        write_json(record_dir / "final_output.json", final)
        return True, "dry_run"

    if condition == "c0":
        a, a_raw = model_client.call_json(prompts.TASK3_C0_AGENT_A, payload)
        b, b_raw = model_client.call_json(prompts.TASK3_C0_AGENT_B, payload)
        write_json(record_dir / "agentA_output.json", a)
        write_json(record_dir / "agentB_output.json", b)
        (record_dir / "agentA_raw.txt").write_text(a_raw, encoding="utf-8")
        (record_dir / "agentB_raw.txt").write_text(b_raw, encoding="utf-8")
        final, c_raw = model_client.call_json(prompts.TASK3_AGENT_C_FINAL, {"bundle_id": row["bundle_id"], "input": payload, "agentA_output": a, "agentB_output": b})
        (record_dir / "agentC_raw.txt").write_text(c_raw, encoding="utf-8")
    elif condition == "c1":
        onevoice = onevoice_from_timit(row)
        
        # Word events
        word_types = {"long_word", "short_word", "inter_word_gap", "boundary_gap"}
        phone_types = {"long_phone", "short_phone", "closure_segment", "pause_silence_segment", "glottal_stop"}
        
        # Call simplified Agent A
        a_res, a_raw = model_client.call_json(prompts.TASK3_C1C2_AGENT_A, {"bundle_id": row["bundle_id"], "word_intervals_input": row.get("utterances", []), "thresholds": thresholds})
        write_json(record_dir / "agentA_output_flat.json", a_res)
        (record_dir / "agentA_raw.txt").write_text(a_raw, encoding="utf-8")
        word_events = [
            event
            for item in (a_res.get("events", []) if isinstance(a_res.get("events"), list) else [])
            if isinstance(item, dict)
            for event in [normalize_utt_event(item, word_types, "long_word")]
            if event is not None
        ]
        
        # Project Agent A results into OneVoice structure
        onevoice = project_timit_events(onevoice, {"events": word_events}, word_types, "long_word")
        write_json(record_dir / "agentA_output.json", onevoice)
        
        # Call simplified Agent B
        b_res, b_raw = model_client.call_json(prompts.TASK3_C1C2_AGENT_B, {"bundle_id": row["bundle_id"], "phone_intervals_input": row.get("utterances", []), "word_events": word_events, "thresholds": thresholds})
        write_json(record_dir / "agentB_output_flat.json", b_res)
        (record_dir / "agentB_raw.txt").write_text(b_raw, encoding="utf-8")
        phone_events = [
            event
            for item in (b_res.get("events", []) if isinstance(b_res.get("events"), list) else [])
            if isinstance(item, dict)
            for event in [normalize_utt_event(item, phone_types, "long_phone")]
            if event is not None
        ]
        
        # Project Agent B results into OneVoice structure
        onevoice = project_timit_events(onevoice, {"events": phone_events}, phone_types, "long_phone")
        write_json(record_dir / "agentB_output.json", onevoice)
        
        final = task3_final_from_events(row, word_events + phone_events)
    elif condition == "c2":
        final = run_chunked_c2(
            row,
            thresholds,
            record_dir,
            model_client,
            max_repair_rounds,
            getattr(run_row, "chunk_size", 1),
            getattr(run_row, "parallel_chunks", 1),
        )
    else:
        raise ValueError(f"Unsupported condition: {condition}")

    write_json(record_dir / "final_output.json", final)
    errors = validate_task3_final(final)
    if errors:
        write_json(record_dir / "final_validation_errors.json", errors)
        return False, "invalid_final"
    return True, "ok"



def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 3 TIMIT experiment.")
    parser.add_argument("--manifest", type=Path, default=Path("experiments/cross_task_generalization/manifests/task3_timit_manifest.jsonl"))
    parser.add_argument("--thresholds", type=Path, default=Path("experiments/cross_task_generalization/manifests/task3_timit_thresholds.json"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/cross_task_generalization/runs"))
    parser.add_argument("--condition", choices=["c0", "c1", "c2"], required=True)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--parallel-records", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--parallel-chunks", type=int, default=1)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    thresholds = load_json(args.thresholds)
    if args.start_index:
        rows = rows[args.start_index :]
    if args.max_records is not None:
        rows = rows[: args.max_records]
    run_dir = args.run_dir or (args.output_root / f"task3_{args.condition}_{args.model}_{stamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["record_count"] = len(rows)
    write_json(run_dir / "run_config.json", cfg)
    run_row.chunk_size = args.chunk_size
    run_row.parallel_chunks = args.parallel_chunks

    def process_row(idx: int, row: dict[str, Any]) -> dict[str, Any]:
        record_dir = run_dir / row["bundle_id"]
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(rows)}] {row['bundle_id']}")
        if args.skip_existing and (record_dir / "final_output.json").exists():
            return {"record_id": row["bundle_id"], "ok": True, "status": "skipped_existing"}
        model_client = None if args.dry_run else make_model_client(args.model, max_output_tokens=args.max_output_tokens)
        try:
            ok, status = run_row(row, thresholds, record_dir, args.condition, model_client, args.max_repair_rounds)
        except Exception as exc:
            ok, status = False, f"exception: {exc}"
            (record_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        return {"record_id": row["bundle_id"], "ok": ok, "status": status}

    indexed_rows = list(enumerate(rows, start=1))
    if args.parallel_records > 1:
        summary = []
        with ThreadPoolExecutor(max_workers=args.parallel_records) as executor:
            futures = [executor.submit(process_row, idx, row) for idx, row in indexed_rows]
            for future in as_completed(futures):
                summary.append(future.result())
        summary.sort(key=lambda item: item["record_id"])
    else:
        summary = [process_row(idx, row) for idx, row in indexed_rows]

    write_json(run_dir / "summary.json", summary)
    write_summary_csv(run_dir / "summary.csv", summary)
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
