from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import load_jsonl, write_json
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
    onevoice_from_childes,
    repair_with_validator,
    stamp,
    task2_empty_final,
    onevoice_with_turn_subset,
    write_summary_csv,
)
from experiments.cross_task_generalization.common.validators import TASK2_EVENT_TYPES, task2_event_count_rows, validate_task2_final
from experiments.cross_task_generalization.task2_childes import prompts


def payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_id": row["window_id"],
        "source_file": row["source_file"],
        "raw_chat": row["raw_chat"],
        "utterances": [
            {
                "utt_index": u["utt_index"],
                "speaker_hint": u.get("speaker_id", ""),
                "raw": u.get("raw", ""),
                "dependent_tiers": u.get("dependent_tiers", {}),
            }
            for u in row.get("utterances", [])
        ],
    }


def project_speaker_assignments(onevoice: dict[str, Any], a: dict[str, Any], total_turns: int) -> dict[str, Any]:
    assignments = a.get("utterance_speaker_assignments", [])
    if not isinstance(assignments, list):
        assignments = []
    
    speakers_seen = set()
    for item in assignments:
        if not isinstance(item, dict):
            continue
        utt_index = item.get("utt_index")
        speaker_id = str(item.get("pred_speaker_id") or "UNK").strip()
        if not speaker_id:
            speaker_id = "UNK"
        if isinstance(utt_index, int) and 0 <= utt_index < total_turns:
            speakers_seen.add(speaker_id)
            for turn in onevoice.get("turns", []):
                if turn.get("turn_index") == utt_index:
                    turn["speaker_id"] = speaker_id
                    
    onevoice["speakers"] = [
        {
            "speaker_id": sid,
            "gender": "",
            "language": "en",
            "native_language": "english",
            "accent": "",
            "age": "",
            "name": "",
            "age_group": "child" if sid == "CHI" else "adult",
        }
        for sid in sorted(speakers_seen)
    ]
    return onevoice


def normalize_utt_index(value: Any, chunk_utts: list[dict[str, Any]]) -> int | None:
    original_indices = [int(u["utt_index"]) for u in chunk_utts if isinstance(u.get("utt_index"), int)]
    if not isinstance(value, int):
        return None
    if value in original_indices:
        return value
    if 0 <= value < len(original_indices):
        return original_indices[value]
    return None


def chunk_payload(row: dict[str, Any], chunk_utts: list[dict[str, Any]], chunk_index: int) -> dict[str, Any]:
    lines = []
    for utt in chunk_utts:
        lines.append(
            f"utt_index={utt.get('utt_index')} speaker_hint={utt.get('speaker_id', '')}: {utt.get('raw', '')}"
        )
        for tier, values in sorted((utt.get("dependent_tiers") or {}).items()):
            for value in values:
                lines.append(f"%{tier}: {value}")
    return {
        "window_id": row["window_id"],
        "chunk_index": chunk_index,
        "instruction": "Use the provided original utt_index values in every returned item.",
        "raw_chat": "\n".join(lines),
        "utterances": [
            {
                "utt_index": u.get("utt_index"),
                "speaker_hint": u.get("speaker_id", ""),
                "raw": u.get("raw", ""),
                "dependent_tiers": u.get("dependent_tiers", {}),
            }
            for u in chunk_utts
        ],
    }


def project_speaker_assignments_chunk(onevoice: dict[str, Any], a: dict[str, Any], chunk_utts: list[dict[str, Any]]) -> dict[str, Any]:
    valid_indices = {int(u["utt_index"]) for u in chunk_utts if isinstance(u.get("utt_index"), int)}
    turn_by_index = {int(t["turn_index"]): t for t in onevoice.get("turns", []) if isinstance(t.get("turn_index"), int)}
    speakers_seen = {str(s.get("speaker_id")) for s in onevoice.get("speakers", []) if isinstance(s, dict) and s.get("speaker_id")}
    assignments = a.get("utterance_speaker_assignments", [])
    if not isinstance(assignments, list):
        assignments = []
    for item in assignments:
        if not isinstance(item, dict):
            continue
        utt_index = normalize_utt_index(item.get("utt_index"), chunk_utts)
        if utt_index is None or utt_index not in valid_indices:
            continue
        speaker_id = str(item.get("pred_speaker_id") or "UNK").strip() or "UNK"
        if utt_index in turn_by_index:
            turn_by_index[utt_index]["speaker_id"] = speaker_id
            speakers_seen.add(speaker_id)
    onevoice["speakers"] = [
        {
            "speaker_id": sid,
            "gender": "",
            "language": "en",
            "native_language": "english",
            "accent": "",
            "age": "",
            "name": "",
            "age_group": "child" if sid == "CHI" else "adult",
        }
        for sid in sorted(speakers_seen)
    ]
    return onevoice


def project_events(onevoice: dict[str, Any], b: dict[str, Any], total_turns: int) -> dict[str, Any]:
    events = b.get("events", [])
    if not isinstance(events, list):
        events = []
    
    for turn in onevoice.get("turns", []):
        turn["behavioral_events"] = []
        
    for item in events:
        if not isinstance(item, dict):
            continue
        utt_index = item.get("utt_index")
        event_type = str(item.get("event_type") or "").strip().lower()
        if event_type not in TASK2_EVENT_TYPES:
            # surgical correction: default to repair
            event_type = "repair"
        if isinstance(utt_index, int) and 0 <= utt_index < total_turns:
            for turn in onevoice.get("turns", []):
                if turn.get("turn_index") == utt_index:
                    turn["behavioral_events"].append({
                        "type": event_type,
                        "start": "",
                        "end": ""
                    })
    return onevoice


def project_events_chunk(onevoice: dict[str, Any], b: dict[str, Any], chunk_utts: list[dict[str, Any]]) -> dict[str, Any]:
    valid_indices = {int(u["utt_index"]) for u in chunk_utts if isinstance(u.get("utt_index"), int)}
    turn_by_index = {int(t["turn_index"]): t for t in onevoice.get("turns", []) if isinstance(t.get("turn_index"), int)}
    for turn in onevoice.get("turns", []):
        turn["behavioral_events"] = []

    events = b.get("events", [])
    if not isinstance(events, list):
        events = []
    for item in events:
        if not isinstance(item, dict):
            continue
        utt_index = normalize_utt_index(item.get("utt_index"), chunk_utts)
        if utt_index is None or utt_index not in valid_indices or utt_index not in turn_by_index:
            continue
        event_type = str(item.get("event_type") or "").strip().lower()
        if event_type not in TASK2_EVENT_TYPES:
            continue
        turn_by_index[utt_index].setdefault("behavioral_events", []).append(
            {
                "type": event_type,
                "start": "",
                "end": "",
            }
        )
    return onevoice


def final_from_onevoice(row: dict[str, Any], onevoice: dict[str, Any]) -> dict[str, Any]:
    assignments = []
    events = []
    role_by_speaker = {str(s.get("speaker_id")): str(s.get("age_group") or "") for s in onevoice.get("speakers", []) if isinstance(s, dict)}
    for turn in sorted(onevoice.get("turns", []) or [], key=lambda t: int(t.get("turn_index", 0))):
        if not isinstance(turn, dict):
            continue
        try:
            idx = int(turn["turn_index"])
        except Exception:
            continue
        speaker_id = str(turn.get("speaker_id") or "UNK")
        assignments.append({"utt_index": idx, "pred_speaker_id": speaker_id})
        for event in turn.get("behavioral_events", []) or []:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "").strip().lower()
            if event_type in TASK2_EVENT_TYPES:
                events.append({"event_type": event_type, "pred_speaker_id": speaker_id, "utt_index": idx})
    speakers = [
        {
            "pred_speaker_id": sid,
            "role": "child" if role_by_speaker.get(sid) == "child" or sid == "CHI" else "adult",
        }
        for sid in sorted({a["pred_speaker_id"] for a in assignments})
    ]
    return {
        "window_id": row["window_id"],
        "speakers": speakers,
        "utterance_speaker_assignments": assignments,
        "utterance_event_counts": task2_event_count_rows(events, assignments),
        "events": events,
    }


def run_chunked_c2(
    row: dict[str, Any],
    record_dir: Path,
    model_client: JsonModelClient | ClaudeJsonModelClient,
    max_repair_rounds: int,
    chunk_size: int,
    parallel_chunks: int,
) -> dict[str, Any]:
    base_onevoice = onevoice_from_childes(row)
    utterance_chunks = chunks(row.get("utterances", []), chunk_size)

    def process_chunk(chunk_index: int, chunk_utts: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        chunk_dir = record_dir / "chunks" / f"chunk_{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        turn_indices = {int(u["utt_index"]) for u in chunk_utts if isinstance(u.get("utt_index"), int)}
        turns = [dict(t) for t in base_onevoice.get("turns", []) if int(t.get("turn_index", -1)) in turn_indices]
        onevoice = onevoice_with_turn_subset(base_onevoice, turns)
        payload = chunk_payload(row, chunk_utts, chunk_index)
        write_json(chunk_dir / "input_payload.json", payload)

        a_res, a_raw = model_client.call_json(prompts.TASK2_C1C2_AGENT_A, payload)
        write_json(chunk_dir / "agentA_output_flat.json", a_res)
        (chunk_dir / "agentA_raw.txt").write_text(a_raw, encoding="utf-8")
        onevoice = project_speaker_assignments_chunk(onevoice, a_res, chunk_utts)

        b_payload = {
            **payload,
            "utterance_speaker_assignments": a_res.get("utterance_speaker_assignments", []),
        }
        b_res, b_raw = model_client.call_json(prompts.TASK2_C1C2_AGENT_B, b_payload)
        write_json(chunk_dir / "agentB_output_flat.json", b_res)
        (chunk_dir / "agentB_raw.txt").write_text(b_raw, encoding="utf-8")
        onevoice = project_events_chunk(onevoice, b_res, chunk_utts)
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
        return chunk_index, onevoice

    chunk_docs: list[dict[str, Any]] = []
    if parallel_chunks > 1 and len(utterance_chunks) > 1:
        with ThreadPoolExecutor(max_workers=parallel_chunks) as executor:
            futures = {
                executor.submit(process_chunk, idx, chunk_utts): idx
                for idx, chunk_utts in enumerate(utterance_chunks)
            }
            for future in as_completed(futures):
                _idx, doc = future.result()
                chunk_docs.append(doc)
    else:
        for idx, chunk_utts in enumerate(utterance_chunks):
            _idx, doc = process_chunk(idx, chunk_utts)
            chunk_docs.append(doc)

    merged = merge_onevoice_chunks(base_onevoice, chunk_docs)
    write_json(record_dir / "agentB_output.json", merged)
    return final_from_onevoice(row, merged)


def run_row(row: dict[str, Any], record_dir: Path, condition: str, model_client: JsonModelClient | None, max_repair_rounds: int) -> tuple[bool, str]:
    payload = payload_from_row(row)
    write_json(record_dir / "input_payload.json", payload)
    if condition in {"c1", "c2"}:
        write_json(record_dir / "input_onevoice.json", onevoice_from_childes(row))

    if model_client is None:
        final = task2_empty_final(row["window_id"], row.get("utterances", []))
        write_json(record_dir / "final_output.json", final)
        return True, "dry_run"

    if condition == "c0":
        a, a_raw = model_client.call_json(prompts.TASK2_C0_AGENT_A, payload)
        b, b_raw = model_client.call_json(prompts.TASK2_C0_AGENT_B, payload)
        write_json(record_dir / "agentA_output.json", a)
        write_json(record_dir / "agentB_output.json", b)
        (record_dir / "agentA_raw.txt").write_text(a_raw, encoding="utf-8")
        (record_dir / "agentB_raw.txt").write_text(b_raw, encoding="utf-8")
        c_payload = {"window_id": row["window_id"], "input": payload, "agentA_output": a, "agentB_output": b}
        final, c_raw = model_client.call_json(prompts.TASK2_AGENT_C_FINAL, c_payload)
        (record_dir / "agentC_raw.txt").write_text(c_raw, encoding="utf-8")
    elif condition == "c1":
        onevoice = onevoice_from_childes(row)
        total_turns = len(row.get("utterances", []))
        
        # Call simplified Agent A
        a_res, a_raw = model_client.call_json(prompts.TASK2_C1C2_AGENT_A, {"window_id": row["window_id"], "raw_chat": row.get("raw_chat", "")})
        write_json(record_dir / "agentA_output_flat.json", a_res)
        (record_dir / "agentA_raw.txt").write_text(a_raw, encoding="utf-8")
        
        # Project Agent A results into OneVoice structure
        onevoice = project_speaker_assignments(onevoice, a_res, total_turns)
        write_json(record_dir / "agentA_output.json", onevoice)
        
        # Call simplified Agent B
        b_res, b_raw = model_client.call_json(prompts.TASK2_C1C2_AGENT_B, {"window_id": row["window_id"], "utterance_speaker_assignments": a_res.get("utterance_speaker_assignments", []), "raw_chat": row.get("raw_chat", "")})
        write_json(record_dir / "agentB_output_flat.json", b_res)
        (record_dir / "agentB_raw.txt").write_text(b_raw, encoding="utf-8")
        
        # Project Agent B results into OneVoice structure
        onevoice = project_events(onevoice, b_res, total_turns)
        write_json(record_dir / "agentB_output.json", onevoice)
        
        # Merge Agent C
        final, c_raw = model_client.call_json(prompts.TASK2_AGENT_C_FINAL, {"window_id": row["window_id"], "input": payload, "onevoice_json": onevoice})
        (record_dir / "agentC_raw.txt").write_text(c_raw, encoding="utf-8")
    elif condition == "c2":
        final = run_chunked_c2(
            row,
            record_dir,
            model_client,
            max_repair_rounds,
            getattr(run_row, "chunk_size", 10),
            getattr(run_row, "parallel_chunks", 1),
        )
    else:
        raise ValueError(f"Unsupported condition: {condition}")

    write_json(record_dir / "final_output.json", final)
    errors = validate_task2_final(final)
    if errors:
        write_json(record_dir / "final_validation_errors.json", errors)
        return False, "invalid_final"
    return True, "ok"



def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 2 CHILDES experiment.")
    parser.add_argument("--manifest", type=Path, default=Path("experiments/cross_task_generalization/manifests/task2_childes_manifest_enhanced.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/cross_task_generalization/runs"))
    parser.add_argument("--condition", choices=["c0", "c1", "c2"], required=True)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--parallel-chunks", type=int, default=1)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    if args.max_records is not None:
        rows = rows[: args.max_records]
    run_dir = args.output_root / f"task2_{args.condition}_{args.model}_{stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["record_count"] = len(rows)
    write_json(run_dir / "run_config.json", cfg)
    model_client = None if args.dry_run else make_model_client(args.model, max_output_tokens=args.max_output_tokens)
    run_row.chunk_size = args.chunk_size
    run_row.parallel_chunks = args.parallel_chunks

    summary = []
    for idx, row in enumerate(rows, start=1):
        record_dir = run_dir / row["window_id"]
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(rows)}] {row['window_id']}")
        try:
            ok, status = run_row(row, record_dir, args.condition, model_client, args.max_repair_rounds)
        except Exception as exc:
            ok, status = False, f"exception: {exc}"
            (record_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        summary.append({"record_id": row["window_id"], "ok": ok, "status": status})

    write_json(run_dir / "summary.json", summary)
    write_summary_csv(run_dir / "summary.csv", summary)
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
