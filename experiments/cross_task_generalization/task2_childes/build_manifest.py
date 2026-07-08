from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.common.io_utils import rel, write_jsonl
from experiments.cross_task_generalization.task2_childes.parse_chat import parse_chat_file, window_to_chat_text


def event_total(window: list[dict[str, Any]]) -> int:
    return sum(len(u.get("events", []) or []) for u in window)


def iter_cha_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.cha") if p.is_file()], key=lambda p: p.as_posix().lower())


def build_manifest(
    source_root: Path,
    out_path: Path,
    target_windows: int,
    min_files: int,
    max_files: int,
    window_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used_files = 0

    for cha in iter_cha_files(source_root):
        parsed = parse_chat_file(cha)
        utterances = parsed["utterances"]
        speakers = sorted({u["speaker_id"] for u in utterances})
        if len(speakers) < 2 or "CHI" not in speakers or len(utterances) < window_size:
            continue

        file_rows: list[dict[str, Any]] = []
        starts = list(range(0, max(1, len(utterances) - window_size + 1), window_size))
        for start in starts[:2]:
            window = utterances[start : start + window_size]
            if len({u["speaker_id"] for u in window}) < 2:
                continue
            if event_total(window) == 0:
                continue
            local_window = []
            for local_idx, utt in enumerate(window):
                copy = dict(utt)
                copy["source_utt_index"] = utt["utt_index"]
                copy["utt_index"] = local_idx
                local_window.append(copy)
            window_id = f"childes_{cha.stem}_{start:04d}_{start + len(window) - 1:04d}"
            file_rows.append(
                {
                    "task": "task2_childes",
                    "window_id": window_id,
                    "dataset": "CHILDES",
                    "source_file": rel(cha),
                    "source_start_utt_index": start,
                    "utterance_count": len(local_window),
                    "speakers": list(parsed["participants"].values()),
                    "event_count_hint": event_total(local_window),
                    "raw_chat": window_to_chat_text(local_window),
                    "utterances": local_window,
                }
            )
        if not file_rows:
            continue
        rows.extend(file_rows)
        used_files += 1
        if len(rows) >= target_windows and used_files >= min_files:
            break
        if used_files >= max_files and len(rows) >= target_windows:
            break

    rows = rows[:target_windows]
    write_jsonl(out_path, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task 2 CHILDES manifest.")
    parser.add_argument("--source-root", type=Path, default=Path("CHILDES/raw/childes"))
    parser.add_argument("--out", type=Path, default=Path("experiments/cross_task_generalization/manifests/task2_childes_manifest.jsonl"))
    parser.add_argument("--target-windows", type=int, default=50)
    parser.add_argument("--min-files", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=40)
    args = parser.parse_args()
    rows = build_manifest(args.source_root, args.out, args.target_windows, args.min_files, args.max_files, args.window_size)
    print(f"Wrote {len(rows)} windows to {args.out}")


if __name__ == "__main__":
    main()
