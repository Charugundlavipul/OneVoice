#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.uxssd_setup.runner_core import (
    RunnerConfig,
    add_common_args,
    parse_record_ids,
    run_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1 (OneVoice-agnostic) with 3-agent flow.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/uxssd_setup/experiment1_agnostic_manifest.jsonl"),
        help="Manifest JSONL for Experiment 1.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/uxssd_setup/runs"),
        help="Directory where run artifacts are stored.",
    )
    parser.add_argument(
        "--max-file-chars",
        type=int,
        default=3200,
        help="Per-text-file character cap when building LLM input payload.",
    )
    parser.add_argument(
        "--max-files-per-utt",
        type=int,
        default=8,
        help="Max number of text files per utterance included in payload.",
    )
    parser.add_argument(
        "--max-utterances-per-record",
        type=int,
        default=None,
        help="Optional cap on utterances included per record payload.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = RunnerConfig(
        mode="exp1",
        manifest=args.manifest,
        output_root=args.output_root,
        model=args.model,
        temperature=args.temperature,
        max_records=args.max_records,
        record_ids=parse_record_ids(args.record_ids),
        max_file_chars=args.max_file_chars,
        max_files_per_utt=args.max_files_per_utt,
        max_utterances_per_record=args.max_utterances_per_record,
        dry_run=args.dry_run,
        max_output_tokens=args.max_output_tokens,
    )
    run_dir = run_experiment(cfg)
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
