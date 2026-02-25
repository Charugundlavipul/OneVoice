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
    parser = argparse.ArgumentParser(description="Run Experiment 2 (OneVoice + validator-in-loop) with 3-agent flow.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/uxssd_setup/experiment2_onevoice_manifest.jsonl"),
        help="Manifest JSONL for Experiment 2.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/uxssd_setup/runs"),
        help="Directory where run artifacts are stored.",
    )
    parser.add_argument(
        "--validator-path",
        type=Path,
        default=Path("validate_onevoice.py"),
        help="Path to OneVoice validator script.",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=3,
        help="Max validator-driven repair rounds per agent.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = RunnerConfig(
        mode="exp2",
        manifest=args.manifest,
        output_root=args.output_root,
        model=args.model,
        temperature=args.temperature,
        max_records=args.max_records,
        record_ids=parse_record_ids(args.record_ids),
        max_repair_rounds=args.max_repair_rounds,
        validator_path=args.validator_path,
        dry_run=args.dry_run,
        max_output_tokens=args.max_output_tokens,
    )
    run_dir = run_experiment(cfg)
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
