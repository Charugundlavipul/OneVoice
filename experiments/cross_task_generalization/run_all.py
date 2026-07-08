from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = ["gpt-5.4-mini", "gpt-5-mini"]
CONDITIONS = ["c0", "c1", "c2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all cross-task model experiments.")
    parser.add_argument("--execute", action="store_true", help="Actually launch model runs. Without this flag, only prints commands.")
    parser.add_argument("--dry-run", action="store_true", help="Create run artifacts without API calls.")
    parser.add_argument("--models", type=str, default=",".join(MODELS))
    parser.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    commands: list[list[str]] = []
    for model in models:
        for condition in conditions:
            for task in ("task2_childes", "task3_timit"):
                cmd = [sys.executable, f"experiments/cross_task_generalization/{task}/run.py", "--condition", condition, "--model", model]
                if args.dry_run:
                    cmd.append("--dry-run")
                if args.max_records is not None:
                    cmd.extend(["--max-records", str(args.max_records)])
                commands.append(cmd)

    for cmd in commands:
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)

    if not args.execute:
        print("No runs launched. Add --execute to run these commands.")


if __name__ == "__main__":
    main()
