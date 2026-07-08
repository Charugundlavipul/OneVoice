from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cross_task_generalization.task2_childes.build_manifest import build_manifest as build_task2_manifest
from experiments.cross_task_generalization.task3_timit.build_manifest import build_manifest as build_task3_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cross-task manifests.")
    parser.add_argument("--childes-root", type=Path, default=Path("CHILDES/raw/childes"))
    parser.add_argument("--timit-root", type=Path, default=Path("TIMIT"))
    parser.add_argument("--task2-target-windows", type=int, default=50)
    parser.add_argument("--task3-target-bundles", type=int, default=40)
    args = parser.parse_args()

    task2_manifest = Path("experiments/cross_task_generalization/manifests/task2_childes_manifest.jsonl")
    task3_manifest = Path("experiments/cross_task_generalization/manifests/task3_timit_manifest.jsonl")
    task3_thresholds = Path("experiments/cross_task_generalization/manifests/task3_timit_thresholds.json")

    rows2 = build_task2_manifest(args.childes_root, task2_manifest, args.task2_target_windows, 20, 30, 40)
    rows3 = build_task3_manifest(args.timit_root, task3_manifest, task3_thresholds, args.task3_target_bundles, "TEST", "TRAIN")

    # Reuse the adapter code through its CLI-shaped entry point only when called directly would parse args.
    print(f"Task 2: {len(rows2)} manifest windows")
    print(f"Task 3: {len(rows3)} manifest bundles")
    print(f"Task 3: thresholds written to {task3_thresholds}")
    print("Task 1: legacy summary can be exported with:")
    print("  python experiments/cross_task_generalization/task1_uxssd/adapter.py")


if __name__ == "__main__":
    main()
