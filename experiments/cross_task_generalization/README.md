# Cross-Task Generalization Experiments

This folder is the new rebuttal-stage experiment setup. The legacy UXSSD experiment has been archived at:

```text
experiments/legacy/uxssd_setup
```

The new setup evaluates the same three-condition design across tasks:

- `c0`: implicit agent-defined handoff
- `c1`: OneVoice intermediate representation without validator repair
- `c2`: OneVoice intermediate representation with validator repair

The actual model runs are not launched by setup scripts unless explicitly requested.

## Layout

```text
experiments/cross_task_generalization/
  common/              shared IO, metrics, validators, OpenAI runner, run utilities
  configs/             documented experiment settings
  schemas/             final output JSON schemas
  task1_uxssd/         adapter for archived UXSSD results
  task2_childes/       CHILDES parser, manifest builder, prompts, runner, evaluator
  task3_timit/         TIMIT parser, manifest builder, prompts, runner, evaluator
  manifests/           generated manifest JSONL files
  gold/                gold label files and TIMIT calibration thresholds
  runs/                run artifacts
  reports/             setup validation and summary reports
```

## Prepare Data

Build Task 2 and Task 3 manifests:

```bash
python experiments/cross_task_generalization/prepare_setup.py
```

Export the archived Task 1 UXSSD summary:

```bash
python experiments/cross_task_generalization/task1_uxssd/adapter.py
```

Validate setup without API calls:

```bash
python experiments/cross_task_generalization/validate_setup.py
```

The Task 2 runner defaults to the enhanced CHILDES manifest set.

## Dry Runs

Create run artifacts without calling OpenAI:

```bash
python experiments/cross_task_generalization/run_all.py --dry-run --execute --max-records 1
```

## Model Runs

Print the full command list without launching:

```bash
python experiments/cross_task_generalization/run_all.py
```

Launch selected real runs:

```bash
python experiments/cross_task_generalization/task2_childes/run.py --condition c2 --model gpt-5-mini
python experiments/cross_task_generalization/task3_timit/run.py --condition c2 --model gpt-5-mini
```

Launch all configured Task 2 and Task 3 runs:

```bash
python experiments/cross_task_generalization/run_all.py --execute
```

## Evaluation

Evaluate a Task 2 run:

```bash
python experiments/cross_task_generalization/task2_childes/evaluate.py --pred-run experiments/cross_task_generalization/runs/<task2_run_dir>
```

Evaluate a Task 3 run:

```bash
python experiments/cross_task_generalization/task3_timit/evaluate.py --pred-run experiments/cross_task_generalization/runs/<task3_run_dir>
```

Summarize metric files:

```bash
python experiments/cross_task_generalization/summarize_results.py
```

## Rebuttal-Stage Design Choices

- TIMIT thresholds are computed from a deterministic held-out `TRAIN` calibration subset and evaluated on `TEST`; thresholds are not fit on evaluation bundles.
- Task 2 uses only reliable CHAT markers in the selected files: speaker tiers, repetition/repair markers, pause markers, non-speech markers, and overlap markers.
- Task 3 reports event counts plus link and boundary metrics, not only aggregate count proxies.
- C2 records validator logs and repair rounds for OneVoice intermediates.

## Expected Paper Tables

Use the generated metric CSV/JSON files to fill:

- Task 2 table: speaker match, event count F1, event count MAE, event attribution
- Task 3 table: event count F1, event count MAE, word-link accuracy, invalid link rate, boundary error
- Validator diagnostic table: initial pass, final pass, mean repair attempts, remaining invalid

Task 1 UXSSD remains an archived legacy result unless you decide to rerun it with the archived scripts.
