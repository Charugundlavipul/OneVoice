# UXSSD Experiment Setup

This directory is the default UXSSD experiment setup used in the paper.

Goal:
- keep per-record proxy event counts in a controlled range:
  - behavioral proxy in `[5, 10]`
  - mispronunciation proxy in `[5, 10]`
- keep a mixed profile of records:
  - `mispronunciation_heavy` (mis > beh)
  - `behavioral_heavy` (beh > mis)
  - optional `balanced` (mis == beh)
- build `15` records by default.

## Build Manifests And CSVs

```bash
python experiments/uxssd_setup/build_experiment_manifests.py
```

Outputs:
- `experiments/uxssd_setup/selected_bundles.csv`
- `experiments/uxssd_setup/selected_utterances.csv`
- `experiments/uxssd_setup/experiment1_agnostic_manifest.jsonl`
- `experiments/uxssd_setup/experiment2_onevoice_manifest.jsonl`
- `experiments/uxssd_setup/mispronunciation_proxy_targets.csv`
- `experiments/uxssd_setup/behavioral_proxy_targets.csv`

Optional range overrides:

```bash
python experiments/uxssd_setup/build_experiment_manifests.py \
  --target-records 15 \
  --min-beh 5 --max-beh 10 \
  --min-mis 5 --max-mis 10 \
  --max-overlap-ratio 0.55 \
  --max-records-per-speaker 4
```

## Human Annotation Templates

Scaffold empty annotation templates:

```bash
python experiments/uxssd_setup/gold_labels/scaffold_annotation_templates.py \
  --manifest experiments/uxssd_setup/experiment1_agnostic_manifest.jsonl \
  --out-dir experiments/uxssd_setup/gold_labels/templates
```

Validate:

```bash
python experiments/uxssd_setup/gold_labels/validate_gold_labels.py \
  experiments/uxssd_setup/gold_labels/templates
```

## Run Experiments

Environment:

1. Copy `.env.example` to `.env` at the repository root.
2. Set your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Experiment 1 (agnostic):

```bash
python experiments/uxssd_setup/run_experiment1_agnostic.py
```

Experiment 2 (OneVoice + validator):

```bash
python experiments/uxssd_setup/run_experiment2_onevoice.py
```

Default model:
- `gpt-5-mini`

Run outputs:
- `experiments/uxssd_setup/runs/`

## Refresh Comparison CSV

```bash
python experiments/uxssd_setup/refresh_counts_csv.py
```

Output:
- `experiments/uxssd_setup/runs/event_counts_gold_exp1_exp2.csv`
