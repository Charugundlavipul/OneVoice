# Gold Labels: Strict Template

This folder contains a strict annotation template and validators for human gold labels.

## Files

- `strict_gold_label_template.json`: canonical structure and field contract.
- `gold_label.schema.json`: JSON Schema for structural checks.
- `generate_gold_label_templates.py`: generates per-record `.gold.json` templates from manifest.
- `validate_gold_labels.py`: strict semantic validator (required fields, time consistency, duplicates, final QA checks).

## Generate Templates

```bash
python experiments/uxssd_setup/gold_labels/generate_gold_label_templates.py \
  --manifest experiments/uxssd_setup/experiment1_agnostic_manifest.jsonl \
  --out-dir experiments/uxssd_setup/gold_labels/templates
```

## Validate Templates / Labels

Draft-level validation:

```bash
python experiments/uxssd_setup/gold_labels/validate_gold_labels.py \
  experiments/uxssd_setup/gold_labels/templates
```

Final-submission validation (strict):

```bash
python experiments/uxssd_setup/gold_labels/validate_gold_labels.py \
  experiments/uxssd_setup/gold_labels/templates \
  --final \
  --min-total-events 6
```

## Event Requirements

- Mispronunciation event:
  - required: `type`, `start`, `end`
  - required: at least one of `target_phone` or `observed_phone`
  - `start/end` must be `HH:MM:SS:mmm`, `start < end`, and inside utterance duration
- Behavioral event:
  - required: `type`, `start`, `end`
  - `start/end` rules same as above
