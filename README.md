# OneVoice

## What OneVoice Is
OneVoice is a unified intermediate representation (IR) for speech-processing pipelines.  
It gives all tools and agents one shared JSON format for a session so data can move between stages without losing structure.

Core goals:

- keep stable identifiers across artifacts
- preserve multiple transcript layers in one place
- attach optional word/phone alignments when available
- keep annotation provenance and coverage explicit
- support validator-driven quality checks before downstream use

## Repository Structure
- `reference_toolkit/`: folder-driven conversion pipeline that builds OneVoice JSON from audio + annotations
- `experiments/uxssd_setup/`: experimental setup comparing agnostic handoff vs OneVoice handoff
- `json_structure.json`: concise field map of the OneVoice record
- `validate_onevoice.py`: schema/consistency validator
- `.env.example`: environment template for OpenAI-based experiment runs

## OneVoice Toolkit
The toolkit is designed to be low-friction: you drop files into fixed folders and convert them into session records.

Reference conversion command:

```powershell
python reference_toolkit/convert_to_onevoice.py `
  --root reference_toolkit `
  --speaker-details reference_toolkit/templates/speaker_details.csv `
  --dataset-metadata reference_toolkit/templates/dataset_metadata.json `
  --out reference_toolkit/output `
  --validate `
  --validator validate_onevoice.py
```

Output:

- `reference_toolkit/output/sessions/<session_id>.json`
- `reference_toolkit/output/all.jsonl`

See `reference_toolkit/README.md` for full folder layout and matching rules.

## OneVoice JSON Structure
The record format is documented in `json_structure.json`.

Top-level fields:

- `session_id`, `session_date`
- `audio_file_path`, `audio_format`, `sample_rate`, `session_duration`
- `speakers[]`
- `turns[]`
- `metadata`

Per-turn design:

- transcript layers: `reference_transcript`, `orthographic_transcript`, `phonetic_transcript`, `ipa_transcript`, `raw_transcript`, `phoneme_reference`
- timing: `start`, `end`
- alignments: `word_alignments[].phonemes[]`
- optional event annotations: `mispronunciations[]`, `behavioral_events[]`

Metadata design:

- dataset identity: `dataset_name`, `dataset_split`
- text provenance/coverage
- phoneme provenance/coverage

This allows the same schema to represent fully aligned sessions and partially annotated sessions without ambiguity.

## Validator
`validate_onevoice.py` enforces structural and temporal consistency and catches common integration errors.

Checks include:

- schema structure and unknown-key detection
- time format and interval validity (`start < end`)
- containment (`phoneme` inside `word`, `word` inside `turn`)
- mandatory fields (for example `session_id`, `turn_index`, `utt_id`, `speaker_id`)
- content presence (at least audio path or non-empty transcript content)

Run:

```powershell
python validate_onevoice.py path/to/file.json --mode structure
python validate_onevoice.py path/to/file.json --mode full
```

## Experiment Results
The UXSSD experiment compares:

- `Exp1`: OneVoice-agnostic multi-agent extraction
- `Exp2`: OneVoice JSON handoff with validator-in-loop repair

Reported setup:

- 15 gold bundles, 5 speakers
- total gold events: 206 (`94` mispronunciations, `112` behavioral)
- model: `gpt-5-mini`

Count-based proxy results:

| Event Type | System | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Mispronunciation | Exp1 | 1.000 | 0.128 | 0.226 |
| Mispronunciation | Exp2 | 0.864 | 0.809 | 0.835 |
| Behavioral | Exp1 | 0.595 | 0.393 | 0.473 |
| Behavioral | Exp2 | 0.978 | 0.795 | 0.877 |
| Overall | Exp1 | 0.651 | 0.272 | 0.384 |
| Overall | Exp2 | 0.922 | 0.801 | 0.857 |

Total event-count MAE:

- Exp1: `10.00`
- Exp2: `3.00`

Summary:

- overall F1 improves from `0.384` to `0.857` (`+0.473`)
- MAE drops by `70.0%`
- validator-mediated OneVoice handoff is substantially more stable than agnostic handoff

## Environment
For experiment scripts that call OpenAI:

1. Copy `.env.example` to `.env`
2. Set `OPENAI_API_KEY` in your shell

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "your_openai_api_key_here"
```
