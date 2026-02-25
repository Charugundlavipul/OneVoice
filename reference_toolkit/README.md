# OneVoice Reference Toolkit

This reference toolkit is folder-driven and low-friction.

You only maintain:

1. `templates/speaker_details.csv` (one speaker list)
2. `templates/dataset_metadata.json` (one dataset metadata file)

No per-utterance path CSV is needed.
The converter auto-discovers files from fixed folders and matches by session key
(audio file stem), following the same pattern used in your Ultrasuite conversions.

## Included Reference Sessions

This toolkit now includes two side-by-side examples in the same reference:

1. `demo_session_001` (small/simple)
- single speaker
- one RTTM segment
- short transcript/prompt/CHA/TextGrid

2. `demo_session_002` (complex)
- multiple speakers (`SPK_001`, `SPK_002`, `SPK_003`)
- multiple RTTM turns across three speakers
- multi-turn transcript and prompt files
- multi-speaker CHAT file with dependent tier
- richer TextGrid word/phone alignments

## Required Folder Layout

- `input/audio/` : audio files (root or nested folders)
- `input/diarization/` : diarization files (`.rttm`)
- `input/textgrid/` : TextGrid files (`.TextGrid` / `.textgrid`)
- `input/transcripts/` : transcript text files (`.txt`)
- `input/prompts/` : prompt/reference text files (`.txt`)
- `input/cha/` : CHAT files (`.cha`)

## Session-Key Matching Rule

- For each audio file, `session_id = <audio_stem>`.
- Other files are auto-matched by the same stem.
  Example:
  `demo_session_001.wav` matches `demo_session_001.rttm`, `demo_session_001.TextGrid`, `demo_session_001.txt`, `demo_session_001.cha`.
  `demo_session_002.wav` matches `demo_session_002.rttm`, `demo_session_002.TextGrid`, `demo_session_002.txt`, `demo_session_002.cha`.
- Audio subfolder names are treated only as optional organization hints, not speaker identity.

## What Converter Auto-Builds

- `audio_format` from file extension
- `sample_rate` + `session_duration` from WAV header when possible
- turns from RTTM segments
- fallback transcript segmentation from CHA when RTTM is missing
- transcript/prompt layers from text files
- word/phone alignments from TextGrid tiers
- coverage/source metadata inference when set to `auto`

## Run

```powershell
python reference_toolkit/convert_to_onevoice.py `
  --root reference_toolkit `
  --speaker-details reference_toolkit/templates/speaker_details.csv `
  --dataset-metadata reference_toolkit/templates/dataset_metadata.json `
  --out reference_toolkit/output `
  --validate `
  --validator validate_onevoice.py
```

## Output

- `output/sessions/<session_id>.json`
- `output/all.jsonl`
