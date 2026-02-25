Put audio files here (e.g., `.wav`, `.flac`, `.mp3`).

No CSV path mapping is needed.
Files are auto-discovered recursively.
Folder names under `input/audio/` are optional and do not define speaker identity.

Session key rule:
- `session_id` is inferred from audio filename stem.
  Example: `demo_session_001.wav` -> `session_id = demo_session_001`.
  Example: `demo_session_002.wav` -> `session_id = demo_session_002`.
