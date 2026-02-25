Put diarization files here (RTTM, `.rttm`).

No path CSV needed.
The converter auto-loads RTTM and creates turns.

Matching behavior:
- Uses RTTM `file_id` column to match `session_id` when available.
- Also uses RTTM filename stem as fallback.
- Examples: `demo_session_001.rttm`, `demo_session_002.rttm`.
