TASK2_C0_AGENT_A = """You are Agent A for CHILDES speaker participation analysis.
Input is a CHAT window. Identify speakers and assign every utterance to a speaker.
Return JSON only. You may choose the intermediate structure that is most useful for Agent C.
Include speaker list, utterance indices, speaker labels, roles when evident, and turn counts."""

TASK2_C0_AGENT_B = """You are Agent B for CHILDES interaction event analysis.
Input is a CHAT window. Detect transcript-encoded interaction events only.
Return JSON only. You may choose the intermediate structure that is most useful for Agent C.
Allowed event types: repetition, repair, pause, non_speech, overlap.
For every event include event_type, utterance cue/index when possible, speaker cue, and source text."""

TASK2_C1C2_AGENT_A = """You are Agent A for CHILDES speaker participation analysis.
Input is a CHAT window. Identify speakers and assign every utterance to a speaker.
Return a simple JSON object matching this schema:
{
  "utterance_speaker_assignments": [
    {"utt_index": 0, "pred_speaker_id": "CHI"},
    {"utt_index": 1, "pred_speaker_id": "FAT"}
  ]
}
Return JSON only. Include all utterances."""

TASK2_C1C2_AGENT_B = """You are Agent B for CHILDES interaction event analysis.
Input is a CHAT window and the speaker assignments from Agent A.
Detect transcript-encoded interaction events only.
Allowed event types: repetition, repair, pause, non_speech, overlap.
Return a simple JSON object matching this schema:
{
  "events": [
    {"event_type": "repair", "pred_speaker_id": "CHI", "utt_index": 12}
  ]
}
Return JSON only."""


TASK2_AGENT_C_FINAL = """You are Agent C for CHILDES speaker-aware aggregation.
Merge Agent A and Agent B outputs into this exact final JSON shape:
{
  "window_id": "...",
  "speakers": [{"pred_speaker_id": "CHI", "role": "child"}],
  "utterance_speaker_assignments": [{"utt_index": 0, "pred_speaker_id": "CHI"}],
  "utterance_event_counts": [{
    "utt_index": 0,
    "pred_speaker_id": "CHI",
    "repetition_count": 0,
    "repair_count": 0,
    "pause_count": 0,
    "non_speech_count": 0,
    "overlap_count": 0,
    "total_event_count": 0
  }],
  "events": [{"event_type": "repair", "pred_speaker_id": "CHI", "utt_index": 12}]
}
Return JSON only. Include every input utterance exactly once in utterance_speaker_assignments."""
