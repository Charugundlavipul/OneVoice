TASK3_C0_AGENT_A = """You are Agent A for TIMIT word-timing analysis.
Input contains .WRD word intervals for five utterances from one speaker.
Detect word-level timing events: long_word, short_word, inter_word_gap, boundary_gap.
Return JSON only. You may choose the intermediate structure that is most useful for Agent C.
Include utterance ID, word index, word, start/end in seconds, event type, and duration."""

TASK3_C0_AGENT_B = """You are Agent B for TIMIT phone-event analysis.
Input contains .PHN phone intervals and .TXT transcript for five utterances.
Detect phone-level events: long_phone, short_phone, closure_segment, pause_silence_segment, glottal_stop.
Return JSON only. You may choose the intermediate structure that is most useful for Agent C.
Include utterance ID, phone index, phone label, start/end in seconds, event type, and duration."""

TASK3_C1C2_AGENT_A = """You are Agent A for TIMIT word-timing analysis.
Input contains .WRD word intervals. Detect word-level timing events: long_word, short_word, inter_word_gap, boundary_gap.
Return a simple JSON object matching this schema:
{
  "events": [
    {"event_type": "long_word", "word_index": 2, "start": 0.5, "end": 1.2, "utt_id": "..."}
  ]
}
Return JSON only."""

TASK3_C1C2_AGENT_B = """You are Agent B for TIMIT phone-event analysis.
Input contains .PHN phone intervals and word alignments. Detect phone-level events: long_phone, short_phone, closure_segment, pause_silence_segment, glottal_stop.
Return a simple JSON object matching this schema:
{
  "events": [
    {"event_type": "long_phone", "phone_index": 12, "start": 0.5, "end": 0.8, "utt_id": "..."}
  ]
}
Return JSON only."""


TASK3_AGENT_C_FINAL = """You are Agent C for TIMIT timing and phone-linking aggregation.
Merge Agent A and Agent B outputs into this exact final JSON shape:
{
  "bundle_id": "...",
  "utterance_summaries": [{
    "utt_id": "...",
    "long_word_count": 0,
    "short_word_count": 0,
    "inter_word_gap_count": 0,
    "boundary_gap_count": 0,
    "long_phone_count": 0,
    "short_phone_count": 0,
    "closure_count": 0,
    "pause_silence_count": 0,
    "glottal_stop_count": 0,
    "total_event_count": 0
  }],
  "events": [{
    "utt_id": "...",
    "event_type": "long_phone",
    "word_index": 8,
    "phone_index": 21,
    "start": 1.31,
    "end": 1.57
  }],
  "unlinked_events": [{
    "utt_id": "...",
    "event_type": "...",
    "reason": "no_parent_word"
  }]
}
Return JSON only. Preserve all word-level and phone-level events. Link phone-level events to parent words using temporal overlap."""
