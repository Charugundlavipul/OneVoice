# UXSSD Small-Scale Experiment Report (Exp1 vs Exp2)

## 1) Purpose
This experiment compares two extraction strategies for the same 15 UXSSD records:

- `Experiment 1 (Exp1)`: OneVoice-agnostic (raw files to agents)
- `Experiment 2 (Exp2)`: OneVoice pipeline (structured OneVoice JSON + validator-in-loop)

Main question:
- Does the OneVoice pipeline produce outputs that are closer to gold labels than the agnostic baseline?

## 2) Dataset and Record Selection
Source setup:
- `experiments/uxssd_setup/selected_bundles.csv`

Selection design:
- 15 records total
- Record duration range: 120 to 185 seconds
- Event-density target per record: 5 to 10 mispronunciation proxy events and 5 to 10 behavioral proxy events
- Cohort mix:
  - 5 `mispronunciation_heavy`
  - 8 `behavioral_heavy`
  - 2 `balanced`

Speaker spread:
- `02M_CHILD`: 4 records
- `04M_CHILD`: 3 records
- `05M_CHILD`: 3 records
- `06M_CHILD`: 3 records
- `08M_CHILD`: 2 records

## 3) Gold Labels
Gold files:
- `experiments/uxssd_setup/gold_labels/templates/*.gold.json`

Gold totals used for evaluation:
- Mispronunciations: 94
- Behavioral events: 112
- Total events: 206

Behavioral-label audit was rechecked (read-only) and no structural/temporal/type issues were found in existing events.

## 4) How the Experiments Were Conducted
Core runner logic:
- `experiments/uxssd_setup/runner_core.py`

Model:
- `gpt-5-mini`

Input limits used:
- `max_file_chars=3200`
- `max_files_per_utt=8`

Run artifacts:
- Exp1 primary run: `experiments/uxssd_setup/runs/exp1_20260220_195028`
- Exp2 was resumed across multiple runs and aggregated:
  - `experiments/uxssd_setup/runs/exp2_aggregate_resume/run_config.json`

Final comparison CSV:
- `experiments/uxssd_setup/runs/event_counts_gold_exp1_exp2.csv`

## 5) Outcome Summary (from latest CSV)
Raw predicted totals:
- Exp1:
  - Mispronunciations predicted: 8
  - Behavioral predicted: 1143
  - Total predicted: 1151
- Exp2:
  - Mispronunciations predicted: 81
  - Behavioral predicted: 262
  - Total predicted: 343

Gold totals:
- Mispronunciations: 94
- Behavioral: 112
- Total: 206

## 6) Analytics (Count-Based Proxy Metrics)
Important: this CSV contains event counts, not event-level matching by timestamps/types.  
So precision/recall below are count-based proxy metrics (`TP = min(gold_count, predicted_count)` per record).

### 6.1 Mispronunciation Detection
- Exp1: Precision 1.000, Recall 0.085, F1 0.157
- Exp2: Precision 0.778, Recall 0.670, F1 0.720

Interpretation:
- Exp1 almost always under-predicts mispronunciations.
- Exp2 recovers most mispronunciation counts much better.

### 6.2 Behavioral Event Detection
- Exp1: Precision 0.090, Recall 0.920, F1 0.164
- Exp2: Precision 0.397, Recall 0.929, F1 0.556

Interpretation:
- Both methods tend to over-predict behavioral events.
- Exp2 still over-predicts, but much less severely than Exp1.

### 6.3 Overall (Mispronunciation + Behavioral)
- Exp1: Precision 0.156, Recall 0.874, F1 0.265
- Exp2: Precision 0.592, Recall 0.985, F1 0.740

Error scale (absolute count error per record):
- Exp1 MAE: 66.47
- Exp2 MAE: 9.53

Net:
- Exp2 improves overall F1 by ~0.475 points (0.740 vs 0.265).
- Exp2 reduces overall MAE by ~85.7%.

## 7) Record-Level Comparison
Exp2 vs Exp1 by absolute total-count error:
- Improved on 13 of 15 records
- Worse on 2 records
- Tied on 0 records

Largest improvements:
- `06M_Post_w020_036`: gold 14, exp1 159, exp2 20
- `05M_Post_w013_023`: gold 16, exp1 155, exp2 18
- `06M_Post_w005_012`: gold 15, exp1 147, exp2 12

Cases where Exp2 was worse than Exp1:
- `02M_Maint2_w015_039`: gold 13, exp1 10, exp2 18
- `04M_Post_w003_012`: gold 12, exp1 7, exp2 24

## 8) What This Experiment Proves
What is supported by this experiment:
- The OneVoice pipeline (Exp2) is substantially more stable and closer to gold than the OneVoice-agnostic baseline (Exp1) on this dataset.
- Structured schema + validator loop prevents extreme failure behavior seen in Exp1, especially runaway behavioral event counts.
- Exp2 provides a much better balance between detecting mispronunciations and behavioral events.

What is not fully proven yet:
- Exact event-level precision/recall by time overlap and event type matching (not available from count-only CSV).
- Generalization beyond this 15-record small-scale setup.

## 9) Recommended Paper Wording (Simple English)
Suggested phrasing:

> We compared two pipelines for extracting mispronunciation and behavioral events from UXSSD recordings.  
> The OneVoice-agnostic baseline (Exp1) produced unstable outputs, especially severe overprediction of behavioral events.  
> The OneVoice pipeline with schema constraints and validator-in-loop (Exp2) was substantially closer to gold labels, improving overall proxy F1 from 0.265 to 0.740 and reducing average per-record count error from 66.47 to 9.53.  
> This indicates that structured intermediate representation and validation are critical for reliable multi-event extraction.

## 10) Caveats
- Metrics above are count-based proxies, not event-boundary matching.
- Gold labels were built in this workflow and should still be considered a small-scale evaluation set.
- Some residual behavioral overprediction remains in Exp2 and is a target for next iteration.
