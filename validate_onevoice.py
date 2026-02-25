
import json
import argparse
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# Define the expected OneVoice schema structure
# Keys map to expected types or list of types.
# For nested structures, we define dictionaries.

ONEVOICE_SCHEMA = {
    "session_id": str,
    "session_date": str,
    "sample_rate": (int, float, str),
    "audio_file_path": str,
    "audio_format": str,
    "session_duration": str,
    "speakers": [
        {
            "speaker_id": str,
            "gender": str,
            "language": str,
            "native_language": str,
            "accent": str,
            "age": (int, float, str),
            "name": str,
            "age_group": str
        }
    ],
    "turns": [
        {
            "turn_index": (int, str),
            "utt_id": str,
            "speaker_id": str,
            "reference_transcript": str,
            "phoneme_reference": str,
            "orthographic_transcript": str,
            "phonetic_transcript": str,
            "ipa_transcript": str,
            "raw_transcript": str,
            "start": str,
            "end": str,
            "word_alignments": [
                {
                    "word_index": (int, str),
                    "word": str,
                    "start": str,
                    "end": str,
                    "phonemes": [
                        {
                            "phone": str,
                            "start": str,
                            "end": str,
                            "phone_index": (int, str)
                        }
                    ]
                }
            ],
            "mispronunciations": [
                {
                    "word_index": (int, str),
                    "phone_index": (int, str),
                    "target_phone": str,
                    "observed_phone": str,
                    "type": str,
                    "start": str,
                    "end": str
                }
            ],
            "behavioral_events": [
                {
                    "type": str,
                    "start": str,
                    "end": str
                }
            ]
        }
    ],
    "metadata": {
        "dataset_name": str,
        "dataset_split": str,
        "text_annotation_source": str,
        "text_annotation_details": str,
        "text_coverage": str,
        "phoneme_annotation_source": str,
        "phoneme_annotation_details": str,
        "phoneme_coverage": str
    }
}

TIME_FORMAT = "%H:%M:%S:%f"

def parse_time(time_str: str) -> Optional[datetime]:
    """Parses a time string in format HH:MM:SS:mmm."""
    if not time_str or not isinstance(time_str, str):
        return None
    try:
        # The schema says HH:MM:SS:mmm, commonly this means milliseconds which is 3 digits.
        # Python's %f expects microseconds (6 digits). We might need to handle padding/stripping.
        # But 'mmm' usually implies milliseconds.
        # Let's try to handle standard format.
        # If the input has 3 digits for ms, we can pad it to 6 for parsing.
        parts = time_str.split(':')
        if len(parts) == 4:
            ms = parts[3]
            if len(ms) == 3:
                time_str = f"{parts[0]}:{parts[1]}:{parts[2]}:{ms}000"
        return datetime.strptime(time_str, TIME_FORMAT)
    except ValueError:
        return None

def validate_time_range(start_str: str, end_str: str, path: str) -> List[str]:
    """Validates that start < end."""
    errors = []
    start = parse_time(start_str)
    end = parse_time(end_str)

    if start_str and not start:
         errors.append(f"{path}: Invalid start time format '{start_str}' (expected HH:MM:SS:mmm)")
    if end_str and not end:
         errors.append(f"{path}: Invalid end time format '{end_str}' (expected HH:MM:SS:mmm)")

    if start and end:
        if start >= end:
            errors.append(f"{path}: Start time ({start_str}) must be less than end time ({end_str})")
    
    return errors

def check_containment(parent_start: str, parent_end: str, child_start: str, child_end: str, path: str) -> List[str]:
    """Validates that child interval is within parent interval."""
    errors = []
    p_start = parse_time(parent_start)
    p_end = parse_time(parent_end)
    c_start = parse_time(child_start)
    c_end = parse_time(child_end)

    if p_start and c_start and c_start < p_start:
        errors.append(f"{path}: Child start ({child_start}) is before parent start ({parent_start})")
    
    if p_end and c_end and c_end > p_end:
        errors.append(f"{path}: Child end ({child_end}) is after parent end ({parent_end})")
        
    return errors

def validate_structure(data: Any, schema: Any, path: str = "") -> List[str]:
    """
    Validates structure recursively.
    Rules:
    1. Keys in data MUST be present in schema (Unknown keys forbidden).
    2. Missing keys in data are ALLOWED (Optionality).
    3. Types and nesting must match.
    """
    errors = []

    if isinstance(schema, dict):
        if not isinstance(data, dict):
            errors.append(f"{path}: Expected dictionary, got {type(data).__name__}")
            return errors
        
        # Check for unknown keys
        allowed_keys = set(schema.keys())
        actual_keys = set(data.keys())
        unknown_keys = actual_keys - allowed_keys
        if unknown_keys:
            for key in unknown_keys:
                errors.append(f"{path}: Unknown key '{key}' found. Not allowed in OneVoice schema.")
        
        # Recurse for present keys
        for key, sub_schema in schema.items():
            if key in data:
                errors.extend(validate_structure(data[key], sub_schema, path=f"{path}.{key}" if path else key))
                
    elif isinstance(schema, list):
        if not isinstance(data, list):
            errors.append(f"{path}: Expected list, got {type(data).__name__}")
            return errors
        
        if schema:
            # Validate each item in data against the schema items
            # Assuming homogeneous list definition in schema (single item list)
            item_schema = schema[0]
            for i, item in enumerate(data):
                 errors.extend(validate_structure(item, item_schema, path=f"{path}[{i}]"))
    
    return errors

def validate_temporal_logic(data: Dict[str, Any]) -> List[str]:
    """
    Performs specific temporal consistency checks.
    """
    errors = []
    
    # Session duration
    session_duration = data.get("session_duration")
    
    if "turns" in data:
        for t_idx, turn in enumerate(data["turns"]):
            t_path = f"turns[{t_idx}]"
            t_start = turn.get("start")
            t_end = turn.get("end")
            
            # Turn time range
            errors.extend(validate_time_range(t_start, t_end, t_path))
            
            # Turn within Session
            if session_duration:
                 # Assuming session starts at 00:00:00:000 implicitly? 
                 # Or just end check. Start is usually absolute or relative to 0.
                 # Let's check end <= session_duration
                 p_end = parse_time(session_duration)
                 c_end = parse_time(t_end)
                 if p_end and c_end and c_end > p_end:
                     errors.append(f"{t_path}: Turn end ({t_end}) exceeds session duration ({session_duration})")

            if "word_alignments" in turn:
                for w_idx, word in enumerate(turn["word_alignments"]):
                    w_path = f"{t_path}.word_alignments[{w_idx}]"
                    w_start = word.get("start")
                    w_end = word.get("end")
                    
                    # Word time range
                    errors.extend(validate_time_range(w_start, w_end, w_path))
                    
                    # Word within Turn
                    errors.extend(check_containment(t_start, t_end, w_start, w_end, w_path))
                    
                    if "phonemes" in word:
                        for p_idx, phone in enumerate(word["phonemes"]):
                            p_path = f"{w_path}.phonemes[{p_idx}]"
                            p_start = phone.get("start")
                            p_end = phone.get("end")
                            
                            # Phone time range
                            errors.extend(validate_time_range(p_start, p_end, p_path))
                            
                            # Phone within Word
                            errors.extend(check_containment(w_start, w_end, p_start, p_end, p_path))

    return errors

def validate_mandatory_fields(data: Dict[str, Any]) -> List[str]:
    """
    Checks for specific mandatory fields:
    - Root: session_id, turns, speakers, metadata
    - Turns: turn_index, utt_id
    - Speakers: speaker_id (when speakers array is present)
    - Mispronunciations: type and at least one of target_phone/observed_phone
    - Behavioral events: type
    """
    errors = []

    def is_populated(value: Any) -> bool:
        """Treat None/empty-string as missing; numeric 0 is valid."""
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        return True
    
    # Root level checks
    mandatory_root = ["session_id", "turns", "speakers", "metadata"]
    for field in mandatory_root:
        if field not in data:
            errors.append(f"Missing mandatory root field: '{field}'")
            
    # Sub-level checks
    if "turns" in data and isinstance(data["turns"], list):
        for i, turn in enumerate(data["turns"]):
            if not isinstance(turn, dict):
                continue

            if not is_populated(turn.get("turn_index")):
                errors.append(f"turns[{i}]: Missing mandatory field 'turn_index'")
            if not is_populated(turn.get("utt_id")):
                errors.append(f"turns[{i}]: Missing mandatory field 'utt_id'")

            if "mispronunciations" in turn and isinstance(turn["mispronunciations"], list):
                for j, mis in enumerate(turn["mispronunciations"]):
                    if not isinstance(mis, dict):
                        continue
                    if not is_populated(mis.get("type")):
                        errors.append(f"turns[{i}].mispronunciations[{j}]: Missing mandatory field 'type'")
                    has_target_phone = is_populated(mis.get("target_phone"))
                    has_observed_phone = is_populated(mis.get("observed_phone"))
                    if not (has_target_phone or has_observed_phone):
                        errors.append(
                            f"turns[{i}].mispronunciations[{j}]: At least one of 'target_phone' or 'observed_phone' is mandatory"
                        )

            if "behavioral_events" in turn and isinstance(turn["behavioral_events"], list):
                for j, event in enumerate(turn["behavioral_events"]):
                    if not isinstance(event, dict):
                        continue
                    if not is_populated(event.get("type")):
                        errors.append(f"turns[{i}].behavioral_events[{j}]: Missing mandatory field 'type'")

    if "speakers" in data and isinstance(data["speakers"], list):
        for i, speaker in enumerate(data["speakers"]):
            if not isinstance(speaker, dict):
                continue
            if not is_populated(speaker.get("speaker_id")):
                errors.append(f"speakers[{i}]: Missing mandatory field 'speaker_id'")

    return errors

def validate_content_presence(data: Dict[str, Any]) -> List[str]:
    """
    Validates that either 'audio_file_path' OR 'raw_transcript' (in turns) is populated.
    Populated means: Not None and not empty string "".
    """
    errors = []
    
    # Check Audio
    audio_path = data.get("audio_file_path")
    has_audio = audio_path is not None and isinstance(audio_path, str) and audio_path.strip() != ""
    
    # Check Text
    has_text = False
    if "turns" in data and isinstance(data["turns"], list):
        for turn in data["turns"]:
            if isinstance(turn, dict):
                raw = turn.get("raw_transcript")
                if raw is not None and isinstance(raw, str) and raw.strip() != "":
                    has_text = True
                    break
    
    if not has_audio and not has_text:
        errors.append("Content Error: Document must have either a populated 'audio_file_path' or at least one turn with a populated 'raw_transcript'.")
        
    return errors

def main():
    parser = argparse.ArgumentParser(description="OneVoice JSON Validator")
    parser.add_argument("json_file", help="Path to the JSON file to validate")
    parser.add_argument("--mode", choices=["structure", "full"], default="full", 
                        help="Validation mode. 'structure' checks keys only (ignoring types slightly, but checking containment). 'full' runs all checks including temporal logic.")
    
    args = parser.parse_args()

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

    print(f"Validating {args.json_file}...")
    
    structure_errors = validate_structure(data, ONEVOICE_SCHEMA)
    temporal_errors = []
    
    if args.mode == "full":
        temporal_errors = validate_temporal_logic(data)

    all_errors = structure_errors + temporal_errors
    
    # Mandatory field check
    mandatory_errors = validate_mandatory_fields(data)
    all_errors.extend(mandatory_errors)
    
    # Content presence check
    content_errors = validate_content_presence(data)
    all_errors.extend(content_errors)

    if all_errors:
        print("Validation FAILED with the following errors:")
        for error in all_errors:
            print(f" - {error}")
        sys.exit(1)
    else:
        print("Validation SUCCESSFUL: Structure and Constraints passed.")

if __name__ == "__main__":
    main()
