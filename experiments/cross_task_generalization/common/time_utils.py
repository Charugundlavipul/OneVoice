from __future__ import annotations


def seconds_to_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000.0))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}:{ms:03d}"


def ts_to_seconds(ts: object) -> float | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    parts = ts.strip().split(":")
    if len(parts) != 4:
        return None
    try:
        h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3][:3])
    except ValueError:
        return None
    return h * 3600 + m * 60 + s + ms / 1000.0


def sample_to_seconds(value: int | float, sample_rate: int = 16000) -> float:
    return float(value) / float(sample_rate)
