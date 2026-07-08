from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def prf_from_counts(tp: int, pred: int, gold: int) -> dict[str, float]:
    precision = safe_div(tp, pred)
    recall = safe_div(tp, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def count_proxy_f1(gold_counts: Counter[tuple[Any, ...]], pred_counts: Counter[tuple[Any, ...]]) -> dict[str, float]:
    keys = set(gold_counts) | set(pred_counts)
    tp = sum(min(gold_counts[k], pred_counts[k]) for k in keys)
    pred = sum(pred_counts.values())
    gold = sum(gold_counts.values())
    out = prf_from_counts(tp, pred, gold)
    out.update({"tp": float(tp), "pred": float(pred), "gold": float(gold)})
    return out


def mae(gold_values: list[float], pred_values: list[float]) -> float:
    if not gold_values:
        return 0.0
    return sum(abs(g - p) for g, p in zip(gold_values, pred_values)) / len(gold_values)


def greedy_label_mapping(gold_labels: list[str], pred_labels: list[str]) -> dict[str, str]:
    scores: dict[tuple[str, str], int] = defaultdict(int)
    for gold, pred in zip(gold_labels, pred_labels):
        scores[(pred, gold)] += 1

    mapping: dict[str, str] = {}
    used_gold: set[str] = set()
    for (pred, gold), _score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        if pred in mapping or gold in used_gold:
            continue
        mapping[pred] = gold
        used_gold.add(gold)
    return mapping
