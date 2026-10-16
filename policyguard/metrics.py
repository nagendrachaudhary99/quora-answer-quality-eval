"""Offline evaluation metrics for content-quality agents."""
from __future__ import annotations

import math


def binary_confusion(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    return tp, fp, fn, tn


def prf1(y_true, y_pred):
    tp, fp, fn, _ = binary_confusion(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def percentiles(values, ps=(50, 90, 99)):
    if not values:
        return {p: 0.0 for p in ps}
    s = sorted(values)
    out = {}
    for p in ps:
        k = (len(s) - 1) * p / 100
        f, c = math.floor(k), math.ceil(k)
        out[p] = s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)
    return out


def evaluate(records, policies, threshold=0.5):
    """records: list of {labels: {policy: bool}, probs: {policy: float},
    latency_ms: float}. Returns per-policy PR/F1 + latency percentiles."""
    report = {"threshold": threshold, "n": len(records), "policies": {}}
    for pol in policies:
        y_true = [bool(r["labels"].get(pol)) for r in records]
        y_pred = [r["probs"].get(pol, 0.0) >= threshold for r in records]
        report["policies"][pol] = prf1(y_true, y_pred)
    report["latency_ms"] = percentiles([r["latency_ms"] for r in records])
    # any-violation rollup: flag if any policy trips
    y_true_any = [any(r["labels"].get(p) for p in policies) for r in records]
    y_pred_any = [any(r["probs"].get(p, 0.0) >= threshold for p in policies)
                  for r in records]
    report["any_violation"] = prf1(y_true_any, y_pred_any)
    return report


def threshold_sweep(records, policies, thresholds):
    """Pick the threshold maximizing any-violation F1."""
    best = None
    for t in thresholds:
        rep = evaluate(records, policies, threshold=t)
        f1 = rep["any_violation"]["f1"]
        if best is None or f1 > best[1]:
            best = (t, f1)
    return best
