#!/usr/bin/env python3
"""Run a scorer over the labeled set and print an offline evaluation report:
per-policy precision/recall/F1, any-violation rollup, latency percentiles,
estimated cost, and a threshold sweep for the accuracy/latency/cost tradeoff.

Usage: python3 -m policyguard.evaluate [--data PATH] [--scorer rule-based]
"""
import argparse
import json
import os

from .dataset import load
from .metrics import evaluate, threshold_sweep
from .scorers import POLICIES, LLMScorer, RuleBasedScorer


def run(data_path, scorer):
    rows = load(data_path)
    records = []
    for r in rows:
        s = scorer.score(r["text"])
        records.append({"labels": r["labels"], "probs": s.probs,
                        "latency_ms": s.latency_ms})
    base = evaluate(records, POLICIES, threshold=0.5)
    best_t, best_f1 = threshold_sweep(records, POLICIES,
                                      [i / 100 for i in range(10, 95, 5)])
    n = len(records)
    report = {
        "scorer": scorer.name,
        "n_items": n,
        "threshold_0.5": base,
        "best_threshold": best_t,
        "best_threshold_f1": best_f1,
        "est_cost_per_1k_items_usd": getattr(scorer, "cost_per_1k_usd", 0.0),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__),
                                                   "..", "data",
                                                   "labeled_answers.jsonl"))
    ap.add_argument("--scorer", default="rule-based",
                    choices=["rule-based", "llm"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.scorer == "llm":
        scorer = LLMScorer(api_key=os.environ.get("OPENAI_API_KEY"))
        if not scorer.enabled:
            print("LLM scorer disabled (no OPENAI_API_KEY); using rule-based")
            scorer = RuleBasedScorer()
    else:
        scorer = RuleBasedScorer()
    report = run(args.data, scorer)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
