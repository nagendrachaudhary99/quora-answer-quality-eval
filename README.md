# Answer Quality Agent - Offline Evaluation Harness

[![ci](https://github.com/nagendrachaudhary99/quora-answer-quality-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/nagendrachaudhary99/quora-answer-quality-eval/actions/workflows/ci.yml)

Q&A platforms are drowning in AI-written, low-value answers on top of the
usual spam and harassment. An agent that scores answers against content
policies is the natural fix - but such agents fail silently, and a model or
prompt change can quietly tank quality until users notice. This repo is the
measurement layer that catches that before it ships: an offline evaluation
harness over a versioned golden set, with per-policy precision/recall/F1,
latency, cost, a threshold sweep, and a CI regression gate.

Built by Nagendra Chaudhary. Architecture, tradeoffs, failure handling, and
the scaling path are in [DESIGN.md](DESIGN.md). Related prior work:
Proxyeval, a harness that caught a 26.1% scoring bug in an LLM rubric
pipeline by diffing per-criterion scores across runs.

## What it does

- **Policies**: `spam`, `harassment`, `ai_slop` (AI-written low-value answers).
- **Scorers** (one interface, `policyguard/base.py`): `RuleBasedScorer`, a
  deterministic offline zero-cost baseline; `LLMScorer`, a stub for any
  OpenAI-compatible endpoint, disabled without `OPENAI_API_KEY` and clearly
  marked unimplemented.
- **Evaluation**: per-policy precision/recall/F1, any-violation rollup,
  p50/p90/p99 latency, estimated cost per 1k items, and a threshold sweep
  that picks the operating point maximizing any-violation F1.
- **Regression gate**: `--gate 0.80` fails the run (and the CI build) if
  any-violation F1 at threshold 0.5 drops below the floor.

## Run it

```sh
python3 data/make_dataset.py                 # regenerates the labeled set
python3 -m unittest discover -s tests -v     # 18 tests
python3 -m policyguard.evaluate --out results/report.json
python3 -m policyguard.evaluate --gate 0.80  # regression gate (exit 2 on fail)
```

## Verified results (this machine, rule-based scorer, n=34)

| policy      | precision | recall | f1   |
|-------------|-----------|--------|------|
| spam        | 1.00      | 1.00   | 1.00 |
| harassment  | 1.00      | 0.33   | 0.50 |
| ai_slop     | 1.00      | 0.83   | 0.91 |
| ANY (rollup)| 1.00      | 0.75   | 0.86 |

- p50 latency 0.07 ms/item, p99 1.1 ms/item, $0 estimated marginal cost.
- Threshold sweep: 0.4 lifts any-violation F1 to **0.97** (harassment recall
  is the binding constraint - flagged honestly, not hidden).
- Full numbers, dataset hash, and skip counts in `results/report.json`.

## What is real vs mocked

- REAL: the harness, scorer interface, metrics, threshold sweep, regression
  gate, CI, tests, and every reported number (reproduce with the commands
  above).
- SYNTHETIC (labeled by construction, `data/make_dataset.py`): the 34-item
  evaluation set. No real platform user content is used. The two "hard
  negative" rows are deliberate probes for over-flagging.
- NOT INCLUDED: a live LLM scorer (needs an API key; interface, cost model,
  and fallback are ready), and production-scale concerns (Spark-scale
  batching, rollout, online A/B) - this harness is the offline-eval half of
  that loop. See DESIGN.md for how those plug in.

## Sources

- Quora's fight against AI-written answers: https://www.pangram.com/blog/quora-case-study
- Quora's ML platform at scale: https://aws.amazon.com/blogs/containers/how-quora-modernized-mlops-on-amazon-eks-to-improve-customer-experience-with-scalable-ml-applications/
