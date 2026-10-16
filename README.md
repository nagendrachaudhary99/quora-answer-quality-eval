# Answer Quality Agent — Offline Evaluation Harness

A small, honest version of the loop Quora's ML team describes in its
Machine Learning Engineer (New Grad) posting (Ashby req
40659fe7-545e-4b08-b6d7-2567b18b9ed3): an agent that scores answers against
content policies, evaluated **offline** on accuracy, latency, and cost before
anything ships.

Built by Nagendra Chaudhary. Related prior work: Proxyeval, a harness that
caught a 26.1% scoring bug in an LLM rubric pipeline by diffing per-criterion
scores across runs (same "investigate failure cases, iterate" muscle).

## What it does

- **Policies**: `spam`, `harassment`, `ai_slop` (AI-written low-value answers —
  the failure mode Quora publicly fought with Pangram; see README sources).
- **Scorers** (pluggable): `RuleBasedScorer` — deterministic, offline,
  zero-cost baseline; `LLMScorer` — stub for any OpenAI-compatible endpoint,
  disabled without `OPENAI_API_KEY` and clearly marked unimplemented.
- **Evaluation**: per-policy precision/recall/F1, any-violation rollup,
  p50/p90/p99 latency, estimated cost per 1k items, and a threshold sweep that
  picks the operating point maximizing any-violation F1 — the
  accuracy/reliability/latency/cost tradeoff the JD calls out.

## Run it

```sh
python3 data/make_dataset.py                 # regenerates the labeled set
python3 -m unittest discover -s tests -v     # 11 tests
python3 -m policyguard.evaluate --out results/report.json
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
  is the binding constraint — flagged honestly, not hidden).
- Full numbers in `results/report.json`.

## What is real vs mocked

- REAL: the harness, metrics, threshold sweep, scorer interface, tests, and
  every reported number (reproduce with the commands above).
- SYNTHETIC (labeled by construction, `data/make_dataset.py`): the 34-item
  evaluation set. No real Quora user content is used. The two "hard negative"
  rows are deliberate probes for over-flagging.
- NOT INCLUDED: a live LLM scorer (needs an API key; interface is ready), and
  production concerns the JD mentions (Spark-scale batching, Kubernetes
  rollout, online A/B) — the harness is the offline-eval piece of that loop.

## Sources

- JD: https://jobs.ashbyhq.com/quora/40659fe7-545e-4b08-b6d7-2567b18b9ed3
- Quora + AI-written answers: https://www.pangram.com/blog/quora-case-study
- Quora ML platform: https://aws.amazon.com/blogs/containers/how-quora-modernized-mlops-on-amazon-eks-to-improve-customer-experience-with-scalable-ml-applications/
