# Design: offline evaluation harness for a content-quality agent

## Problem

Q&A platforms are now flooded with AI-written, low-value answers on top of the
classic spam and harassment load (Quora's public fight against AI-generated
answers with Pangram is the well-known example). The natural fix is an agent
that scores every answer against content policies. The hard part is not
building that agent; it is knowing whether it is any good, and noticing when a
prompt, model, or heuristic change quietly makes it worse. Scoring quality
fails silently: nobody sees the regression until users do.

This repo is the measurement layer that prevents that: a repeatable offline
evaluation loop over a versioned golden set, wired into CI as a regression
gate.

## What "solved" means here

Solved: given any scorer behind the `Scorer` interface, you get per-policy
precision/recall/F1, an any-violation rollup, latency percentiles, a cost
estimate, and a threshold sweep in one command; the dataset is content-hashed
so results are attributable; CI fails the build if quality drops below the
pinned floor. The failure mode "we shipped a worse scorer and found out from
users" is closed for any scorer plugged into this harness.

Deliberately not solved (and labeled as such): the bundled scorer is a
deterministic heuristic baseline over a small synthetic set, so its absolute
numbers are not a production claim. The LLM scorer is an interface + cost
model + fallback, not a wired endpoint. Real production data and a live
endpoint are the two known gaps, both bounded by the interfaces here.

## Architecture

```
            +-------------------------+
            | data/labeled_answers    |  golden set, versioned by sha1
            | (JSONL: text + labels)  |  (hash printed in every report)
            +------------+------------+
                         v
+----------------+   +---+------------------+   +-------------------------+
| Scorer         |-->| evaluate.run         |-->| report.json             |
| (Protocol in   |   |  - lenient loader    |   | per-policy PR/F1        |
|  base.py)      |   |  - metrics module    |   | any-violation rollup    |
+-------+--------+   |  - threshold sweep   |   | latency p50/90/99       |
        ^            +---+------------------+   | cost per 1k items       |
        |                |                      +-----------+-------------+
   two backends:         v                                  |
   RuleBasedScorer   threshold sweep                        v
   LLMScorer (stub,                            CI regression gate:
   disabled w/o key)                           F1@0.5 < floor -> build fails
```

Components:

- `policyguard/base.py` - the `Scorer` protocol and `Score` value type. The
  whole harness depends only on this contract.
- `policyguard/scorers.py` - `RuleBasedScorer` (deterministic, offline,
  zero-cost baseline) and `LLMScorer` (stub for any OpenAI-compatible
  endpoint, disabled without a key).
- `policyguard/dataset.py` - lenient JSONL loader + content hash.
- `policyguard/metrics.py` - confusion counts, PR/F1, percentiles, sweep.
- `policyguard/evaluate.py` - runner, report writer, regression gate.
- `.github/workflows/ci.yml` - unit tests + gated eval on every push.

## Key tradeoffs

- **Rule-based baseline first, LLM behind the same interface.** The baseline
  is deterministic and free, so CI is stable and every future scorer has a
  floor to beat. Considered: mocking an LLM for the demo. Rejected: a mock
  proves nothing, and an unverifiable quality claim is worse than a stub.
- **Any-violation rollup as the headline metric, per-policy breakdown beside
  it.** Moderation cares first about "did we catch the bad answer at all";
  the per-policy table is where the weakness shows (harassment recall 0.33,
  reported openly, not averaged away).
- **Threshold sweep instead of a fixed 0.5.** The operating point is a
  product decision (over-flagging good answers vs missing bad ones). The
  harness reports the whole tradeoff curve; CI pins only the floor (0.80).
- **Golden set versioned by content hash.** Every report carries the dataset
  sha1, so a metric delta is attributable to the scorer or the data, not
  lost in "did someone edit the labels?"
- **Lenient loader.** A malformed row is skipped and counted
  (`n_skipped_malformed` in the report) instead of aborting the run.
  Considered: fail hard. Rejected for eval batches, but the skip count keeps
  silent data loss visible.

## Failure handling

- Malformed dataset rows: skipped, counted, reported.
- LLM scorer without a key: explicit disabled state with a printed fallback
  notice; never silently scored by the wrong backend.
- Scorer quality regression: CI gate exits non-zero below the floor.
- Latency/cost regressions: measured and reported every run (p50/p90/p99,
  $/1k items) but not gated yet - see limits.

## Scaling path

- **Larger golden sets.** Scoring is embarrassingly parallel over rows; shard
  the set, map `score()`, reduce the additive confusion counts (percentiles
  switch to t-digest past a few million rows).
- **A real LLM scorer.** Batching, retry/backoff, and response caching by
  text hash all live behind `score()`; the cost model is already in the
  report, so the accuracy-vs-cost comparison against the baseline is one
  flag away.
- **Dataset growth.** The production loop mines hard negatives from appealed
  over-flags; the two hand-written hard negatives here are the seed of that
  idea.
- **Online half.** Shadow scoring plus sampled human labels feeding this same
  metrics module. This repo is deliberately the offline half of that loop.

## Honest limits

- The heuristic scorer overfits its own synthetic set (spam F1 1.00 is not a
  real-world claim; the README says so).
- n=34 is demo scale. The weak number (harassment recall 0.33 at threshold
  0.5) is the real result, kept visible.
- `LLMScorer.score` raises `NotImplementedError`. What is built is the
  interface, the cost model, the fallback, and the gate around it.
