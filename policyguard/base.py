"""The scorer contract.

Every scorer - heuristic, local model, or an LLM endpoint - implements this
interface. The evaluation harness only ever sees Scorer, so backends are
swappable without touching metrics or reporting code. See DESIGN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Score:
    """One scorer's output for one answer."""
    probs: dict = field(default_factory=dict)  # policy -> P(violation)
    latency_ms: float = 0.0
    provider: str = ""
    cost_usd: float = 0.0


@runtime_checkable
class Scorer(Protocol):
    """Maps answer text to per-policy violation probabilities in [0, 1]."""
    name: str
    cost_per_1k_usd: float

    def score(self, text: str) -> Score:
        ...
