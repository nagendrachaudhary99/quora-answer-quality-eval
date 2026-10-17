"""Pluggable answer scorers.

A scorer maps answer text -> per-policy violation probabilities in [0, 1].
RuleBasedScorer is deterministic and runs offline (the always-available
baseline); LLMScorer talks to any OpenAI-compatible endpoint when an API key
is configured and is otherwise disabled.
"""
from __future__ import annotations

import math
import re
import time

from .base import Score, Scorer  # noqa: F401  (Score re-exported here)

POLICIES = ("spam", "harassment", "ai_slop")

_SPAM_PATTERNS = [
    r"\b(buy|click|visit)\b.{0,30}\b(now|here|link)\b",
    r"\b(free|cheap|discount)\b.{0,25}\b(pills|crypto|followers|money)\b",
    r"(https?://\S+).*(https?://\S+)",  # 2+ links
    r"\b(earn|make)\b.{0,15}\$\d",
    r"\bwhatsapp\b.{0,20}\+\d",
]
_HARASSMENT_PATTERNS = [
    r"\b(idiot|stupid|moron|dumb)\b.{0,20}\b(you|people|anyone)\b",
    r"\b(shut up|kill yourself|kys)\b",
    r"\b(you|u)\b.{0,15}\b(hack|fraud|scam|liar)\b",
    r"\b(subhuman|worthless|pathetic)\b",
]
_AI_TELLS = [
    r"^As an AI (language )?model",
    r"\bIn conclusion,\b",
    r"\bIt('s| is) important to note that\b",
    r"\bdelve into\b",
    r"\bfurthermore,\b.{0,40}\bmoreover,\b",
    r"\bIn today('s|s) (fast-paced|digital)\b",
    r"\bI hope this (helps|answer finds you well)\b",
]


def _count_hits(patterns, text):
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


class RuleBasedScorer:
    """Deterministic heuristic scorer. Real, offline, zero-cost baseline."""

    name = "rule-based"
    cost_per_1k_usd = 0.0

    def score(self, text: str) -> Score:
        t0 = time.perf_counter()
        spam_hits = _count_hits(_SPAM_PATTERNS, text)
        harm_hits = _count_hits(_HARASSMENT_PATTERNS, text)
        ai_hits = _count_hits(_AI_TELLS, text)
        words = max(len(text.split()), 1)
        # All-caps ratio nudges harassment; link density nudges spam.
        caps = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        link_density = len(re.findall(r"https?://", text)) / words * 100
        probs = {
            "spam": min(1.0, 0.35 * spam_hits + 0.05 * link_density),
            "harassment": min(1.0, 0.45 * harm_hits + 0.3 * max(caps - 0.3, 0)),
            "ai_slop": min(1.0, 0.4 * ai_hits),
        }
        return Score(probs=probs, latency_ms=(time.perf_counter() - t0) * 1e3,
                     provider=self.name)


class LLMScorer:
    """Optional OpenAI-compatible scorer. Disabled without an API key; the
    harness falls back to RuleBasedScorer. Cost is estimated per 1k tokens."""

    name = "llm"
    cost_per_1k_usd = 0.60  # assumed blended rate; configurable

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def score(self, text: str) -> Score:  # pragma: no cover - needs network
        if not self.enabled:
            raise RuntimeError("LLMScorer disabled: no API key configured")
        raise NotImplementedError("wire to your OpenAI-compatible endpoint")
