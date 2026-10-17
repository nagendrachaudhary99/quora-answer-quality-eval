import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policyguard.base import Scorer
from policyguard.evaluate import run
from policyguard.scorers import LLMScorer, RuleBasedScorer

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "labeled_answers.jsonl")


class TestGate(unittest.TestCase):
    def test_gate_passes_below_observed(self):
        r = run(DATA, RuleBasedScorer(), gate=0.50)
        self.assertTrue(r["gate"]["passed"])

    def test_gate_fails_above_observed(self):
        r = run(DATA, RuleBasedScorer(), gate=0.999)
        self.assertFalse(r["gate"]["passed"])
        self.assertEqual(r["gate"]["metric"], "any_violation.f1@0.5")

    def test_no_gate_by_default(self):
        self.assertNotIn("gate", run(DATA, RuleBasedScorer()))

    def test_scorers_satisfy_protocol(self):
        self.assertIsInstance(RuleBasedScorer(), Scorer)
        self.assertIsInstance(LLMScorer(), Scorer)


if __name__ == "__main__":
    unittest.main()
