import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policyguard.dataset import load
from policyguard.scorers import RuleBasedScorer


class TestRuleBasedScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = RuleBasedScorer()

    def test_spam_scores_high(self):
        s = self.scorer.score("Earn $500 fast! Click here now http://x.co http://y.co")
        self.assertGreaterEqual(s.probs["spam"], 0.5)

    def test_harassment_scores_high(self):
        s = self.scorer.score("You are an idiot, shut up, everyone knows you are a liar")
        self.assertGreaterEqual(s.probs["harassment"], 0.5)

    def test_ai_slop_scores_high(self):
        s = self.scorer.score("As an AI language model, it's important to note that in today's fast-paced world we should delve into this. I hope this helps!")
        self.assertGreaterEqual(s.probs["ai_slop"], 0.5)

    def test_clean_scores_low(self):
        s = self.scorer.score("Compound interest rewards starting early because returns compound on prior returns over decades.")
        self.assertTrue(all(p < 0.5 for p in s.probs.values()))

    def test_dataset_labels_load(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "labeled_answers.jsonl")
        rows = load(path)
        self.assertGreaterEqual(len(rows), 30)
        for r in rows:
            self.assertIn("text", r)
            self.assertIn("labels", r)


if __name__ == "__main__":
    unittest.main()
