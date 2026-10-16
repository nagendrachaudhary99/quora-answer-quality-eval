import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policyguard.metrics import binary_confusion, evaluate, percentiles, prf1, threshold_sweep
from policyguard.scorers import POLICIES


class TestMetrics(unittest.TestCase):
    def test_confusion_counts(self):
        tp, fp, fn, tn = binary_confusion([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((tp, fp, fn, tn), (1, 1, 1, 1))

    def test_prf1_perfect(self):
        m = prf1([1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)

    def test_prf1_no_positives_predicted(self):
        m = prf1([1, 0], [0, 0])
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)

    def test_percentiles(self):
        p = percentiles(list(range(1, 101)))
        self.assertAlmostEqual(p[50], 50.5)
        self.assertEqual(p[99], 99.01)

    def test_evaluate_rollup(self):
        recs = [
            {"labels": {"spam": True}, "probs": {"spam": 0.9}, "latency_ms": 1.0},
            {"labels": {"spam": False}, "probs": {"spam": 0.1}, "latency_ms": 2.0},
        ]
        rep = evaluate(recs, POLICIES, threshold=0.5)
        self.assertEqual(rep["n"], 2)
        self.assertEqual(rep["policies"]["spam"]["f1"], 1.0)
        self.assertEqual(rep["any_violation"]["f1"], 1.0)

    def test_threshold_sweep_picks_best(self):
        recs = [
            {"labels": {"spam": True}, "probs": {"spam": 0.4}, "latency_ms": 1},
            {"labels": {"spam": False}, "probs": {"spam": 0.3}, "latency_ms": 1},
        ]
        t, f1 = threshold_sweep(recs, POLICIES, [0.35, 0.5, 0.6])
        self.assertEqual(t, 0.35)  # only threshold catching the 0.4 positive


if __name__ == "__main__":
    unittest.main()
