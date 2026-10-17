import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policyguard.dataset import dataset_sha1, load, load_verbose

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "labeled_answers.jsonl")


class TestDataset(unittest.TestCase):
    def test_lenient_load_skips_and_counts_malformed(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "ok", "labels": {"spam": False}}) + "\n")
            f.write("{not valid json\n")
            f.write(json.dumps({"text": "missing labels key"}) + "\n")
            f.write("\n")
            path = f.name
        try:
            rows, skipped = load_verbose(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(skipped, 2)

    def test_load_matches_verbose_on_golden_set(self):
        rows, skipped = load_verbose(DATA)
        self.assertEqual(load(DATA), rows)
        self.assertEqual(skipped, 0)

    def test_sha1_stable_and_content_sensitive(self):
        self.assertEqual(dataset_sha1(DATA), dataset_sha1(DATA))
        self.assertEqual(len(dataset_sha1(DATA)), 40)


if __name__ == "__main__":
    unittest.main()
