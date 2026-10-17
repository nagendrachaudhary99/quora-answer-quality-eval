"""Load the labeled evaluation set (JSONL: text + per-policy bool labels).

Lenient by design: a malformed row is skipped and counted rather than
aborting the whole eval run, and the skip count is surfaced in the report
so silent data loss stays visible. See DESIGN.md, failure handling.
"""
import hashlib
import json


def dataset_sha1(path):
    """Content hash of the golden set. Every report carries it, so a metric
    change is attributable to the scorer vs the data."""
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def load_verbose(path):
    """Return (rows, n_skipped). Rows with invalid JSON or missing
    'text'/'labels' keys are skipped and counted."""
    rows, skipped = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(row, dict) or "text" not in row or "labels" not in row:
                skipped += 1
                continue
            rows.append(row)
    return rows, skipped


def load(path):
    return load_verbose(path)[0]
