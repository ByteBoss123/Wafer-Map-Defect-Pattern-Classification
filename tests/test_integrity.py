"""Integrity checks. Run from the repo root after src/train.py: python -m pytest tests -q"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import PATTERNS, load, split  # noqa: E402

M = json.load(open(ROOT / "results" / "metrics.json"))


def test_checksums():
    for line in (ROOT / "data" / "raw" / "SHA256SUMS").read_text().split("\n"):
        if not line.strip():
            continue
        want, rel = line.split()
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == want, rel


def test_documented_shape_and_counts():
    X, Y, n_three = load()
    assert X.shape == (38015, 52, 52) and Y.shape == (38015, 8)
    assert n_three == 214 and set(np.unique(X)) == {0, 1, 2}
    assert len({"".join(map(str, r)) for r in Y}) == 38
    per = dict(zip(PATTERNS, Y.sum(axis=0).tolist()))
    assert per == {"Center": 13000, "Donut": 12000, "Edge_Loc": 13000, "Edge_Ring": 12000,
                   "Loc": 18000, "Near_full": 149, "Scratch": 19000, "Random": 866}


def test_split_has_no_shared_maps():
    X, Y, _ = load()
    idx, h = split(X, Y)
    assert sum(len(v) for v in idx.values()) == 38015
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(h[idx[a]]) & set(h[idx[b]])
    assert [len(idx[k]) for k in ("train", "val", "test")] == [M["data"][k] for k in ("train", "val", "test")]


def test_metrics_consistent():
    for k in ("B0_no_skill", "B1_logistic", "M1_cnn"):
        r = M[k]
        assert 0 <= r["macro_f1"] <= 1 and 0 <= r["exact_match"] <= 1
        assert list(r["per_pattern"]) == PATTERNS
        assert abs(np.mean([v["f1"] for v in r["per_pattern"].values()]) - r["macro_f1"]) < 1e-3
        assert sum(v["support"] for v in r["per_pattern"].values()) == sum(
            v["support"] for v in M["B0_no_skill"]["per_pattern"].values())
    lo, hi = M["M1_cnn"]["bootstrap"]["macro_f1_95ci"]
    assert lo <= M["M1_cnn"]["macro_f1"] <= hi
    best = max(e["val_macro_f1"] for e in M["M1_cnn"]["training_log"])
    assert best == M["M1_cnn"]["best_val_macro_f1"]
