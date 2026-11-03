"""Load MixedWM38, clean it, and make a duplicate-aware stratified split (see PLAN.md)."""
import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Wafer_Map_Datasets.npz"
PATTERNS = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Scratch", "Random"]
SEED = 42


def load():
    d = np.load(RAW)
    X, Y = d["arr_0"].astype(np.int8), d["arr_1"].astype(np.int8)
    assert X.shape == (38015, 52, 52) and Y.shape == (38015, 8)
    n_three = int((X == 3).sum())
    X[X == 3] = 2                       # undocumented value -> defective die (PLAN.md)
    return X, Y, n_three


def map_hashes(X):
    return np.array([hashlib.sha1(np.ascontiguousarray(m).tobytes()).hexdigest() for m in X])


def split(X, Y):
    """70/15/15 split by unique map, stratified by label combination; duplicates stay together."""
    h = map_hashes(X)
    uniq, first, inverse = np.unique(h, return_index=True, return_inverse=True)
    combo = np.array(["".join(map(str, Y[i])) for i in first])          # one label per unique map
    g = np.arange(len(uniq))
    g_tr, g_rest = train_test_split(g, test_size=0.30, stratify=combo, random_state=SEED)
    g_va, g_te = train_test_split(g_rest, test_size=0.50, stratify=combo[g_rest], random_state=SEED)
    sets = {"train": set(g_tr), "val": set(g_va), "test": set(g_te)}
    idx = {k: np.where(np.isin(inverse, list(v)))[0] for k, v in sets.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(h[idx[a]]) & set(h[idx[b]]), f"duplicate map shared by {a} and {b}"
    return idx, h


def naive_leakage(h):
    """Secondary: test maps with an identical twin in training under a naive map-level random split."""
    i = np.arange(len(h))
    tr, rest = train_test_split(i, test_size=0.30, random_state=SEED)
    _, te = train_test_split(rest, test_size=0.50, random_state=SEED)
    train_h = set(h[tr])
    return int(sum(x in train_h for x in h[te])), len(te)
