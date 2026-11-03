"""Train B0/B1/M1 as fixed in PLAN.md; score the test set once. Run: python src/train.py"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from data import PATTERNS, ROOT, SEED, load, naive_leakage, split

torch.manual_seed(SEED); np.random.seed(SEED)
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)

X, Y, n_three = load()
idx, h = split(X, Y)
tr, va, te = idx["train"], idx["val"], idx["test"]
leak, n_naive_test = naive_leakage(h)
print(f"maps train/val/test: {len(tr)}/{len(va)}/{len(te)} | value-3 pixels recoded: {n_three}")
print(f"naive random split would leave {leak} of {n_naive_test} test maps with an identical twin in training")


def metrics(yt, yp):
    per = {p: {"precision": round(float(precision_score(yt[:, j], yp[:, j], zero_division=0)), 3),
               "recall": round(float(recall_score(yt[:, j], yp[:, j], zero_division=0)), 3),
               "f1": round(float(f1_score(yt[:, j], yp[:, j], zero_division=0)), 3),
               "support": int(yt[:, j].sum())} for j, p in enumerate(PATTERNS)}
    return {"macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            "exact_match": round(float((yt == yp).all(axis=1).mean()), 4), "per_pattern": per}


def bootstrap(yt, yp, n=1000):
    rng = np.random.default_rng(SEED); f, e = [], []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        f.append(f1_score(yt[i], yp[i], average="macro", zero_division=0)); e.append((yt[i] == yp[i]).all(axis=1).mean())
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"macro_f1_95ci": q(f), "exact_match_95ci": q(e)}


results = {"data": {"train": len(tr), "val": len(va), "test": len(te), "value3_pixels_recoded": n_three,
                    "naive_split_test_maps_with_train_twin": leak, "naive_split_test_size": n_naive_test}}

# ---- B0: no skill (always predict "absent"; every pattern is a minority class) ----
prev = Y[tr].mean(axis=0)
b0 = np.tile((prev > 0.5).astype(int), (len(te), 1))
results["B0_no_skill"] = metrics(Y[te], b0)

# ---- B1: logistic regression per pattern on flattened pixels ----
flat = lambda i: (X[i].reshape(len(i), -1) / 2.0).astype(np.float32)
t0 = time.time(); b1 = np.zeros((len(te), 8), dtype=int)
for j in range(8):
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(flat(tr), Y[tr, j]); b1[:, j] = clf.predict(flat(te))
results["B1_logistic"] = metrics(Y[te], b1)
print(f"B1 done in {time.time() - t0:.0f}s: macro F1 {results['B1_logistic']['macro_f1']}")

# ---- M1: CNN ----
def onehot(i):                                   # 3 channels: off-wafer, good die, defective die
    x = torch.from_numpy(X[i].astype(np.int64))
    return nn.functional.one_hot(x, 3).permute(0, 3, 1, 2).float()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128))   # 52 -> 26 -> 13 -> 6
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 6 * 6, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 8))
    def forward(self, x): return self.h(self.f(x))


Xtr, Ytr = onehot(tr), torch.from_numpy(Y[tr]).float()
Xva, Xte = onehot(va), onehot(te)
net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.BCEWithLogitsLoss()


def predict(x):
    net.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(x), 512): out.append(torch.sigmoid(net(x[k:k + 512])))
    return (torch.cat(out).numpy() >= 0.5).astype(int)


best, best_state, wait, log = -1, None, 0, []
for ep in range(1, 26):
    net.train(); perm = torch.randperm(len(Xtr)); t0 = time.time()
    for k in range(0, len(perm), 128):
        b = perm[k:k + 128]; opt.zero_grad(); l = lossf(net(Xtr[b]), Ytr[b]); l.backward(); opt.step()
    vf = f1_score(Y[va], predict(Xva), average="macro", zero_division=0)
    log.append({"epoch": ep, "val_macro_f1": round(float(vf), 4), "secs": round(time.time() - t0)})
    print(f"epoch {ep}: val macro F1 {vf:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if vf > best: best, best_state, wait = vf, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= 4: break
net.load_state_dict(best_state)
m1 = predict(Xte)                                # the single test-set scoring
results["M1_cnn"] = metrics(Y[te], m1)
results["M1_cnn"]["bootstrap"] = bootstrap(Y[te], m1)
results["M1_cnn"]["best_val_macro_f1"] = round(float(best), 4)
results["M1_cnn"]["training_log"] = log
torch.save(best_state, OUT / "cnn_state.pt")

json.dump(results, open(OUT / "metrics.json", "w"), indent=2)
for k in ["B0_no_skill", "B1_logistic", "M1_cnn"]:
    print(f"{k}: macro F1 {results[k]['macro_f1']} | exact match {results[k]['exact_match']}")
print("M1 bootstrap:", results["M1_cnn"]["bootstrap"])
