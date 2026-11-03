"""Make results/examples.png: one test map per single-pattern class, with the CNN's predicted labels."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import PATTERNS, ROOT, load, split

class Net(nn.Module):  # same architecture as src/train.py
    def __init__(self):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 6 * 6, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 8))
    def forward(self, x): return self.h(self.f(x))

X, Y, _ = load(); te = split(X, Y)[0]["test"]
net = Net(); net.load_state_dict(torch.load(ROOT / "results" / "cnn_state.pt")); net.eval()
rng = np.random.default_rng(0); picks = []
for j in range(8):                                   # a random single-pattern test map per class
    c = te[(Y[te].sum(1) == 1) & (Y[te, j] == 1)]; picks.append(rng.choice(c))
for combo in ([1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1, 0]):   # two mixed-type examples
    c = te[(Y[te] == combo).all(1)]; picks.append(rng.choice(c))
x = nn.functional.one_hot(torch.from_numpy(X[picks].astype(np.int64)), 3).permute(0, 3, 1, 2).float()
with torch.no_grad(): P = (torch.sigmoid(net(x)).numpy() >= 0.5).astype(int)
name = lambda v: " + ".join(p.replace("_", "-") for p, b in zip(PATTERNS, v) if b) or "none"
fig, ax = plt.subplots(2, 5, figsize=(14, 7.4))
cm = ListedColormap(["#ffffff", "#9ecae1", "#d62728"])
for a, i, p in zip(ax.flat, picks, P):
    a.imshow(X[i], cmap=cm, vmin=0, vmax=2, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
    ok = (p == Y[i]).all()
    a.set_title(f"true: {name(Y[i])}\npred: {name(p)}", fontsize=9, color="black" if ok else "#d62728")
fig.suptitle("Random held-out test maps (blue = good die, red = defective die)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.subplots_adjust(hspace=0.35); fig.savefig(ROOT / "results" / "examples.png", dpi=110)
print("correct:", int((P == Y[picks]).all(1).sum()), "of", len(picks))
