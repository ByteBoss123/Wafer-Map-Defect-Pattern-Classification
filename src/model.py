"""CNN architecture (identical to the one trained in src/train.py) and a loader for the saved weights."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PATTERNS = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Scratch", "Random"]
WEIGHTS = Path(__file__).resolve().parents[1] / "results" / "cnn_state.pt"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128))   # 52 -> 26 -> 13 -> 6
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 6 * 6, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 8))
    def forward(self, x): return self.h(self.f(x))


def load_model(path=WEIGHTS):
    net = Net(); net.load_state_dict(torch.load(path, map_location="cpu")); return net.eval()


def encode(maps):
    """(N, 52, 52) or (52, 52) array coded 0 off-wafer, 1 good die, 2 defective die -> (N, 3, 52, 52) one-hot."""
    m = np.asarray(maps)
    if m.ndim == 2: m = m[None]
    assert m.shape[1:] == (52, 52), "maps must be 52 x 52"
    m = np.where(m == 3, 2, m).astype(np.int64)
    return nn.functional.one_hot(torch.from_numpy(m), 3).permute(0, 3, 1, 2).float()


@torch.no_grad()
def predict_proba(maps, net=None):
    net = net or load_model(); x = encode(maps)
    return torch.cat([torch.sigmoid(net(x[k:k + 512])) for k in range(0, len(x), 512)]).numpy()
