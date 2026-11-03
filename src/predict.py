"""Label the defect patterns on wafer maps with the trained CNN.

    python src/predict.py my_maps.npy      # .npy of shape (52, 52) or (N, 52, 52); 0 off-wafer, 1 good, 2 defective
"""
import sys

import numpy as np

from model import PATTERNS, load_model, predict_proba

if __name__ == "__main__":
    maps = np.load(sys.argv[1]); P = predict_proba(maps, load_model())
    for i, p in enumerate(P):
        found = [f"{n} ({v:.2f})" for n, v in zip(PATTERNS, p) if v >= 0.5]
        print(f"map {i}: {', '.join(found) or 'no pattern'}")
