# Plan (written before any model was trained)

**Task:** detect which of 8 defect patterns appear on a wafer map (multi-label: a map can have 0 to 4).
**Data:** MixedWM38, 38,015 maps (52 x 52), 8 binary pattern labels. See `data/raw/SOURCE.md`.

## Data handling
- Pixel value 3 (214 pixels in 105 maps, outside the documented 0/1/2 coding) is treated as 2 (defective die).
- 1,023 groups of byte-identical maps (2,047 maps, no label conflicts) are kept together: every copy of a
  map goes to the same split, so no test map has an identical twin in training.
- Split 70 / 15 / 15 (train / validation / test), stratified by the 38 label combinations, seed 42.
  **The test set is scored once, after all choices are fixed.**

## Models
| ID | Model |
|---|---|
| B0 | No skill: predicts each pattern's training prevalence (reported as F1 of predicting "absent") |
| B1 | Logistic regression, one per pattern, on the flattened 2,704-pixel map |
| M1 | CNN (PyTorch, CPU): 3 conv blocks (32/64/128 filters), dropout 0.3, 8 sigmoid outputs, BCE loss, Adam 1e-3, batch 128, up to 25 epochs, early stopping on validation macro F1 (patience 4) |

Decision threshold fixed at 0.5 for every pattern (no threshold tuning).

## Metrics (test set)
Macro F1 over the 8 patterns (primary), per-pattern precision/recall/F1, exact-match accuracy (all 8
labels correct). 95% bootstrap interval (1,000 resamples of test maps) for macro F1 and exact match.

## Secondary (no model involved)
How many test maps would have an identical twin in training under a naive random split that ignores
duplicates, to show the leakage risk this split avoids.
