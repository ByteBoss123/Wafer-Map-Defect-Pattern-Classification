# Data source

**MixedWM38** wafer-map defect-pattern dataset, introduced by Wang et al. (2020), "Deformable
Convolutional Networks for Efficient Mixed-Type Wafer Defect Pattern Recognition", IEEE Transactions
on Semiconductor Manufacturing.

The paper and the dataset's original host were not reachable from the build environment. The file
`Wafer_Map_Datasets.npz` was taken from the public GitHub repository
`github.com/Fmohammadsofi/ViTTinyMixed-Defect-Wafer-Maps` (file `Wafer_Map_Datasets.zip`, commit
`5cdfcd1fbefa1d1e0c1f45861ff51b7ade12294e`). The zip (20 MB) is committed here; the extracted `.npz`
(394 MB) exceeds GitHub's file limit, so extract it with `unzip data/raw/Wafer_Map_Datasets.zip -d data/raw`.
Only one copy could be found, so it could not be cross-checked against a second mirror; SHA-256 hashes
of the zip, the extracted `.npz` and `Description.pdf` are in `SHA256SUMS`.

## Checks against the published description

| Property | This file |
|---|---|
| Maps | 38,015, each 52 x 52 |
| Labels | 8 binary pattern columns, 38 distinct combinations |
| Defects per map | 0: 1,000 maps; 1: 7,015; 2: 13,000; 3: 13,000; 4: 4,000 |
| Maps per pattern | Center 13,000; Donut 12,000; Edge-Loc 13,000; Edge-Ring 12,000; Loc 18,000; Near-full 149; Scratch 19,000; Random 866 |
| Pixel coding | 0 off-wafer, 1 good die, 2 defective die; plus 214 pixels coded 3 in 105 maps (undocumented) |
| Duplicates | 1,023 groups of byte-identical maps (2,047 maps), no conflicting labels |

## How the maps were made (from the dataset's own description)

The dataset description shipped with the source repository (`Description.pdf`) states that the maps
were collected at a wafer manufacturing plant from electrical die tests, and that **a generative
adversarial network was used to generate some maps to balance the number of samples across types**.
So part of MixedWM38 is synthetic, and the description does not say which maps. Results here describe
performance on this benchmark, not on raw fab data alone.

The same description gives the label column order used in this repo: Center, Donut, Edge_Loc,
Edge_Ring, Loc, Near_Full, Scratch, Random (single-type labels `[1 0 0 0 0 0 0 0]` to
`[0 0 0 0 0 0 0 1]`), consistent with the per-pattern counts above.
