# NMG Image Matching Challenge

A classical feature-matching pipeline for stone slab identification. Determines whether two photographs show the **same physical stone piece** using local keypoint geometry — not colour or texture similarity.

## Approach

**SIFT → RootSIFT → FLANN (KD-Tree) → Lowe's ratio test (0.75) → USAC_MAGSAC homography → confidence scoring**

The pipeline extracts scale-invariant keypoints (SIFT), enhances descriptor discriminability (RootSIFT), matches them efficiently (FLANN), filters ambiguous matches (ratio test), validates geometric consistency (RANSAC), and classifies the pair into SAME / SIMILAR / DIFFERENT bands based on inlier count and ratio.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Compare two images
python match.py --image-a data/images/img_001.jpg --image-b data/images/img_002.jpg --out output/

# Run every pair in a pairs file
python evaluate.py --data data --pairs data/pairs.csv --out results.csv
```

## Results on Provided Pairs

| Metric | Result |
|--------|--------|
| Same pairs classified SAME | **10 / 10** |
| Different pairs classified non-SAME | **90 / 90** |
| Lookalike pairs wrongly SAME | **0 / 10** ✓ |
| Weakest same pair (inliers) | 124 |
| Strongest non-same pair (inliers) | 5 |
| Average time per pair | 170 ms |

## Project Structure

```
├── match.py              # Single-pair comparison CLI
├── evaluate.py           # Batch evaluation from pairs.csv
├── pipeline/
│   ├── preprocessing.py  # Load, resize, grayscale, CLAHE
│   ├── features.py       # SIFT / RootSIFT extraction
│   ├── matching.py       # FLANN matching + Lowe's ratio test
│   ├── geometric.py      # USAC_MAGSAC homography + sanity checks
│   ├── scoring.py        # Confidence formula + band classification
│   └── visualization.py  # Inlier match drawing
├── data/
│   ├── images/           # All reference + query images
│   └── pairs.csv         # 100 evaluation pairs
├── output/               # Match visualisations
├── results.csv           # Evaluation results
├── requirements.txt
└── REPORT.md             # Algorithm choices & analysis
```
