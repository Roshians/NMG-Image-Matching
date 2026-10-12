"""
Step 6a: Similarity Scoring & Band Classification.

Confidence formula combines inlier count (absolute evidence) with
inlier ratio (match quality).  Thresholds are derived from benchmark
data: weakest true-same pair has 128 inliers, strongest non-same
has 9.  The SAME threshold at 40 provides a 4× safety margin.
"""

# ── Tunable constants ────────────────────────────────────────────
INLIER_CEILING = 200       # Score saturates at this many inliers
SAME_INLIER_MIN = 40       # Minimum inliers to classify SAME
SAME_CONFIDENCE_MIN = 50.0  # Minimum confidence % for SAME
SIMILAR_INLIER_MIN = 10    # Minimum inliers for SIMILAR
SIMILAR_CONFIDENCE_MIN = 20.0  # Minimum confidence % for SIMILAR


def compute_confidence(
    inliers: int,
    inlier_ratio: float,
    is_valid_homography: bool,
) -> float:
    """
    Confidence score 0 – 100 %.

    Formula:
        count_part = min(inliers / INLIER_CEILING, 1.0)   # 0–1
        ratio_part = inlier_ratio                          # 0–1
        confidence = (count_part × 0.6 + ratio_part × 0.4) × 100

    Why two components:
      - Count  → rewards absolute evidence strength
      - Ratio  → rewards match precision / quality
      - 60/40  → count is the primary signal for stone matching
                  (a pair can have a high ratio with very few matches
                   due to random coincidence)

    Returns 0.0 when geometric verification failed.
    """
    if not is_valid_homography or inliers == 0:
        return 0.0

    count_part = min(inliers / INLIER_CEILING, 1.0)
    ratio_part = min(inlier_ratio, 1.0)

    raw = count_part * 0.6 + ratio_part * 0.4
    return round(raw * 100.0, 1)


def classify_band(inliers: int, confidence: float) -> str:
    """
    Three-band classification.

    SAME:      inliers ≥ 40  AND  confidence ≥ 50 %
    SIMILAR:   inliers ≥ 10  OR   confidence ≥ 20 %
    DIFFERENT: everything else

    Threshold rationale (from benchmark):
      - Weakest true-same:      128 inliers → easily clears 40
      - Strongest non-same:       9 inliers → well below 40
      - Safety margin:           40 / 9 ≈ 4.4×

    A lookalike pair must NEVER reach SAME.
    """
    if inliers >= SAME_INLIER_MIN and confidence >= SAME_CONFIDENCE_MIN:
        return "SAME"
    elif inliers >= SIMILAR_INLIER_MIN or confidence >= SIMILAR_CONFIDENCE_MIN:
        return "SIMILAR"
    else:
        return "DIFFERENT"


def score_pair(geo_result: dict) -> dict:
    """
    Compute confidence and band from geometric verification output.

    Args:
        geo_result: dict from geometric_verify()

    Returns:
        dict with 'confidence' (float) and 'band' (str)
    """
    confidence = compute_confidence(
        geo_result["inliers"],
        geo_result["inlier_ratio"],
        geo_result["is_valid"],
    )
    band = classify_band(geo_result["inliers"], confidence)
    return {"confidence": confidence, "band": band}
