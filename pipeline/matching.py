"""
Steps 3–4: Feature Matching + Lowe's Ratio Test.

FLANN (Fast Library for Approximate Nearest Neighbours) with a KD-tree
index for SIFT/RootSIFT (float) descriptors, or LSH for binary
descriptors (ORB/AKAZE).

Lowe's ratio test keeps a match only when the best candidate is
significantly closer than the second-best.  ratio=0.75 is validated
by the dataset benchmark as the sweet spot between recall and
precision on stone textures.
"""

import cv2
import numpy as np

# FLANN algorithm IDs
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6


def create_flann_matcher(descriptor_type: str = "sift") -> cv2.FlannBasedMatcher:
    """
    Create a FLANN-based matcher tuned for the descriptor type.

    SIFT / RootSIFT → KD-tree (L2 distance, 5 trees)
    ORB / AKAZE      → LSH    (Hamming distance)
    """
    if descriptor_type in ("sift", "rootsift"):
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
    else:
        # Binary descriptors — Locality Sensitive Hashing
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )
        search_params = dict(checks=50)

    return cv2.FlannBasedMatcher(index_params, search_params)


def create_bf_matcher(descriptor_type: str = "sift") -> cv2.BFMatcher:
    """
    Brute-force matcher (alternative to FLANN, used for ORB bonus).

    Uses L2 norm for float descriptors, Hamming for binary.
    """
    if descriptor_type in ("sift", "rootsift"):
        return cv2.BFMatcher(cv2.NORM_L2)
    else:
        return cv2.BFMatcher(cv2.NORM_HAMMING)


def match_descriptors(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    matcher: cv2.DescriptorMatcher,
) -> list:
    """
    k-nearest-neighbour match with k=2 (needed for ratio test).

    Edge cases:
      - Either descriptor array is None → empty list
      - Fewer than 2 descriptors in one set → knnMatch may fail
    """
    if desc_a is None or desc_b is None:
        return []
    if len(desc_a) < 2 or len(desc_b) < 2:
        return []

    try:
        matches = matcher.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return []
    return matches


def apply_ratio_test(matches: list, ratio: float = 0.75) -> list[cv2.DMatch]:
    """
    Lowe's ratio test.

    Keep match m only if  m.distance < ratio × n.distance,
    where n is the second-nearest neighbour.

    ratio = 0.75:
      - Lowe's original recommendation was 0.7–0.8
      - 0.75 is the benchmark-validated sweet spot for this dataset
      - Filters ~70-80 % of raw matches
    """
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def get_good_matches(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    ratio: float = 0.75,
    descriptor_type: str = "sift",
    use_flann: bool = True,
) -> tuple[list[cv2.DMatch], int]:
    """
    High-level: create matcher → knnMatch → ratio test.

    Returns:
        (good_matches, total_raw_matches)
    """
    if use_flann:
        matcher = create_flann_matcher(descriptor_type)
    else:
        matcher = create_bf_matcher(descriptor_type)

    raw_matches = match_descriptors(desc_a, desc_b, matcher)
    good_matches = apply_ratio_test(raw_matches, ratio)

    return good_matches, len(raw_matches)
