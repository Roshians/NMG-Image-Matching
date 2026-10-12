"""
Step 5: Geometric Verification — Homography estimation via USAC_MAGSAC.

USAC_MAGSAC (Baráth et al., 2019) marginalises over the noise scale σ
instead of requiring a fixed inlier threshold, giving cleaner
inlier / outlier separation than plain RANSAC.  Available in
OpenCV ≥ 4.5 as cv2.USAC_MAGSAC.

Sanity checks reject degenerate homographies whose determinant is
near-zero (collapsed) or very large (explosion), which can happen
when RANSAC overfits on coincidental texture matches in lookalike pairs.
"""

import cv2
import numpy as np


def find_homography(
    kp_a: list[cv2.KeyPoint],
    kp_b: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    method: int = cv2.USAC_MAGSAC,
    reproj_thresh: float = 5.0,
    max_iters: int = 2000,
    confidence: float = 0.995,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate homography from matched keypoints.

    Returns:
        (H, inlier_mask) or (None, None) when fewer than 4 matches
        are available (minimum for a homography).
    """
    if len(good_matches) < 4:
        return None, None

    # Extract matched point coordinates
    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        method,
        ransacReprojThreshold=reproj_thresh,
        maxIters=max_iters,
        confidence=confidence,
    )
    return H, mask


def validate_homography(H: np.ndarray | None) -> bool:
    """
    Reject degenerate homographies.

    Checks:
      1. H exists and is 3×3
      2. |det(H)| > 0.05  — not collapsed to a line/point
      3. |det(H)| < 20    — not an explosion (unrealistic scale change)
      4. H[2,2] ≠ 0       — projective component is finite

    These thresholds are generous; real stone-to-stone transforms
    (rotation, mild perspective) have det ≈ 0.5–2.0.
    """
    if H is None:
        return False
    if H.shape != (3, 3):
        return False

    det = np.linalg.det(H)
    if abs(det) < 0.05 or abs(det) > 20:
        return False
    if abs(H[2, 2]) < 1e-6:
        return False

    return True


def compute_inlier_metrics(
    inlier_mask: np.ndarray | None,
    total_good_matches: int,
) -> tuple[int, float]:
    """
    Count inliers and compute inlier ratio.

    Returns:
        (inlier_count, inlier_ratio)
    """
    if inlier_mask is None:
        return 0, 0.0

    inlier_count = int(inlier_mask.ravel().sum())
    inlier_ratio = inlier_count / total_good_matches if total_good_matches > 0 else 0.0
    return inlier_count, inlier_ratio


def geometric_verify(
    kp_a: list[cv2.KeyPoint],
    kp_b: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    method: int = cv2.USAC_MAGSAC,
    reproj_thresh: float = 5.0,
) -> dict:
    """
    Full geometric verification pipeline.

    Returns dict:
        homography     — 3×3 matrix or None
        inlier_mask    — numpy array or None
        inliers        — int
        inlier_ratio   — float
        is_valid       — bool (passed sanity checks)
        inlier_matches — list[DMatch], only the inlier subset
    """
    H, mask = find_homography(kp_a, kp_b, good_matches, method, reproj_thresh)
    is_valid = validate_homography(H)
    inliers, inlier_ratio = compute_inlier_metrics(mask, len(good_matches))

    # If homography is invalid, zero out the inlier evidence
    if not is_valid:
        inliers = 0
        inlier_ratio = 0.0

    # Extract only the inlier matches for visualisation
    inlier_matches = []
    if mask is not None and is_valid:
        mask_flat = mask.ravel()
        inlier_matches = [m for m, flag in zip(good_matches, mask_flat) if flag]

    return {
        "homography": H,
        "inlier_mask": mask,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "is_valid": is_valid,
        "inlier_matches": inlier_matches,
    }
