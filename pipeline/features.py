"""
Step 2: Feature Extraction — SIFT with optional RootSIFT enhancement.

RootSIFT (Arandjelović & Zisserman, 2012) L1-normalises each SIFT
descriptor then takes the element-wise square root.  This makes
standard L2 matching behave like Hellinger (Bhattacharyya) distance,
which is more discriminative for histogram-style descriptors — a free
accuracy gain at negligible cost.
"""

import cv2
import numpy as np


def create_sift_detector(
    n_features: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: int = 10,
    sigma: float = 1.6,
) -> cv2.SIFT:
    """
    Create SIFT detector with given parameters.

    Defaults are OpenCV's standard values:
      - n_features=0 → unlimited keypoints
      - contrastThreshold=0.04 → discard low-contrast
      - edgeThreshold=10 → discard edge-like responses
      - sigma=1.6 → Gaussian blur of input
    """
    return cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )


def create_orb_detector(n_features: int = 5000) -> cv2.ORB:
    """
    Create ORB detector (for bonus SIFT-vs-ORB comparison).
    Higher n_features to compensate for ORB's lower discriminability.
    """
    return cv2.ORB_create(nfeatures=n_features)


def extract_features(
    image_gray: np.ndarray,
    detector: cv2.Feature2D,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """
    Detect keypoints and compute descriptors.

    Returns:
        (keypoints, descriptors).  descriptors is None when no
        keypoints are found.
    """
    keypoints, descriptors = detector.detectAndCompute(image_gray, None)
    if keypoints is None or len(keypoints) == 0:
        return [], None
    return keypoints, descriptors


def apply_rootsift(descriptors: np.ndarray) -> np.ndarray:
    """
    RootSIFT transform: L1-normalise, then sqrt.

    After this transform, L2 distance between two RootSIFT vectors
    equals the Hellinger distance between the original SIFT histograms.
    """
    # L1 normalise each row (add eps to avoid division by zero)
    eps = 1e-7
    descriptors = descriptors / (np.sum(descriptors, axis=1, keepdims=True) + eps)
    # Element-wise square root
    descriptors = np.sqrt(descriptors)
    return descriptors.astype(np.float32)


def get_features(
    image_gray: np.ndarray,
    use_rootsift: bool = True,
    detector_type: str = "sift",
) -> tuple[list[cv2.KeyPoint], np.ndarray | None, int]:
    """
    High-level feature extraction.

    Args:
        image_gray:   preprocessed grayscale image
        use_rootsift: apply RootSIFT on SIFT descriptors (ignored for ORB)
        detector_type: "sift" or "orb"

    Returns:
        (keypoints, descriptors, keypoint_count)
    """
    if detector_type == "orb":
        detector = create_orb_detector()
    else:
        detector = create_sift_detector()

    keypoints, descriptors = extract_features(image_gray, detector)

    # Apply RootSIFT only to SIFT (float) descriptors
    if descriptors is not None and use_rootsift and detector_type == "sift":
        descriptors = apply_rootsift(descriptors)

    return keypoints, descriptors, len(keypoints)
