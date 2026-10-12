"""
Step 1: Image Preprocessing

Load, resize, convert to grayscale, and optionally enhance with CLAHE.
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray | None:
    """
    Read image from disk with OpenCV.

    Returns:
        BGR image array, or None if file is missing / unreadable.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"[WARNING] Could not load image: {path}")
    return img


def resize_image(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """
    Resize so the longest side equals *max_side*.
    If already smaller, return unchanged.
    Uses INTER_AREA for downscaling (best quality for shrinking).
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image

    scale = max_side / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR → grayscale.  If already single-channel, return as-is.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation.

    Improves local contrast, which helps SIFT find more keypoints
    in low-contrast stone regions (validated by benchmark: CLAHE ON
    produces better separation than raw grayscale).
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(gray)


def preprocess(
    path: str,
    max_side: int = 1024,
    use_clahe: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Full preprocessing pipeline.

    Returns:
        (color_resized, gray_processed) — colour kept for visualisation,
        grayscale used for feature extraction.
        Returns (None, None) on load failure.
    """
    color = load_image(path)
    if color is None:
        return None, None

    color = resize_image(color, max_side)
    gray = to_grayscale(color)

    if use_clahe:
        gray = apply_clahe(gray)

    return color, gray
