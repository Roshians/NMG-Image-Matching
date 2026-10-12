"""
Step 6b: Visualisation — draw inlier matches and print result summaries.
"""

import os

import cv2
import numpy as np


# Colours (BGR)
BAND_COLOURS = {
    "SAME": (0, 200, 0),       # green
    "SIMILAR": (0, 200, 255),  # yellow-orange
    "DIFFERENT": (0, 0, 220),  # red
}


def draw_inlier_matches(
    img_a: np.ndarray,
    kp_a: list[cv2.KeyPoint],
    img_b: np.ndarray,
    kp_b: list[cv2.KeyPoint],
    inlier_matches: list[cv2.DMatch],
    band: str,
    confidence: float = 0.0,
) -> np.ndarray:
    """
    Draw ONLY inlier matches (not all matches) between two images,
    colour-coded by classification band.

    Adds a text overlay with inlier count, confidence, and band.
    """
    colour = BAND_COLOURS.get(band, (200, 200, 200))

    vis = cv2.drawMatches(
        img_a,
        kp_a,
        img_b,
        kp_b,
        inlier_matches,
        None,
        matchColor=colour,
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Text overlay — top-left
    label = f"{band} | Inliers: {len(inlier_matches)} | Conf: {confidence:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Background rectangle for readability
    cv2.rectangle(vis, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
    cv2.putText(vis, label, (10, 10 + th), font, font_scale, colour, thickness)

    return vis


def save_visualization(
    vis_image: np.ndarray,
    name_a: str,
    name_b: str,
    output_dir: str = "output",
) -> str:
    """
    Save the visualisation image.

    File name: output/{stem_a}__{stem_b}.jpg
    Creates the output directory if needed.
    Returns the saved filepath.
    """
    os.makedirs(output_dir, exist_ok=True)

    stem_a = os.path.splitext(os.path.basename(name_a))[0]
    stem_b = os.path.splitext(os.path.basename(name_b))[0]
    filename = f"{stem_a}__{stem_b}.jpg"
    filepath = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, vis_image)
    return filepath


def print_result_summary(result: dict) -> None:
    """
    Print a formatted single-pair result to stdout.

    Expected keys in *result*:
        keypoints_a, keypoints_b, total_matches, good_matches,
        inliers, inlier_ratio, confidence, band, time_ms
    """
    print(f"  Keypoints A:    {result['keypoints_a']}")
    print(f"  Keypoints B:    {result['keypoints_b']}")
    print(f"  Total matches:  {result['total_matches']}")
    print(f"  Good matches:   {result['good_matches']}")
    print(f"  Inliers:        {result['inliers']}")
    print(f"  Inlier ratio:   {result['inlier_ratio']:.4f}")
    print(f"  Confidence:     {result['confidence']:.1f}%")
    print(f"  Band:           {result['band']}")
    print(f"  Time:           {result['time_ms']:.0f} ms")
