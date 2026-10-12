#!/usr/bin/env python3
"""
Compare two images and determine if they show the same physical object.

Usage:
    python match.py --image-a data/images/img_001.jpg \
                    --image-b data/images/img_002.jpg \
                    --out output/

Prints:
    keypoints A and B, total matches, good matches, inliers,
    inlier ratio, confidence (%), band (SAME / SIMILAR / DIFFERENT), time_ms

Saves:
    output/{stem_a}__{stem_b}.jpg  — inlier match visualisation
"""

import argparse
import os
import time

from pipeline.preprocessing import preprocess
from pipeline.features import get_features
from pipeline.matching import get_good_matches
from pipeline.geometric import geometric_verify
from pipeline.scoring import score_pair
from pipeline.visualization import (
    draw_inlier_matches,
    save_visualization,
    print_result_summary,
)


def run_pipeline(
    path_a: str,
    path_b: str,
    output_dir: str = "output",
    use_rootsift: bool = True,
    use_clahe: bool = True,
    ratio: float = 0.75,
    detector_type: str = "sift",
) -> dict:
    """
    Run the full matching pipeline on a single image pair.

    Returns a result dict with all metrics, or a zeroed-out dict on failure.
    """
    start = time.time()

    # ── Step 1: Preprocess ────────────────────────────────────────
    color_a, gray_a = preprocess(path_a, use_clahe=use_clahe)
    color_b, gray_b = preprocess(path_b, use_clahe=use_clahe)

    if gray_a is None or gray_b is None:
        elapsed = (time.time() - start) * 1000
        return _empty_result(path_a, path_b, elapsed)

    # ── Step 2: Extract features ──────────────────────────────────
    kp_a, desc_a, n_kp_a = get_features(gray_a, use_rootsift=use_rootsift, detector_type=detector_type)
    kp_b, desc_b, n_kp_b = get_features(gray_b, use_rootsift=use_rootsift, detector_type=detector_type)

    # ── Steps 3–4: Match + ratio test ────────────────────────────
    use_flann = (detector_type == "sift")
    good_matches, total_matches = get_good_matches(
        desc_a, desc_b,
        ratio=ratio,
        descriptor_type=detector_type,
        use_flann=use_flann,
    )

    # ── Step 5: Geometric verification ───────────────────────────
    geo = geometric_verify(kp_a, kp_b, good_matches)

    # ── Step 6: Score + classify ─────────────────────────────────
    score = score_pair(geo)

    elapsed = (time.time() - start) * 1000

    # ── Build result dict ────────────────────────────────────────
    result = {
        "image_a": os.path.basename(path_a),
        "image_b": os.path.basename(path_b),
        "keypoints_a": n_kp_a,
        "keypoints_b": n_kp_b,
        "total_matches": total_matches,
        "good_matches": len(good_matches),
        "inliers": geo["inliers"],
        "inlier_ratio": geo["inlier_ratio"],
        "confidence": score["confidence"],
        "band": score["band"],
        "time_ms": elapsed,
    }

    # ── Visualise + save ─────────────────────────────────────────
    vis = draw_inlier_matches(
        color_a, kp_a, color_b, kp_b,
        geo["inlier_matches"],
        score["band"],
        score["confidence"],
    )
    saved_path = save_visualization(vis, path_a, path_b, output_dir)
    result["visualization_path"] = saved_path

    return result


def _empty_result(path_a: str, path_b: str, time_ms: float) -> dict:
    """Return a zeroed-out result when an image fails to load."""
    return {
        "image_a": os.path.basename(path_a),
        "image_b": os.path.basename(path_b),
        "keypoints_a": 0,
        "keypoints_b": 0,
        "total_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "confidence": 0.0,
        "band": "DIFFERENT",
        "time_ms": time_ms,
        "visualization_path": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two images for same-object matching."
    )
    parser.add_argument("--image-a", required=True, help="Path to first image")
    parser.add_argument("--image-b", required=True, help="Path to second image")
    parser.add_argument("--out", default="output/", help="Output directory for visualisation")
    args = parser.parse_args()

    print(f"\nComparing:\n  A: {args.image_a}\n  B: {args.image_b}\n")

    result = run_pipeline(args.image_a, args.image_b, args.out)

    print("Results:")
    print_result_summary(result)

    if result.get("visualization_path"):
        print(f"\n  Saved: {result['visualization_path']}")
    print()


if __name__ == "__main__":
    main()
