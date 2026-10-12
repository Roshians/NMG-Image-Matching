#!/usr/bin/env python3
"""
Batch evaluation: run the matching pipeline on every pair in a CSV file.

Usage:
    python evaluate.py --data data --pairs data/pairs.csv --out results.csv

Writes results.csv with columns:
    image_a, image_b, label, pair_type, keypoints_a, keypoints_b,
    total_matches, good_matches, inliers, inlier_ratio,
    confidence, band, time_ms

When labels are known (not 'unknown'), also prints an accuracy table.
"""

import argparse
import csv
import os
import time

from match import run_pipeline


# CSV columns for output
RESULT_COLUMNS = [
    "image_a", "image_b", "label", "pair_type",
    "keypoints_a", "keypoints_b",
    "total_matches", "good_matches", "inliers", "inlier_ratio",
    "confidence", "band", "time_ms",
]


def load_pairs(pairs_path: str) -> list[dict]:
    """Read the pairs CSV into a list of dicts."""
    with open(pairs_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_results(results: list[dict], out_path: str) -> None:
    """Write results to CSV."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute accuracy metrics when labels are known.

    Returns dict with:
        same_correct, same_total,
        diff_correct, diff_total,
        lookalike_false_same, lookalike_total,
        breakdown — per pair_type stats
    """
    same_correct = 0
    same_total = 0
    diff_correct = 0
    diff_total = 0
    lookalike_false_same = 0
    lookalike_total = 0

    # Per pair_type breakdown
    breakdown = {}

    for r in results:
        label = r.get("label", "unknown")
        pair_type = r.get("pair_type", "unknown")
        band = r.get("band", "DIFFERENT")

        if label == "unknown":
            continue

        # Init breakdown entry
        if pair_type not in breakdown:
            breakdown[pair_type] = {"total": 0, "correct": 0, "false_same": 0}
        breakdown[pair_type]["total"] += 1

        if label == "same":
            same_total += 1
            if band == "SAME":
                same_correct += 1
                breakdown[pair_type]["correct"] += 1
        else:
            diff_total += 1
            if band in ("DIFFERENT", "SIMILAR"):
                diff_correct += 1
                breakdown[pair_type]["correct"] += 1
            if band == "SAME":
                breakdown[pair_type]["false_same"] += 1
                if pair_type == "lookalike":
                    lookalike_false_same += 1

            if pair_type == "lookalike":
                lookalike_total += 1

    return {
        "same_correct": same_correct,
        "same_total": same_total,
        "diff_correct": diff_correct,
        "diff_total": diff_total,
        "lookalike_false_same": lookalike_false_same,
        "lookalike_total": lookalike_total,
        "breakdown": breakdown,
    }


def print_metrics(metrics: dict) -> None:
    """Print a formatted accuracy table."""
    print("\n" + "=" * 65)
    print("  ACCURACY RESULTS")
    print("=" * 65)

    sc = metrics["same_correct"]
    st = metrics["same_total"]
    dc = metrics["diff_correct"]
    dt = metrics["diff_total"]
    lfs = metrics["lookalike_false_same"]
    lt = metrics["lookalike_total"]

    print(f"\n  Same pairs classified SAME:           {sc} / {st}")
    print(f"  Different pairs classified non-SAME:   {dc} / {dt}")
    print(f"  Lookalike pairs wrongly SAME:          {lfs} / {lt}  {'✓' if lfs == 0 else '✗ PROBLEM!'}")

    print(f"\n  {'Pair Type':<25} {'Total':>6} {'Correct':>8} {'False SAME':>11}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*11}")
    for pt, stats in sorted(metrics["breakdown"].items()):
        print(f"  {pt:<25} {stats['total']:>6} {stats['correct']:>8} {stats['false_same']:>11}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate matching pipeline on a set of image pairs."
    )
    parser.add_argument("--data", required=True, help="Base data directory (contains images/)")
    parser.add_argument("--pairs", required=True, help="Path to pairs CSV")
    parser.add_argument("--out", default="results.csv", help="Output results CSV path")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    print(f"\nLoaded {len(pairs)} pairs from {args.pairs}")

    images_dir = os.path.join(args.data, "images")
    output_dir = os.path.join("output")

    results = []
    has_labels = False
    total_start = time.time()

    for i, pair in enumerate(pairs):
        image_a = pair["image_a"]
        image_b = pair["image_b"]
        label = pair.get("label", "unknown")
        pair_type = pair.get("pair_type", "unknown")

        if label != "unknown":
            has_labels = True

        path_a = os.path.join(images_dir, image_a)
        path_b = os.path.join(images_dir, image_b)

        # Run pipeline
        result = run_pipeline(path_a, path_b, output_dir)

        # Add label info from pairs CSV
        result["label"] = label
        result["pair_type"] = pair_type

        results.append(result)

        # Progress
        status = "✓" if (label == "same" and result["band"] == "SAME") or \
                        (label == "different" and result["band"] != "SAME") or \
                        label == "unknown" else "✗"
        print(f"  [{i+1:3d}/{len(pairs)}] {status}  {image_a} vs {image_b}  "
              f"→ {result['band']:>9}  inliers={result['inliers']:>4}  "
              f"conf={result['confidence']:>5.1f}%  ({result['time_ms']:.0f}ms)")

    total_elapsed = time.time() - total_start

    # Write results CSV
    write_results(results, args.out)
    print(f"\nResults saved to {args.out}")
    print(f"Total time: {total_elapsed:.1f}s  ({total_elapsed/len(pairs)*1000:.0f} ms/pair)")

    # Print metrics if labels are known
    if has_labels:
        metrics = compute_metrics(results)
        print_metrics(metrics)


if __name__ == "__main__":
    main()
