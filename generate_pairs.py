#!/usr/bin/env python3
"""
Generate pairs.csv: 5 references × 20 queries = 100 pairs.
Labels derived from piece codes in filenames.
"""
import csv
import os
import re

REFERENCE_DIR = "Intern_Assignment_Dataset/reference"
QUERY_DIR = "Intern_Assignment_Dataset/queries"
OUTPUT = "data/pairs.csv"


def extract_piece_code(filename):
    """Extract piece code like B01A from filename."""
    match = re.search(r'__(B\d{2}[A-Z])', filename)
    return match.group(1) if match else None


def extract_condition(filename):
    """Extract condition like rotation_scale, viewpoint, lighting, occlusion."""
    # Pattern: img_XXX__PIECE_CONDITION_RELATION.jpg
    match = re.search(r'__B\d{2}[A-Z]_(.+?)_(same|lookalike|reference)', filename)
    return match.group(1) if match else None


def extract_relation(filename):
    """Extract relation: same, lookalike, or reference."""
    match = re.search(r'_(same|lookalike|reference)\.jpg$', filename)
    return match.group(1) if match else None


def get_block_id(piece_code):
    """Get block ID from piece code: B01A → B01."""
    return piece_code[:3] if piece_code else None


def main():
    references = sorted(os.listdir(REFERENCE_DIR))
    queries = sorted(os.listdir(QUERY_DIR))

    print(f"References: {len(references)}")
    print(f"Queries: {len(queries)}")

    # Build reference lookup: piece_code → filename
    ref_info = {}
    for ref in references:
        piece = extract_piece_code(ref)
        ref_info[ref] = piece
        print(f"  REF: {ref} → piece={piece}")

    # Build query lookup
    query_info = {}
    for q in queries:
        piece = extract_piece_code(q)
        condition = extract_condition(q)
        relation = extract_relation(q)
        query_info[q] = (piece, condition, relation)
        print(f"  QUERY: {q} → piece={piece}, condition={condition}, relation={relation}")

    # Generate all pairs
    rows = []
    pair_id = 1
    same_count = 0
    lookalike_count = 0
    different_count = 0

    for ref in references:
        ref_piece = ref_info[ref]
        ref_block = get_block_id(ref_piece)

        for query in queries:
            q_piece, q_condition, q_relation = query_info[query]
            q_block = get_block_id(q_piece)

            # Determine label: same piece code = same, else different
            if ref_piece == q_piece:
                label = "same"
                pair_type = q_condition  # rotation_scale, viewpoint, etc.
                same_count += 1
            elif ref_block == q_block:
                # Same block, different piece = lookalike
                label = "different"
                pair_type = "lookalike"
                lookalike_count += 1
            else:
                label = "different"
                pair_type = "different"
                different_count += 1

            rows.append({
                "pair_id": pair_id,
                "image_a": ref,
                "image_b": query,
                "label": label,
                "pair_type": pair_type,
            })
            pair_id += 1

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "image_a", "image_b", "label", "pair_type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nGenerated {OUTPUT} with {len(rows)} pairs:")
    print(f"  Same:      {same_count}")
    print(f"  Lookalike: {lookalike_count}")
    print(f"  Different: {different_count}")


if __name__ == "__main__":
    main()
