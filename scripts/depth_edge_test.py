#!/usr/bin/env python3

import glob
import os
import sys

import cv2
import numpy as np

DIFF_TOL_M = 0.01          # 1 cm == the uint16 quantisation step
EXPLAINED_THRESHOLD = 0.95  # 95% of differing pixels must be edge-explainable
MAX_DIFFERING_FRAC = 0.01   # and no more than 1% of pixels may differ at all


def neighbourhood_bounds(img, k=3):
    """Per-pixel min and max over a kxk window."""
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(img, kernel), cv2.dilate(img, kernel)


def analyse(pa, pb, verbose=True):
    a = cv2.imread(pa, cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
    b = cv2.imread(pb, cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0

    diff = np.abs(a - b)
    differing = diff > DIFF_TOL_M
    n_diff = int(differing.sum())
    frac = n_diff / a.size

    if n_diff == 0:
        if verbose:
            print("    identical")
        return True, 0.0, 1.0

    # Explainable in either direction: B's value present near A's pixel, or
    # A's value present near B's. Either means the silhouette shifted rather
    # than the geometry moving.
    a_lo, a_hi = neighbourhood_bounds(a)
    b_lo, b_hi = neighbourhood_bounds(b)
    tol = 0.02
    explained = (((b >= a_lo - tol) & (b <= a_hi + tol)) |
                 ((a >= b_lo - tol) & (a <= b_hi + tol)))
    expl_frac = float(explained[differing].mean())

    # Contiguity: unexplained pixels forming blobs indicate moved geometry;
    # threads one pixel wide indicate outlines.
    unexplained = (differing & ~explained).astype(np.uint8)
    n_unexp = int(unexplained.sum())
    largest_blob = 0
    if n_unexp:
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(unexplained, 8)
        if n_lab > 1:
            largest_blob = int(stats[1:, cv2.CC_STAT_AREA].max())

    if verbose:
        print(f"    differing pixels : {n_diff:7d}  ({frac*100:.3f}%)")
        print(f"    edge-explainable : {expl_frac*100:6.2f}%")
        print(
            f"    unexplained      : {n_unexp:7d}  largest blob {largest_blob} px")

    return (expl_frac >= EXPLAINED_THRESHOLD and frac <= MAX_DIFFERING_FRAC
            and largest_blob < 50), frac, expl_frac


def main():
    a_dir = sys.argv[1] if len(sys.argv) > 1 else "./run_a"
    b_dir = sys.argv[2] if len(sys.argv) > 2 else "./run_b"

    files = sorted(os.path.basename(f)
                   for f in glob.glob(os.path.join(a_dir, "depth", "*.png")))
    if not files:
        sys.exit(f"No depth PNGs in {a_dir}/depth")

    n = len(files)
    sample = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    print(f"Testing {len(sample)} of {n} frames\n")

    all_ok = True
    worst_frac, worst_expl = 0.0, 1.0
    for i in sample:
        name = files[i]
        print(f"  frame {name}")
        ok, frac, expl = analyse(os.path.join(a_dir, "depth", name),
                                 os.path.join(b_dir, "depth", name))
        all_ok &= ok
        worst_frac = max(worst_frac, frac)
        worst_expl = min(worst_expl, expl)
        print()

    print("=" * 62)
    if all_ok:
        print("VERDICT: silhouette rasterisation noise.")
        print(f"  Worst frame: {worst_frac*100:.3f}% of pixels differ, "
              f"{worst_expl*100:.1f}% edge-explainable.")
        print()
        print("  Differences are one-pixel threads along object outlines, where")
        print("  the depth value was already present in the immediate")
        print("  neighbourhood. Geometry did not move -- ego and NPC poses are")
        print("  bit-identical, which this test does not contradict.")
        print()
        print("  Impact on this project: none. true_range comes from actor")
        print("  bounding boxes, not the depth buffer. Depth feeds only the")
        print("  visibility mask, which averages over a whole box with a")
        print("  tolerance of max(1.5 m, range x 10%).")
        return 0

    print("VERDICT: geometry genuinely differs.")
    print(f"  Worst frame: {worst_frac*100:.3f}% of pixels differ, only "
          f"{worst_expl*100:.1f}% edge-explainable.")
    print()
    print("  Unexplained pixels in contiguous blobs mean something moved that")
    print("  is not an ego or NPC vehicle. Candidates, in order:")
    print("    - pedestrians (not captured in meta['actors'], which filters")
    print("      vehicle.* only)")
    print("    - map layers that failed to unload (the unload is wrapped in")
    print("      try/except and fails silently on non-_Opt maps)")
    print("    - traffic light or prop state carried over between runs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
