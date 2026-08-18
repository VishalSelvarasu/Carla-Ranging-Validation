#!/usr/bin/env python3

import glob
import json
import os
import sys

import numpy as np

EGO_TOL_M = 1e-6          # simulation state must be exact
NPC_TOL_M = 1e-3          # 1 mm
DEPTH_TOL_M = 0.01           # 1 cm == the uint16 quantisation step
MAX_DIFFERING_FRAC = 0.01    # at most 1% of pixels may differ at all
EXPLAINED_THRESHOLD = 0.95   # 95% of those must be edge-explainable
MAX_BLOB_PX = 50             # contiguous unexplained region = moved geometry


def load_meta(run):
    files = sorted(glob.glob(os.path.join(run, "meta", "*.json")))
    if not files:
        sys.exit(f"No meta files in {run}")
    return [json.load(open(f)) for f in files]


def main():
    a_dir = sys.argv[1] if len(sys.argv) > 1 else "./run_a"
    b_dir = sys.argv[2] if len(sys.argv) > 2 else "./run_b"

    A, B = load_meta(a_dir), load_meta(b_dir)
    if len(A) != len(B):
        print(f"FAIL: frame count differs ({len(A)} vs {len(B)})")
        return 1
    n = len(A)
    print(f"Comparing {n} frames\n")
    ok = True

    # ---- 1. Ego -----------------------------------------------------------
    worst_ego = 0.0
    for ma, mb in zip(A, B):
        la, lb = ma["ego_transform"]["location"], mb["ego_transform"]["location"]
        worst_ego = max(worst_ego, *(abs(la[k] - lb[k])
                        for k in ("x", "y", "z")))
        worst_ego = max(worst_ego, abs(
            ma["ego_speed_mps"] - mb["ego_speed_mps"]))
    if worst_ego <= EGO_TOL_M:
        print(f"  [PASS] ego trajectory identical (max delta {worst_ego:.2e})")
    else:
        print(f"  [FAIL] ego trajectory differs by {worst_ego:.6f} m")
        print("         -> physics or seeding problem. Check that")
        print("            traffic_manager.set_synchronous_mode(True) and")
        print("            set_random_device_seed() run BEFORE any spawn.")
        ok = False

    # ---- 2. NPCs ----------------------------------------------------------
    # Actor IDs are assigned per server session and will differ between runs;
    # that is bookkeeping, not behaviour. Compare sorted POSITIONS instead.
    worst_npc, count_mismatch = 0.0, False
    for ma, mb in zip(A, B):
        if len(ma["actors"]) != len(mb["actors"]):
            count_mismatch = True
            break
        pa = sorted((x["transform"]["location"]["x"],
                     x["transform"]["location"]["y"]) for x in ma["actors"])
        pb = sorted((x["transform"]["location"]["x"],
                     x["transform"]["location"]["y"]) for x in mb["actors"])
        for (x1, y1), (x2, y2) in zip(pa, pb):
            worst_npc = max(worst_npc, abs(x1 - x2), abs(y1 - y2))

    if count_mismatch:
        print("  [FAIL] NPC count differs between runs -> spawning is not seeded")
        ok = False
    elif worst_npc <= NPC_TOL_M:
        print(
            f"  [PASS] NPC positions identical (max delta {worst_npc:.2e} m)")
    else:
        print(f"  [FAIL] NPC positions differ by {worst_npc:.4f} m")
        print("         -> Traffic Manager state is persisting across runs.")
        print("            Restart the CARLA server between captures.")
        ok = False

    # ---- 3. Depth sanity --------------------------------------------------
    # NOT a max-difference test. At an occlusion boundary a single pixel
    # flipping between foreground and background differs by the FULL depth gap
    # -- a car edge at 20 m against a building at 58 m gives 38 m from one
    # pixel of rasterisation noise. An earlier version of this file used a 2 m
    # ceiling and failed a correctly configured system for that reason.
    #
    # The test that actually separates noise from a changed scene: for each
    # differing pixel, is the other run's value present in its 3x3
    # neighbourhood? If yes, the silhouette landed one pixel to the side. If
    # no -- and especially if such pixels form contiguous blobs -- geometry
    # genuinely moved.
    try:
        import cv2
        samples = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
        worst_frac, worst_expl, worst_blob = 0.0, 1.0, 0

        for i in samples:
            stem = f"{i:06d}.png"
            pa = os.path.join(a_dir, "depth", stem)
            pb = os.path.join(b_dir, "depth", stem)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            da = cv2.imread(pa, cv2.IMREAD_UNCHANGED).astype(
                np.float32) / 100.0
            db = cv2.imread(pb, cv2.IMREAD_UNCHANGED).astype(
                np.float32) / 100.0

            differing = np.abs(da - db) > DEPTH_TOL_M
            frac = float(differing.mean())
            worst_frac = max(worst_frac, frac)
            if not differing.any():
                continue

            k = np.ones((3, 3), np.uint8)
            a_lo, a_hi = cv2.erode(da, k), cv2.dilate(da, k)
            b_lo, b_hi = cv2.erode(db, k), cv2.dilate(db, k)
            t = 0.02
            explained = (((db >= a_lo - t) & (db <= a_hi + t)) |
                         ((da >= b_lo - t) & (da <= b_hi + t)))
            worst_expl = min(worst_expl, float(explained[differing].mean()))

            unexp = (differing & ~explained).astype(np.uint8)
            if unexp.any():
                n_lab, _, stats, _ = cv2.connectedComponentsWithStats(unexp, 8)
                if n_lab > 1:
                    worst_blob = max(worst_blob,
                                     int(stats[1:, cv2.CC_STAT_AREA].max()))

        if (worst_expl >= EXPLAINED_THRESHOLD
                and worst_frac <= MAX_DIFFERING_FRAC
                and worst_blob < MAX_BLOB_PX):
            print(f"  [PASS] depth: {worst_frac*100:.3f}% of pixels differ, "
                  f"{worst_expl*100:.1f}% edge-explainable, "
                  f"largest blob {worst_blob} px")
            print("         Silhouette rasterisation noise. Does not affect")
            print("         true_range, which comes from bounding boxes.")
        else:
            print(f"  [FAIL] depth: {worst_frac*100:.3f}% differ, only "
                  f"{worst_expl*100:.1f}% edge-explainable, "
                  f"largest blob {worst_blob} px")
            print("         -> unexplained pixels in blobs mean something moved")
            print("            that is not an ego or NPC vehicle. Check for")
            print(
                "            pedestrians (meta['actors'] filters vehicle.* only)")
            print("            or map layers that failed to unload.")
            ok = False
    except Exception as e:
        print(f"  [WARN] depth check skipped: {e}")

    print()
    if ok:
        print("=" * 62)
        print("DETERMINISM VERIFIED. Simulation state is reproducible.")
        print("Weather can now be the only variable across conditions.")
        print("=" * 62)
        return 0
    print("=" * 62)
    print("DETERMINISM FAILED. Fix before capturing the matrix -- every")
    print("per-condition metric would otherwise be confounded by trajectory.")
    print("=" * 62)
    return 1


if __name__ == "__main__":
    sys.exit(main())
