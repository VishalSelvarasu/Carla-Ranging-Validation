#!/usr/bin/env python3
from src.labels import (SEM_TAG_VEHICLE, bbox_corners_world, world_to_camera,
                        project, intrinsic_matrix, clip_box,
                        visibility_fraction, MIN_BOX_HEIGHT_PX,
                        MIN_BOX_AREA_PX, MIN_VISIBILITY, MAX_RANGE_M)
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    run = sys.argv[1]
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    stem = f"{idx:06d}"

    cfg = json.load(open(os.path.join(run, "run_config.json")))
    meta = json.load(open(os.path.join(run, "meta", stem + ".json")))
    intr = cfg["intrinsics"]
    K = intrinsic_matrix(intr)

    print(f"=== {run} frame {stem} ===\n")

    # ---- what the capture recorded ---------------------------------------
    print("META CONTENTS:")
    print(f"  keys                    : {sorted(meta.keys())}")
    has_cam = "camera_world_transform" in meta
    print(
        f"  camera_world_transform  : {'PRESENT' if has_cam else 'MISSING <-- fatal'}")
    actors = meta.get("actors", [])
    print(f"  actors logged           : {len(actors)}")
    if not actors:
        print("\n  No actors in meta. carla_capture.py did not record them.")
        print("  Check that the meta dict includes the actors list and that it")
        print("  is not nested inside the `if not args.no_rendering` block.")
        return 1
    if not has_cam:
        print("\n  camera_world_transform missing -> labels.py cannot project.")
        return 1

    print(f"  ego location            : {meta['ego_transform']['location']}")
    print(
        f"  camera location         : {meta['camera_world_transform']['location']}")

    # ---- what is actually in the buffers ---------------------------------
    sem = cv2.imread(os.path.join(run, "semantic", stem + ".png"),
                     cv2.IMREAD_UNCHANGED)
    if sem.ndim == 3:
        sem = sem[:, :, 2]
    depth = cv2.imread(os.path.join(run, "depth", stem + ".png"),
                       cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0

    vals, counts = np.unique(sem, return_counts=True)
    print("\nSEMANTIC BUFFER:")
    print(f"  shape {sem.shape} dtype {sem.dtype}")
    print("  tag: pixel count (top 10)")
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])[:10]:
        mark = "  <-- SEM_TAG_VEHICLE" if v == SEM_TAG_VEHICLE else ""
        print(f"    {int(v):3d}: {int(c):8d} ({c/sem.size*100:5.2f}%){mark}")
    n_veh = int((sem == SEM_TAG_VEHICLE).sum())
    print(f"  pixels tagged vehicle   : {n_veh} ({n_veh/sem.size*100:.3f}%)")
    if n_veh == 0:
        print("  -> no vehicle pixels. Either SEM_TAG_VEHICLE is wrong for this")
        print("     build, or the semantic buffer was written with a palette")
        print("     applied instead of raw tags.")

    print(f"\nDEPTH BUFFER: min {depth.min():.2f} m  "
          f"median {np.median(depth):.2f} m  max {depth.max():.2f} m")

    # ---- the funnel -------------------------------------------------------
    print(f"\nREJECTION FUNNEL ({len(actors)} actors):")
    stages = {"behind camera": 0, "range": 0, "off image": 0,
              "too small": 0, "occluded": 0}
    survivors = []
    detail = []

    for a in actors:
        corners_c = world_to_camera(bbox_corners_world(
            a), meta["camera_world_transform"])

        if np.any(corners_c[2] < 0.5):
            stages["behind camera"] += 1
            continue

        true_range = float(corners_c[2].min())
        if not (2.0 < true_range < MAX_RANGE_M):
            stages["range"] += 1
            detail.append(
                f"    range {true_range:7.1f} m  {a['type_id'][:34]}")
            continue

        uv = project(corners_c, K)
        box = clip_box(uv[0], uv[1], intr["width"], intr["height"])
        if box is None:
            stages["off image"] += 1
            continue

        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1
        if h < MIN_BOX_HEIGHT_PX or w * h < MIN_BOX_AREA_PX:
            stages["too small"] += 1
            detail.append(f"    small {w}x{h} px at {true_range:.1f} m")
            continue

        vis = visibility_fraction(box, depth, sem, true_range)
        if vis < MIN_VISIBILITY:
            stages["occluded"] += 1
            # The two halves of the visibility test, separately -- if the
            # semantic half passes and the depth half fails, the problem is
            # the range definition, not the tag constant.
            d = depth[y1:y2, x1:x2]
            s = sem[y1:y2, x1:x2]
            tol = max(1.5, true_range * 0.10)
            detail.append(
                f"    occl  vis={vis:.3f} at {true_range:5.1f} m  "
                f"box {w}x{h}  sem_ok={float((s == SEM_TAG_VEHICLE).mean()):.3f}  "
                f"depth_ok={float((np.abs(d - true_range) < tol).mean()):.3f}  "
                f"depth_med={float(np.median(d)):.1f}")
            continue

        survivors.append((true_range, vis, w, h, a["type_id"]))

    for k, v in stages.items():
        print(f"  rejected: {k:16s} {v}")
    print(f"  SURVIVED                 {len(survivors)}")

    if detail:
        print("\nDETAIL (first 12 near-misses):")
        for line in detail[:12]:
            print(line)

    if survivors:
        print("\nSURVIVORS:")
        for r, vis, w, h, t in sorted(survivors)[:10]:
            print(f"    {r:6.1f} m  vis {vis:.2f}  {w}x{h} px  {t}")

    print("\n" + "=" * 60)
    if survivors:
        print("Labels are being produced. If labels.py still reports zero,")
        print("the bug is in its file-writing path, not its filters.")
    elif stages["occluded"] == max(stages.values()):
        print("Everything dies at the VISIBILITY test. Read sem_ok and")
        print("depth_ok above: sem_ok near 0 means the tag constant or the")
        print("semantic write is wrong; depth_ok near 0 with sem_ok high")
        print("means true_range disagrees with the depth buffer.")
    elif stages["behind camera"] == len(actors):
        print("Every actor is BEHIND the camera. The world->camera transform")
        print("is inverted, or camera_world_transform holds the ego pose")
        print("rather than the camera's.")
    elif stages["range"] == max(stages.values()):
        print("Everything fails the RANGE gate. Check the printed distances:")
        print("implausible values mean the transform chain is wrong.")
    else:
        print("See the funnel above for the dominant rejection stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
