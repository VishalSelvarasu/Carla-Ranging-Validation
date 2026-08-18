#!/usr/bin/env python3

import json
import os
import sys

import numpy as np


def load_meta(run):
    d = os.path.join(run, "meta")
    out = []
    for fn in sorted(os.listdir(d)):
        with open(os.path.join(d, fn)) as f:
            out.append(json.load(f))
    return out


def cmp_field(a, b, get, label, tol=1e-6):
    diffs = [abs(get(x) - get(y)) for x, y in zip(a, b)]
    worst = max(diffs) if diffs else 0.0
    n_bad = sum(1 for d in diffs if d > tol)
    status = "IDENTICAL" if worst <= tol else "DIFFERS"
    print(f"  {label:28s} {status:10s} max delta {worst:.9f}  "
          f"({n_bad}/{len(diffs)} frames)")
    return worst <= tol


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    A, B = load_meta(a_dir), load_meta(b_dir)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    print(f"Comparing {n} frames\n")

    print("EGO (physics + seeding):")
    ego_ok = all([
        cmp_field(A, B, lambda m: m["ego_transform"]
                  ["location"]["x"], "ego x"),
        cmp_field(A, B, lambda m: m["ego_transform"]
                  ["location"]["y"], "ego y"),
        cmp_field(A, B, lambda m: m["ego_transform"]
                  ["rotation"]["yaw"], "ego yaw"),
        cmp_field(A, B, lambda m: m["ego_speed_mps"], "ego speed"),
    ])

    print("\nNPC TRAFFIC (Traffic Manager determinism):")
    ids_a = [sorted(x["id"] for x in m["actors"]) for m in A]
    ids_b = [sorted(x["id"] for x in m["actors"]) for m in B]
    same_count = all(len(x) == len(y) for x, y in zip(ids_a, ids_b))
    print(f"  {'actor count':28s} {'IDENTICAL' if same_count else 'DIFFERS':10s} "
          f"first frame: {len(ids_a[0])} vs {len(ids_b[0])}")

    # IDs are assigned sequentially by the server and will differ between
    # sessions -- that alone is harmless. Positions are what matter.
    npc_ok = same_count
    if same_count:
        worst = 0.0
        for ma, mb in zip(A, B):
            pa = sorted([(x["transform"]["location"]["x"],
                          x["transform"]["location"]["y"]) for x in ma["actors"]])
            pb = sorted([(x["transform"]["location"]["x"],
                          x["transform"]["location"]["y"]) for x in mb["actors"]])
            for (x1, y1), (x2, y2) in zip(pa, pb):
                worst = max(worst, abs(x1 - x2), abs(y1 - y2))
        npc_ok = worst < 1e-3
        print(f"  {'npc positions (sorted)':28s} "
              f"{'IDENTICAL' if npc_ok else 'DIFFERS':10s} max delta {worst:.6f} m")

    print("\nSERVER BOOKKEEPING (expected to differ, harmless):")
    fa, fb = A[0]["carla_frame"], B[0]["carla_frame"]
    print(f"  {'first carla_frame':28s} {fa} vs {fb}  "
          f"(offset {fb - fa}; the server counter is not reset per session)")

    print("\nFILE CATEGORIES THAT DIFFER:")
    import hashlib
    for sub in ("rgb", "depth", "semantic", "meta"):
        pa, pb = os.path.join(a_dir, sub), os.path.join(b_dir, sub)
        if not (os.path.isdir(pa) and os.path.isdir(pb)):
            continue
        names = sorted(set(os.listdir(pa)) & set(os.listdir(pb)))
        bad = 0
        for nm in names:
            ha = hashlib.md5(
                open(os.path.join(pa, nm), "rb").read()).hexdigest()
            hb = hashlib.md5(
                open(os.path.join(pb, nm), "rb").read()).hexdigest()
            if ha != hb:
                bad += 1
        print(f"  {sub:28s} {bad}/{len(names)} files differ")

    # Magnitude matters: a few centimetres is float noise, metres is a
    # different scene.
    print("\nDEPTH MAGNITUDE (frame 0):")
    try:
        import cv2
        da = cv2.imread(os.path.join(a_dir, "depth", "000000.png"),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
        db = cv2.imread(os.path.join(b_dir, "depth", "000000.png"),
                        cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
        diff = np.abs(da - db)
        print(f"  pixels differing : {(diff > 0.01).mean()*100:.2f}%")
        print(f"  max difference   : {diff.max():.2f} m")
        print(f"  mean difference  : {diff.mean():.4f} m")
    except Exception as e:
        print(f"  (skipped: {e})")

    print("\n" + "=" * 60)
    print("VERDICT:")
    if not ego_ok:
        print("  Ego trajectory differs -> physics/seeding problem.")
        print("  Check substepping settings and that the TM seed is set BEFORE spawning.")
    elif not npc_ok:
        print("  Ego identical, NPCs differ -> Traffic Manager state persists")
        print("  across runs. load_world() resets the world but NOT the TM.")
        print("  Fix: restart the CARLA server between runs, or reuse one process")
        print("  for all captures so the TM is seeded exactly once.")
    else:
        print("  Ego and NPCs identical. Divergence is in rendering only.")
        print("  Check whether depth differences are sub-centimetre (harmless")
        print("  float noise -> compare numerically, not by hash) or metres")
        print("  (a genuinely different scene).")


if __name__ == "__main__":
    main()
