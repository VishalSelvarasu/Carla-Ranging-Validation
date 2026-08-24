#!/usr/bin/env python3
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from src.ranging import estimate_all   # noqa: E402  (needs sys.path first)

ESTIMATOR = "ground_plane"
BANDS = [(60, 90), (40, 60), (20, 40), (0, 20)]


def series(cond, root):
    cfg = json.load(open(f"{root}/{cond}/run_config.json"))
    dets = json.load(open(f"{root}/{cond}/detections.json"))
    intr = cfg["intrinsics"]
    cam_h = cfg["camera_transform"]["location"]["z"]

    rows = []
    for k in sorted(dets):
        for obj in dets[k]["objects"]:
            z = estimate_all(obj, intr, cam_h)[ESTIMATOR]
            if np.isfinite(z):
                rows.append(
                    (int(k), obj["true_range_m"], z - obj["true_range_m"]))
                break          # nearest target only, one per frame
    return np.array(rows)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    out = {}
    for cond in ("ClearNoon", "HardRainNight"):
        a = series(cond, root)
        out[cond] = a
        print(f"\n{cond}  ({len(a)} frames with a detection)")
        for lo, hi in BANDS:
            m = (a[:, 1] >= lo) & (a[:, 1] < hi)
            if m.sum():
                print(f"  {lo:2d}-{hi:2d} m   n={int(m.sum()):3d}   "
                      f"mean err {a[m, 2].mean():+6.2f} m   "
                      f"rmse {np.sqrt((a[m, 2] ** 2).mean()):5.2f} m")

    print("\ndifference (HardRainNight - ClearNoon)")
    for lo, hi in BANDS:
        vals = []
        for cond in ("ClearNoon", "HardRainNight"):
            a = out[cond]
            m = (a[:, 1] >= lo) & (a[:, 1] < hi)
            vals.append(a[m, 2].mean() if m.sum() else np.nan)
        if np.isfinite(vals).all():
            print(f"  {lo:2d}-{hi:2d} m   {vals[1] - vals[0]:+6.2f} m")

    # Frame range where the gap is widest -- that is the clip worth showing.
    a, b = out["ClearNoon"], out["HardRainNight"]
    common = sorted(set(a[:, 0].astype(int)) & set(b[:, 0].astype(int)))
    da = {int(r[0]): r[2] for r in a}
    db = {int(r[0]): r[2] for r in b}
    gaps = np.array([[f, db[f] - da[f]] for f in common])
    if len(gaps):
        w = 40
        best, best_i = -1e9, 0
        for i in range(0, max(len(gaps) - w, 1)):
            m = np.abs(gaps[i:i + w, 1]).mean()
            if m > best:
                best, best_i = m, i
        f0, f1 = int(gaps[best_i, 0]), int(
            gaps[min(best_i + w, len(gaps) - 1), 0])
        print(f"\nwidest {w}-frame gap: frames {f0}-{f1}, "
              f"mean |difference| {best:.2f} m")
        print(f"true range there: "
              f"{b[(b[:, 0] >= f0) & (b[:, 0] <= f1), 1].mean():.0f} m")


if __name__ == "__main__":
    main()
