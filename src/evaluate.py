#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .ranging import ESTIMATORS, estimate_all

DIST_BINS = [(0, 20), (20, 40), (40, 60), (60, 100)]
TTC_BRAKE_THRESHOLD_S = 2.0
MIN_CLOSING_SPEED_MPS = 1.0

# Frames with true TTC below this count as "approach phase" and are scored
# individually. A single-target scenario yields ONE threshold crossing per run,
# so brake latency has n=1 per condition per seed -- four seeds gave a
# difference of only 1.2x the seed-to-seed spread, which supports no claim.
# Scoring every approach frame instead gives ~60 samples per run from the same
# data, still in seconds, still safety-framed.
TTC_APPROACH_WINDOW_S = 5.0


# ---------------------------------------------------------------------------
# Level 1 + 2: per-detection table
# ---------------------------------------------------------------------------

def build_table(run_dir, condition, labels_file="labels.json"):
    with open(os.path.join(run_dir, "run_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, labels_file)) as f:
        labels = json.load(f)

    intr = cfg["intrinsics"]
    cam_h = cfg["camera_transform"]["location"]["z"]

    rows = []
    for stem, frame in sorted(labels.items()):
        for obj in frame["objects"]:
            est = estimate_all(obj, intr, cam_h)
            z_true = obj["true_range_m"]

            # Closing speed along the ego's forward axis. Both speeds come
            # from ground truth on purpose: we are isolating the contribution
            # of RANGE error to TTC error, not compounding it with velocity
            # estimation error. State this in the writeup -- a real monocular
            # system estimates closing speed too, so these TTC numbers are a
            # lower bound on the true error.
            v_close = frame["ego_speed_mps"] - obj["target_speed_mps"]

            row = {
                "condition": condition,
                "frame": int(stem),
                "t": frame["sim_timestamp"],
                "actor_id": obj["actor_id"],
                "class": est["class"],
                "true_range_m": z_true,
                "visibility": obj["visibility"],
                "box_height_px": obj["bbox_height_px"],
                "ego_speed_mps": frame["ego_speed_mps"],
                "closing_speed_mps": v_close,
                "ttc_true_s": (z_true / v_close
                               if v_close > MIN_CLOSING_SPEED_MPS else np.inf),
            }

            for name in ESTIMATORS:
                z = est[name]
                row[f"{name}_m"] = z
                row[f"{name}_err_m"] = z - z_true
                row[f"{name}_absrel"] = abs(z - z_true) / z_true
                row[f"{name}_ttc_s"] = (z / v_close
                                        if v_close > MIN_CLOSING_SPEED_MPS
                                        else np.inf)
            rows.append(row)

    return pd.DataFrame(rows)


def range_summary(df):
    out = []
    for cond, g in df.groupby("condition"):
        for name in ESTIMATORS:
            err = g[f"{name}_err_m"]
            valid = err.notna()
            rec = {
                "condition": cond,
                "estimator": name,
                "n": int(valid.sum()),
                # Fraction of detections the estimator simply could not range.
                # Report this alongside accuracy: an estimator that refuses
                # half its inputs can post an excellent AbsRel and still be
                # useless.
                "invalid_frac": float(1.0 - valid.mean()),
                "absrel": float(g.loc[valid, f"{name}_absrel"].mean()),
                "rmse_m": float(np.sqrt((err[valid] ** 2).mean())),
                "bias_m": float(err[valid].mean()),
                "p95_abs_err_m": float(err[valid].abs().quantile(0.95)),
            }
            for lo, hi in DIST_BINS:
                m = valid & g["true_range_m"].between(lo, hi)
                rec[f"rmse_{lo}_{hi}m"] = (float(np.sqrt((err[m] ** 2).mean()))
                                           if m.sum() >= 10 else np.nan)
            out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Level 2b: per-frame TTC error over the approach
# ---------------------------------------------------------------------------

def ttc_summary(df, window=TTC_APPROACH_WINDOW_S):
    """
    TTC error on every frame where the true TTC is inside the approach window.

    Sign convention matters more than magnitude here:
      ttc_err > 0  -> estimated TTC is LONGER than true. The system believes it
                      has more time than it does. Unsafe.
      ttc_err < 0  -> conservative; costs comfort and false positives.

    `late_frac` is the fraction of approach frames on the unsafe side. Unlike
    brake latency it has hundreds of samples per condition, so a difference
    across conditions can actually be defended.
    """
    m = df["ttc_true_s"].between(0, window)
    g = df[m]
    if g.empty:
        return pd.DataFrame()

    out = []
    for (cond, ), grp in g.groupby(["condition"]):
        for name in ESTIMATORS:
            err = (grp[f"{name}_ttc_s"] - grp["ttc_true_s"]).replace(
                [np.inf, -np.inf], np.nan).dropna()
            if err.empty:
                continue
            out.append({
                "condition": cond,
                "estimator": name,
                "n_frames": int(len(err)),
                "mean_ttc_err_s": float(err.mean()),
                "rmse_ttc_err_s": float((err ** 2).mean() ** 0.5),
                "p95_ttc_err_s": float(err.quantile(0.95)),
                "optimistic_frac": float((err > 0).mean()),
                # Frames where the estimate is optimistic by more than a
                # typical driver reaction time. This is the count that would
                # matter to a safety case.
                "over_250ms_frac": float((err > 0.25).mean()),
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Level 3: brake latency
# ---------------------------------------------------------------------------

def first_crossing(t, ttc, threshold):
    """
    First time TTC falls below the threshold, linearly interpolated between
    samples. Interpolation matters: at a 0.05 s timestep, snapping to sample
    boundaries quantises your latency to 50 ms, which is the same order as the
    effect you are trying to measure.
    """
    ttc = np.asarray(ttc, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.isfinite(ttc)
    if ok.sum() < 2:
        return None
    t, ttc = t[ok], ttc[ok]
    below = ttc < threshold
    if not below.any():
        return None
    i = int(np.argmax(below))
    if i == 0:
        return float(t[0])
    t0, t1, y0, y1 = t[i - 1], t[i], ttc[i - 1], ttc[i]
    if y0 == y1:
        return float(t1)
    return float(t0 + (threshold - y0) * (t1 - t0) / (y1 - y0))


def brake_latency(df, threshold=TTC_BRAKE_THRESHOLD_S):
    """
    Per (condition, estimator, actor) track: how late is the brake trigger?

    latency_s > 0  -> estimator triggers LATE. Unsafe.
    latency_s < 0  -> triggers early. Costs comfort and false positives.
    missed         -> true TTC crossed, estimate never did. The worst outcome,
                      and one an averaged error metric will hide completely.
    """
    rows = []
    for (cond, aid), g in df.groupby(["condition", "actor_id"]):
        g = g.sort_values("t")
        t_true = first_crossing(g["t"], g["ttc_true_s"], threshold)
        if t_true is None:
            continue  # this target never became a braking case
        v_ego = float(g["ego_speed_mps"].mean())

        for name in ESTIMATORS:
            t_est = first_crossing(g["t"], g[f"{name}_ttc_s"], threshold)
            if t_est is None:
                rows.append({"condition": cond, "actor_id": aid,
                             "estimator": name, "missed": True,
                             "latency_s": np.nan, "extra_distance_m": np.nan})
                continue
            lat = t_est - t_true
            rows.append({"condition": cond, "actor_id": aid, "estimator": name,
                         "missed": False, "latency_s": lat,
                         "extra_distance_m": lat * v_ego})
    return pd.DataFrame(rows)


def latency_summary(lat):
    if lat.empty:
        return pd.DataFrame()
    out = []
    for (cond, name), g in lat.groupby(["condition", "estimator"]):
        hit = g[~g["missed"]]
        out.append({
            "condition": cond,
            "estimator": name,
            "n_events": len(g),
            "miss_rate": float(g["missed"].mean()),
            "mean_latency_s": float(hit["latency_s"].mean()) if len(hit) else np.nan,
            "p95_latency_s": (float(hit["latency_s"].quantile(0.95))
                              if len(hit) else np.nan),
            "mean_extra_distance_m": (float(hit["extra_distance_m"].mean())
                                      if len(hit) else np.nan),
            "late_frac": (float((hit["latency_s"] > 0).mean())
                          if len(hit) else np.nan),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--threshold", type=float, default=TTC_BRAKE_THRESHOLD_S)
    ap.add_argument("--labels", default="labels.json",
                    help="labels.json = GT boxes (no weather sensitivity, "
                         "useful as an upper bound). detections.json = real "
                         "detector boxes, which is the actual experiment.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conditions = sorted(
        d for d in os.listdir(args.dataset)
        if os.path.exists(os.path.join(args.dataset, d, args.labels))
    )
    if not conditions:
        raise SystemExit(f"No runs with {args.labels}. Run src/labels.py "
                         "(and src/detect.py for detections.json) first.")

    df = pd.concat([build_table(os.path.join(args.dataset, c), c, args.labels)
                    for c in conditions], ignore_index=True)

    # If every condition produces identical metrics, nothing in the pipeline is
    # reading the pixels. That happens when evaluating GT boxes: they come from
    # projected geometry, and the depth and semantic buffers are G-buffer
    # renders, so weather changes none of it.
    if df.groupby("condition")["height_prior_err_m"].std().max() < 1e-9:
        print("\nWARNING: identical results in every condition.")
        print("  Nothing here depends on image content. If you are evaluating")
        print("  labels.json this is expected -- GT boxes are geometric. Run")
        print("  src/detect.py and evaluate detections.json instead.\n")
    df.to_parquet(os.path.join(args.out, "detections.parquet"))

    rng = range_summary(df)
    ttc = ttc_summary(df)
    lat = brake_latency(df, args.threshold)
    lsum = latency_summary(lat)

    rng.to_csv(os.path.join(args.out, "range_summary.csv"), index=False)
    ttc.to_csv(os.path.join(args.out, "ttc_summary.csv"), index=False)
    lat.to_csv(os.path.join(args.out, "brake_events.csv"), index=False)
    lsum.to_csv(os.path.join(args.out, "latency_summary.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\n=== Range error by condition ===")
    print(rng[["condition", "estimator", "n", "absrel",
               "rmse_m", "bias_m", "invalid_frac"]].to_string(index=False))
    if not ttc.empty:
        print(f"\n=== TTC error over the approach (true TTC < "
              f"{TTC_APPROACH_WINDOW_S}s) ===")
        print("positive = estimate thinks there is MORE time than there is")
        print(ttc[["condition", "estimator", "n_frames", "mean_ttc_err_s",
                   "rmse_ttc_err_s", "optimistic_frac",
                   "over_250ms_frac"]].to_string(index=False))

    print(f"\n=== Brake latency (TTC < {args.threshold}s) ===")
    print(lsum.to_string(index=False) if not lsum.empty else "no braking events")

    if not lsum.empty:
        worst = lsum.sort_values("mean_latency_s", ascending=False).iloc[0]
        print("\nHeadline:")
        print(f"  {worst['estimator']} under {worst['condition']}: brake "
              f"trigger {worst['mean_latency_s']:+.3f} s late "
              f"({worst['mean_extra_distance_m']:+.2f} m extra travel), "
              f"miss rate {worst['miss_rate']:.1%}")
        print("\nIf the spread across conditions is small, you have not yet")
        print("found the failure boundary. Extend the matrix (denser fog,")
        print("heavier rain, lower sun) until something breaks.")


if __name__ == "__main__":
    main()
