#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .config import sort_by_severity

ESTIMATORS = ("height_prior", "ground_plane", "fused")


def load_events(results_dirs):
    frames = []
    for d in results_dirs:
        p = os.path.join(d, "brake_events.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"  skip {d} (no brake_events.csv)")
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            print(f"  skip {d} (empty)")
            continue
        if df.empty:
            print(f"  skip {d} (no events)")
            continue
        df["source"] = os.path.basename(d.rstrip("/\\"))
        frames.append(df)
    if not frames:
        raise SystemExit("No brake events found in any results directory.")
    return pd.concat(frames, ignore_index=True)


def load_ttc(results_dirs):
    """Per-frame TTC summaries, one row per (condition, estimator, seed)."""
    frames = []
    for d in results_dirs:
        p = os.path.join(d, "ttc_summary.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["source"] = os.path.basename(d.rstrip("/\\"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_detections(results_dirs, cutoff):
    frames = []
    for d in results_dirs:
        p = os.path.join(d, "detections.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        df["source"] = os.path.basename(d.rstrip("/\\"))
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df[df["true_range_m"] < cutoff] if cutoff else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="./results_pooled")
    ap.add_argument("--ref-speed", type=float, default=8.0,
                    help="Reference ego speed (m/s) for converting pooled "
                         "latency into metres. Report it alongside the number.")
    ap.add_argument("--range-cutoff", type=float, default=60.0,
                    help="Cap range so conditions are compared over the same "
                         "distribution. Where the detector finds nothing at "
                         "distance those samples are absent, not counted as "
                         "errors, which makes the raw comparison unequal.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Pooling {len(args.results)} runs\n")
    ev = load_events(args.results)
    hit = ev[~ev["missed"]].copy()

    order = sort_by_severity(ev["condition"].unique().tolist())

    rows = []
    for cond in order:
        for est in ESTIMATORS:
            g = hit[(hit["condition"] == cond) & (hit["estimator"] == est)]
            all_g = ev[(ev["condition"] == cond) & (ev["estimator"] == est)]
            if all_g.empty:
                continue
            lat = g["latency_s"].to_numpy(dtype=float)
            rows.append({
                "condition": cond,
                "estimator": est,
                "n_seeds": int(all_g["source"].nunique()),
                "n_events": int(len(all_g)),
                "miss_rate": float(all_g["missed"].mean()),
                "mean_latency_s": float(lat.mean()) if len(lat) else np.nan,
                # Sample std over independent seeds. With n=4 it is a weak
                # estimate, but "0.43 +/- 0.05 s" and "0.43 +/- 0.30 s" are
                # very different claims and the reader deserves to know which.
                "std_latency_s": float(lat.std(ddof=1)) if len(lat) > 1 else np.nan,
                "min_latency_s": float(lat.min()) if len(lat) else np.nan,
                "max_latency_s": float(lat.max()) if len(lat) else np.nan,
                "late_frac": float((lat > 0).mean()) if len(lat) else np.nan,
                f"extra_m_at_{args.ref_speed:g}mps":
                    float(lat.mean() * args.ref_speed) if len(lat) else np.nan,
            })

    lat_df = pd.DataFrame(rows)
    lat_df.to_csv(os.path.join(args.out, "latency_pooled.csv"), index=False)

    pd.set_option("display.width", 200)
    print("=== Brake latency pooled across seeds ===")
    print(lat_df.to_string(index=False))

    # Per-frame TTC error. Unlike brake latency this has ~100 frames per run
    # rather than one threshold crossing, so a difference across conditions can
    # actually be defended rather than merely observed.
    ttc = load_ttc(args.results)
    if ttc is not None:
        rows = []
        for cond in order:
            for est in ESTIMATORS:
                g = ttc[(ttc["condition"] == cond) & (ttc["estimator"] == est)]
                if g.empty:
                    continue
                rows.append({
                    "condition": cond,
                    "estimator": est,
                    "n_seeds": int(g["source"].nunique()),
                    "n_frames": int(g["n_frames"].sum()),
                    # Weighted by frames so a short run does not count equally
                    # with a long one.
                    "mean_ttc_err_s": float(
                        np.average(g["mean_ttc_err_s"], weights=g["n_frames"])),
                    "over_250ms_frac": float(
                        np.average(g["over_250ms_frac"], weights=g["n_frames"])),
                    "over_250ms_min": float(g["over_250ms_frac"].min()),
                    "over_250ms_max": float(g["over_250ms_frac"].max()),
                })
        ttc_p = pd.DataFrame(rows)
        ttc_p.to_csv(os.path.join(args.out, "ttc_pooled.csv"), index=False)
        print("\n=== TTC error pooled across seeds "
              "(frames with true TTC < 5 s) ===")
        print("over_250ms_frac = share of approach frames where the estimate")
        print("believes it has >250 ms more time than it does")
        print(ttc_p.to_string(index=False))

        print("\n=== Consistency across seeds ===")
        for est in ESTIMATORS:
            b = ttc_p[(ttc_p.condition == order[0]) & (ttc_p.estimator == est)]
            w = ttc_p[(ttc_p.condition == order[-1])
                      & (ttc_p.estimator == est)]
            if b.empty or w.empty:
                continue
            # Ranges must not overlap for the effect to be visible in every
            # seed. Overlap means at least one seed contradicts the trend.
            sep = "separated" if w.over_250ms_min.iloc[0] > b.over_250ms_max.iloc[0] \
                else "OVERLAPPING -- at least one seed contradicts"
            print(f"  {est:14s} {order[0]}: "
                  f"{b.over_250ms_min.iloc[0]*100:5.1f}-{b.over_250ms_max.iloc[0]*100:5.1f}%   "
                  f"{order[-1]}: "
                  f"{w.over_250ms_min.iloc[0]*100:5.1f}-{w.over_250ms_max.iloc[0]*100:5.1f}%   "
                  f"-> {sep}")

    det = load_detections(args.results, args.range_cutoff)
    if det is not None:
        recs = []
        for cond in order:
            g = det[det["condition"] == cond]
            if g.empty:
                continue
            rec = {"condition": cond, "n": len(g)}
            for est in ESTIMATORS:
                err = g[f"{est}_err_m"].dropna()
                rec[f"{est}_rmse"] = float((err ** 2).mean() ** 0.5)
                rec[f"{est}_bias"] = float(err.mean())
            recs.append(rec)
        rng_df = pd.DataFrame(recs)
        rng_df.to_csv(os.path.join(args.out, "range_pooled.csv"), index=False)
        print(f"\n=== Range error pooled, range < {args.range_cutoff:g} m ===")
        print(rng_df.to_string(index=False))

    # Degradation ratios, baseline to worst, with the spread that supports them
    print("\n=== Degradation, "
          f"{order[0]} -> {order[-1]} ===")
    for est in ESTIMATORS:
        b = lat_df[(lat_df.condition == order[0]) & (lat_df.estimator == est)]
        w = lat_df[(lat_df.condition == order[-1]) & (lat_df.estimator == est)]
        if b.empty or w.empty:
            continue
        bm, wm = b.mean_latency_s.iloc[0], w.mean_latency_s.iloc[0]
        bs, ws = b.std_latency_s.iloc[0], w.std_latency_s.iloc[0]
        print(f"  {est:14s} {bm*1000:+7.0f} +/- {bs*1000:4.0f} ms  ->  "
              f"{wm*1000:+7.0f} +/- {ws*1000:4.0f} ms   "
              f"({wm*args.ref_speed - bm*args.ref_speed:+.2f} m more travel "
              f"at {args.ref_speed:g} m/s)")

    # Separation matters more than the ratio. If the spread across seeds swamps
    # the difference between conditions, there is no result yet -- just noise
    # with a trend drawn through it.
    print("\n=== Is the degradation bigger than the seed-to-seed spread? ===")
    for est in ESTIMATORS:
        b = lat_df[(lat_df.condition == order[0]) & (lat_df.estimator == est)]
        w = lat_df[(lat_df.condition == order[-1]) & (lat_df.estimator == est)]
        if b.empty or w.empty or np.isnan(b.std_latency_s.iloc[0]):
            continue
        diff = w.mean_latency_s.iloc[0] - b.mean_latency_s.iloc[0]
        pooled = np.sqrt((b.std_latency_s.iloc[0] ** 2 +
                          w.std_latency_s.iloc[0] ** 2) / 2)
        if pooled > 0:
            print(f"  {est:14s} difference {diff*1000:+.0f} ms, "
                  f"pooled spread {pooled*1000:.0f} ms  ->  "
                  f"{abs(diff)/pooled:.1f}x the spread")

    print(f"\nWrote {args.out}/latency_pooled.csv and range_pooled.csv")


if __name__ == "__main__":
    main()
