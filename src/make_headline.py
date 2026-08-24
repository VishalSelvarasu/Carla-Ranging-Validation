#!/usr/bin/env python3
from __future__ import annotations
from .config import sort_by_severity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")


ESTIMATORS = ("height_prior", "ground_plane", "fused")
COLOURS = {"height_prior": "#1f77b4", "ground_plane": "#d62728",
           "fused": "#2ca02c"}
BANDS = [(0, 20), (20, 40), (40, 60), (60, 90)]


def load_pooled(results_dirs, cutoff):
    frames = []
    for d in results_dirs:
        p = os.path.join(d, "detections.parquet")
        if not os.path.exists(p):
            print(f"  skip {d} (no detections.parquet)")
            continue
        df = pd.read_parquet(p)
        df["source"] = os.path.basename(d.rstrip("/\\"))
        frames.append(df)
    if not frames:
        raise SystemExit("No detections.parquet found.")
    df = pd.concat(frames, ignore_index=True)
    return df, df[df["true_range_m"] < cutoff]


def recall_table(dataset_dirs):
    """
    Recall recomputed from detections.json rather than re-running the detector.
    Each frame record carries n_gt and n_missed, so this is the same number
    src/detect.py prints, without an 8-minute inference pass.
    """
    rows = []
    for d in dataset_dirs:
        if not os.path.isdir(d):
            continue
        seed = os.path.basename(d.rstrip("/\\"))
        for cond in sorted(os.listdir(d)):
            p = os.path.join(d, cond, "detections.json")
            if not os.path.exists(p):
                continue
            recs = json.load(open(p))
            matched = sum(len(r["objects"]) for r in recs.values())
            missed = sum(r.get("n_missed", 0) for r in recs.values())
            rows.append({"dataset": seed, "condition": cond,
                         "matched": matched, "missed": missed,
                         "recall": matched / max(matched + missed, 1)})
    return pd.DataFrame(rows)


def summarise(df):
    out = []
    for cond, g in df.groupby("condition"):
        rec = {"condition": cond, "n": len(g)}
        for e in ESTIMATORS:
            err = g[f"{e}_err_m"].dropna()
            rec[f"{e}_rmse"] = float((err ** 2).mean() ** 0.5)
            rec[f"{e}_bias"] = float(err.mean())
        out.append(rec)
    df_out = pd.DataFrame(out)
    order = sort_by_severity(df_out["condition"].tolist())
    return df_out.set_index("condition").reindex(order).reset_index()


def band_table(df_all):
    rows = []
    for (src, cond), g in df_all.groupby(["source", "condition"]):
        for lo, hi in BANDS:
            m = g["true_range_m"].between(lo, hi, inclusive="left")
            if m.sum() == 0:
                continue
            rows.append({"dataset": src, "condition": cond,
                         "band": f"{lo}-{hi}m", "n": int(m.sum()),
                         "mean_err_m": float(g.loc[m, "ground_plane_err_m"].mean())})
    return pd.DataFrame(rows)


def figure(summary, cutoff, out):
    order = summary["condition"].tolist()
    x = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))

    for e in ESTIMATORS:
        a1.plot(x, summary[f"{e}_rmse"], "o-", label=e, color=COLOURS[e])
        a2.plot(x, summary[f"{e}_bias"], "o-", label=e, color=COLOURS[e])

    a1.set_ylabel("RMSE (m)")
    a1.set_title(f"Range error, pooled over 4 seeds, range < {cutoff:g} m")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_ylabel("Mean signed bias (m)")
    a2.set_title("Positive bias = target reported farther than it is")

    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    n_note = f"n per condition: {summary['n'].min()}-{summary['n'].max()}"
    a1.text(0.99, 0.02, n_note, transform=a1.transAxes, ha="right",
            va="bottom", fontsize=8, color="0.35")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+",
                    default=["results", "results_s43", "results_s44",
                             "results_s45"])
    ap.add_argument("--datasets", nargs="+",
                    default=["dataset", "dataset_s43", "dataset_s44",
                             "dataset_s45"])
    ap.add_argument("--cutoff", type=float, default=40.0)
    ap.add_argument("--out-dir", default="results_pooled")
    ap.add_argument("--fig", default="results/fig1_range_error_pooled_40m.png")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"pooling {len(args.results)} runs, range < {args.cutoff:g} m\n")

    all_df, capped = load_pooled(args.results, args.cutoff)
    summary = summarise(capped)
    summary.round(3).to_csv(
        os.path.join(args.out_dir, "range_pooled_40m.csv"), index=False)

    bands = band_table(all_df)
    bands.round(3).to_csv(
        os.path.join(args.out_dir, "range_band_by_seed.csv"), index=False)

    rec = recall_table(args.datasets)
    if not rec.empty:
        rec.round(4).to_csv(
            os.path.join(args.out_dir, "recall_by_seed.csv"), index=False)

    figure(summary, args.cutoff, args.fig)

    pd.set_option("display.width", 200)
    cols = ["condition", "n", "ground_plane_rmse", "ground_plane_bias"]
    print(summary[cols].round(2).to_string(index=False))
    print(f"\nwrote {args.fig}")
    print(f"wrote {args.out_dir}/range_pooled_40m.csv, "
          f"range_band_by_seed.csv"
          + (", recall_by_seed.csv" if not rec.empty else ""))


if __name__ == "__main__":
    main()
