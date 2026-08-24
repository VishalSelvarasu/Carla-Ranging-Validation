#!/usr/bin/env python3
from __future__ import annotations
from .config import baseline, eval_params, severity_map, sort_by_severity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os

import matplotlib
matplotlib.use("Agg")


ESTIMATORS = ("height_prior", "ground_plane", "fused")
COLOURS = {"height_prior": "#1f77b4",
           "ground_plane": "#d62728", "fused": "#2ca02c"}


def ordered(df):
    order = sort_by_severity(df["condition"].unique().tolist())
    df = df.copy()
    df["condition"] = pd.Categorical(
        df["condition"], categories=order, ordered=True)
    df["severity"] = df["condition"].astype(str).map(severity_map())
    return df.sort_values(["severity", "condition"]), order


def fig_range_error(rng, out):
    rng, order = ordered(rng)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(order))

    for name in ESTIMATORS:
        g = rng[rng["estimator"] == name].set_index("condition").reindex(order)
        a1.plot(x, g["absrel"] * 100, "o-", label=name, color=COLOURS[name])
        a2.plot(x, g["rmse_m"], "o-", label=name, color=COLOURS[name])

    for ax, ylab, title in ((a1, "AbsRel (%)", "Relative range error"),
                            (a2, "RMSE (m)", "Absolute range error")):
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_by_distance(rng, out):
    bins = [c for c in rng.columns if c.startswith("rmse_") and c.endswith("m")
            and c != "rmse_m"]
    if not bins:
        return False
    rng, order = ordered(rng)
    base = baseline()
    worst = order[-1]
    show = [c for c in (base, worst) if c in order]

    fig, axes = plt.subplots(1, len(show), figsize=(6 * len(show), 4.4),
                             squeeze=False, sharey=True)
    labels = [b.replace("rmse_", "").replace("m", "").replace("_", "-") + " m"
              for b in bins]
    xpos = np.arange(len(bins))

    for ax, cond in zip(axes[0], show):
        missing = []
        for name in ESTIMATORS:
            row = rng[(rng["condition"] == cond) & (rng["estimator"] == name)]
            if row.empty:
                continue
            vals = row[bins].values[0].astype(float)
            ax.plot(xpos, vals, "o-", label=name, color=COLOURS[name])
            missing.append(np.isnan(vals))

        # Bins are dropped when a condition has under 10 samples in them. Left
        # implicit, the panel just gets shorter and the reader assumes the
        # ranges were never sampled. Mark them: in the worst conditions the
        # detector finding nothing at distance IS part of the result.
        if missing:
            gone = np.all(missing, axis=0)
            for i, is_gone in enumerate(gone):
                if is_gone:
                    ax.axvspan(i - 0.4, i + 0.4, color="0.85", zorder=0)
                    ax.annotate("n < 10", (i, 0.5), xycoords=("data", "axes fraction"),
                                ha="center", va="center", fontsize=8, color="0.35",
                                rotation=90)

        # Same x categories in both panels so the two are directly comparable.
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_title(f"{cond}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("RMSE (m)")

    fig.suptitle(
        "Range error by distance: baseline vs worst condition", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def fig_latency(lsum, out):
    if lsum.empty:
        return False
    lsum, order = ordered(lsum)

    # A miss-rate panel with every bar at zero renders as an empty axis with a
    # -0.04 to 0.04 range, which reads as a broken chart rather than as "no
    # misses". Only draw it when there is something to draw.
    show_miss = float(lsum["miss_rate"].fillna(0).max()) > 0
    ncols = 2 if show_miss else 1

    fig, axes = plt.subplots(1, ncols, figsize=(6.8 * ncols, 4.6),
                             squeeze=False)
    a1 = axes[0][0]
    a2 = axes[0][1] if show_miss else None

    x = np.arange(len(order))
    w = 0.26

    for i, name in enumerate(ESTIMATORS):
        g = lsum[lsum["estimator"] == name].set_index(
            "condition").reindex(order)
        a1.bar(x + (i - 1) * w, g["mean_latency_s"] * 1000, w,
               label=name, color=COLOURS[name])
        if a2 is not None:
            a2.bar(x + (i - 1) * w, g["miss_rate"] * 100, w,
                   label=name, color=COLOURS[name])

    a1.axhline(0, color="k", lw=0.8)
    a1.set_ylabel("Brake latency (ms)")
    a1.set_title(f"Brake trigger latency (TTC < {eval_params()['ttc_brake_threshold_s']}s)\n"
                 "positive = LATE = unsafe", fontsize=10)
    if a2 is not None:
        a2.set_ylabel("Miss rate (%)")
        a2.set_title("Braking events never triggered", fontsize=10)

    for ax in [a for a in (a1, a2) if a is not None]:
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    if not show_miss:
        a1.text(0.99, 0.02, "miss rate 0% in all conditions",
                transform=a1.transAxes, ha="right", va="bottom",
                fontsize=8, color="0.35")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if pd.isna(v):
                cells.append("—")
            elif c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def matched_range_note(results_dir, cutoff=40.0):
    """
    RMSE compared across conditions is only like-for-like if the range
    distributions match. They do not: in the worst conditions the detector
    finds nothing at distance, so those samples are missing entirely and the
    headline comparison is computed over different distributions.

    Recomputing with a common range cap gives the honest number.
    """
    path = os.path.join(results_dir, "detections.parquet")
    if not os.path.exists(path):
        return []
    try:
        d = pd.read_parquet(path)
    except Exception:
        return []
    d = d[d["true_range_m"] < cutoff]
    if d.empty:
        return []

    rows = d.groupby("condition").apply(
        lambda g: pd.Series({e: float((g[e + "_err_m"] ** 2).mean() ** 0.5)
                             for e in ESTIMATORS}))
    order = sort_by_severity(rows.index.tolist())
    rows = rows.reindex(order)

    out = ["", f"## RMSE restricted to < {cutoff:.0f} m", "",
           "The unrestricted table above compares conditions over different",
           "range distributions: where the detector finds nothing at distance,",
           "those samples are absent rather than counted as errors. Capping",
           "range makes the comparison like-for-like.", "",
           "| condition | " + " | ".join(ESTIMATORS) + " |",
           "|---|" + "|".join("---" for _ in ESTIMATORS) + "|"]
    for cond, r in rows.iterrows():
        out.append("| " + cond + " | " +
                   " | ".join(f"{r[e]:.3f}" for e in ESTIMATORS) + " |")

    base, worst = order[0], order[-1]
    for e in ESTIMATORS:
        ratio = rows.loc[worst, e] / max(rows.loc[base, e], 1e-9)
        out.append("")
        out.append(f"`{e}`: {rows.loc[base, e]:.2f} m -> "
                   f"{rows.loc[worst, e]:.2f} m ({ratio:.2f}x) "
                   f"from {base} to {worst}.")
    out.append("")
    return out


def write_markdown(rng, lsum, path, figs, results_dir):
    rng, order = ordered(rng)
    def pct(v): return f"{v*100:.1f}%"
    parts = ["# Auto-generated results",
             "",
             "> Generated by `src/report.py` from ONE seed, uncapped unless a",
             "> section says otherwise. The published headline is four seeds",
             "> pooled and capped at 40 m -- see the README and",
             "> `results_pooled/range_pooled_40m.csv`. Numbers here will not",
             "> match it and are not meant to.",
             "",
             md_table(rng, ["condition", "estimator", "n", "absrel", "rmse_m",
                            "bias_m", "invalid_frac"],
                      {"absrel": pct, "invalid_frac": pct}),
             ""]

    parts += matched_range_note(results_dir)

    for f in figs:
        parts += [f"![{f}]({f})", ""]

    if not lsum.empty:
        lsum_o, _ = ordered(lsum)
        parts += ["## Brake latency", "",
                  md_table(lsum_o, ["condition", "estimator", "n_events",
                                    "miss_rate", "mean_latency_s",
                                    "p95_latency_s", "mean_extra_distance_m"],
                           {"miss_rate": pct}),
                  ""]

        n_ev = int(lsum_o["n_events"].max()) if "n_events" in lsum_o else 0
        if n_ev <= 2:
            parts += ["", f"> Only {n_ev} braking event per condition. Every",
                      "> latency figure is a single measurement, so the p95",
                      "> column carries no information and no spread can be",
                      "> quoted. Pool several seeds before treating these as",
                      "> anything but indicative.", ""]

        hit = lsum_o.dropna(subset=["mean_latency_s"])
        if not hit.empty:
            w = hit.loc[hit["mean_latency_s"].idxmax()]
            parts += ["## Headline", "",
                      f"> Under `{w['condition']}`, the `{w['estimator']}` estimator "
                      f"crosses the {eval_params()['ttc_brake_threshold_s']} s brake "
                      f"threshold **{w['mean_latency_s']*1000:+.0f} ms** "
                      f"{'late' if w['mean_latency_s'] > 0 else 'early'} "
                      f"({w['mean_extra_distance_m']:+.2f} m of travel), and misses "
                      f"the trigger on {w['miss_rate']*100:.1f}% of closing events.",
                      ""]

            # Spread must be measured PER ESTIMATOR across conditions. Pooling
            # all rows mixes two axes: estimators differ from each other by
            # construction, and that difference will mask a total absence of
            # variation across conditions -- which is precisely the thing this
            # check exists to catch.
            spread = hit.groupby("estimator")["mean_latency_s"].agg(
                lambda s: s.max() - s.min()).max()
            worst_lat = hit["mean_latency_s"].max()

            if spread < 0.05 or worst_lat <= 0:
                parts += ["", "## No failure boundary found", "",
                          f"Latency variation across conditions is only "
                          f"{spread*1000:.0f} ms (largest for any single estimator), "
                          f"and the worst-case trigger is {worst_lat*1000:+.0f} ms.",
                          "",
                          "The condition matrix is not severe enough to be informative.",
                          "A benchmark that finds no failure has measured nothing.",
                          "",
                          "Extend it with custom `carla.WeatherParameters`: raise",
                          "`fog_density` toward 100, `precipitation` and",
                          "`precipitation_deposits` toward 100, and drop",
                          "`sun_altitude_angle` below 0. Add rows to",
                          "`configs/conditions.yaml` and re-run from phase 1.", ""]

    with open(path, "w", newline="\n") as f:
        f.write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    args = ap.parse_args()
    R = args.results

    rng = pd.read_csv(os.path.join(R, "range_summary.csv"))
    lpath = os.path.join(R, "latency_summary.csv")
    # An empty latency CSV means no braking events occurred, which is a valid
    # outcome, not an error. pandas raises EmptyDataError on a header-only file.
    lsum = pd.DataFrame()
    if os.path.exists(lpath) and os.path.getsize(lpath) > 0:
        try:
            lsum = pd.read_csv(lpath)
        except pd.errors.EmptyDataError:
            pass
    if lsum.empty:
        print("No braking events. TTC never crossed the threshold -- the ego")
        print("keeps a safe gap, so closing speed stays low. Recapture with")
        print("tm.ignore_vehicles_percentage(ego, 100) for genuine closing.")

    figs = []
    fig_range_error(rng, os.path.join(R, "fig1_single_seed_uncapped.png"))
    figs.append("fig1_single_seed_uncapped.png")
    if fig_by_distance(rng, os.path.join(R, "fig2_error_by_distance.png")):
        figs.append("fig2_error_by_distance.png")
    if fig_latency(lsum, os.path.join(R, "fig3_brake_latency.png")):
        figs.append("fig3_brake_latency.png")

    out = os.path.join(R, "RESULTS_AUTO.md")
    write_markdown(rng, lsum, out, figs, R)
    print(f"Wrote {out} and {len(figs)} figures to {R}")
    print("Now write the argument in results/RESULTS.md. The tables are not the finding.")


if __name__ == "__main__":
    main()
