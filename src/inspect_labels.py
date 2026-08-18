from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from .config import sort_by_severity

FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(run_dir, stem, labels, thumb_w):
    img = cv2.imread(os.path.join(run_dir, "rgb", stem + ".png"))
    if img is None:
        return None
    rec = labels.get(stem, {"objects": []})

    for o in rec["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        # Colour encodes visibility: green = clear, red = mostly occluded.
        # Makes a broken occlusion filter obvious at a glance -- a sheet that
        # is uniformly green in heavy rain is suspicious.
        v = o["visibility"]
        colour = (0, int(255 * min(v * 1.5, 1.0)),
                  int(255 * (1 - min(v * 1.5, 1.0))))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, f"{o['true_range_m']:.0f}m", (x1, max(11, y1 - 4)),
                    FONT, 0.4, colour, 1, cv2.LINE_AA)

    label = f"{os.path.basename(run_dir)} f{stem} n={len(rec['objects'])}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, label, (6, 16), FONT, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    scale = thumb_w / img.shape[1]
    return cv2.resize(img, (thumb_w, int(img.shape[0] * scale)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=4,
                    help="Samples per condition.")
    ap.add_argument("--thumb-width", type=int, default=400)
    args = ap.parse_args()

    conds = sort_by_severity([
        d for d in os.listdir(args.dataset)
        if os.path.exists(os.path.join(args.dataset, d, "labels.json"))
    ])
    if not conds:
        raise SystemExit("No labelled runs. Run src/labels.py first.")

    rows, stats = [], []
    for cond in conds:
        run_dir = os.path.join(args.dataset, cond)
        with open(os.path.join(run_dir, "labels.json")) as f:
            labels = json.load(f)
        stems = sorted(labels.keys())
        if not stems:
            continue
        # Even spacing beats random sampling: the same frames appear in every
        # condition, so differences across a row are attributable to weather.
        picks = [stems[int(i * (len(stems) - 1) / max(args.frames - 1, 1))]
                 for i in range(args.frames)]

        tiles = [t for t in (annotate(run_dir, s, labels, args.thumb_width)
                             for s in picks) if t is not None]
        if tiles:
            h = min(t.shape[0] for t in tiles)
            rows.append(np.hstack([t[:h] for t in tiles]))

        n_obj = sum(len(v["objects"]) for v in labels.values())
        vis = [o["visibility"] for v in labels.values() for o in v["objects"]]
        stats.append((cond, len(labels), n_obj, n_obj / max(len(labels), 1),
                      float(np.mean(vis)) if vis else 0.0))

    w = min(r.shape[1] for r in rows)
    sheet = np.vstack([r[:, :w] for r in rows])
    out = args.out or os.path.join(args.dataset, "contact_sheet.png")
    cv2.imwrite(out, sheet)

    print(f"{'condition':18s} {'frames':>7} {'objects':>8} {'per frame':>10} {'mean vis':>9}")
    for c, nf, no, pf, mv in stats:
        print(f"{c:18s} {nf:7d} {no:8d} {pf:10.1f} {mv:9.2f}")

    per_frame = [s[3] for s in stats]
    if per_frame and min(per_frame) < 0.6 * max(per_frame):
        print("\n  Object count varies >40% across conditions. Labels are derived")
        print("  from geometry and depth, so weather should barely affect them.")
        print("  Suspect the visibility test before treating this as a result.")

    print(f"\nWrote {out} -- open it and check boxes sit ON the vehicles.")


if __name__ == "__main__":
    main()
