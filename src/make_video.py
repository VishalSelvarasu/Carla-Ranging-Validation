#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from .ranging import estimate_all

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (80, 220, 80)
RED = (60, 60, 240)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)


def load(run_dir, labels_file):
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    labels = json.load(open(os.path.join(run_dir, labels_file)))
    return cfg, labels


def annotate(run_dir, stem, rec, cfg, estimator):
    img = cv2.imread(os.path.join(run_dir, "rgb", stem + ".png"))
    if img is None:
        return None, None

    intr = cfg["intrinsics"]
    cam_h = cfg["camera_transform"]["location"]["z"]
    err = None

    for o in rec["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        est = estimate_all(o, intr, cam_h)
        z_est = est[estimator]
        z_true = o["true_range_m"]
        if not np.isfinite(z_est):
            continue
        err = z_est - z_true

        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
        # Estimate above the box, truth below, error in between. Colour the
        # error red once it exceeds a metre so the bad frames stand out.
        col = RED if abs(err) > 1.0 else GREEN
        cv2.putText(img, f"est {z_est:5.1f} m", (x1, max(14, y1 - 22)),
                    FONT, 0.5, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"true {z_true:5.1f} m", (x1, max(28, y1 - 6)),
                    FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"{err:+.1f} m", (x2 + 6, y1 + 14),
                    FONT, 0.5, col, 2, cv2.LINE_AA)

    return img, err


def header(img, text, sub=None):
    h, w = img.shape[:2]
    bar = np.zeros((44, w, 3), np.uint8)
    cv2.putText(bar, text, (10, 22), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (10, 38), FONT, 0.42, GREY, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def error_strip(errs_l, errs_r, i, width, height=90):
    """Running error for both panels so the divergence is visible."""
    strip = np.zeros((height, width, 3), np.uint8)
    n = max(len(errs_l), 1)
    lim = 3.0

    mid = height // 2
    cv2.line(strip, (0, mid), (width, mid), (60, 60, 60), 1)
    cv2.putText(strip, "range error (m)", (8, 14),
                FONT, 0.4, GREY, 1, cv2.LINE_AA)
    cv2.putText(strip, f"+{lim:.0f}", (width - 34, 12), FONT, 0.35, GREY, 1)
    cv2.putText(strip, f"-{lim:.0f}", (width - 34,
                height - 5), FONT, 0.35, GREY, 1)

    for errs, col in ((errs_l, GREY), (errs_r, RED)):
        pts = []
        for k, e in enumerate(errs[:i + 1]):
            if e is None or not np.isfinite(e):
                continue
            x = int(k / n * (width - 1))
            y = int(mid - np.clip(e / lim, -1, 1) * (mid - 8))
            pts.append((x, y))
        if len(pts) > 1:
            cv2.polylines(strip, [np.array(pts, np.int32)], False, col, 2,
                          cv2.LINE_AA)
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--left", default="ClearNoon")
    ap.add_argument("--right", default="HardRainNight")
    ap.add_argument("--labels", default="detections.json")
    ap.add_argument("--estimator", default="ground_plane")
    ap.add_argument("--out", default="results/comparison.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    ld = os.path.join(args.dataset, args.left)
    rd = os.path.join(args.dataset, args.right)
    lcfg, llab = load(ld, args.labels)
    rcfg, rlab = load(rd, args.labels)

    stems = sorted(set(llab) & set(rlab))
    if not stems:
        raise SystemExit("No frames in common between the two conditions.")
    print(f"{len(stems)} frames, {args.left} vs {args.right}")

    # Precompute errors so the strip can show the whole approach from frame 0.
    errs_l, errs_r = [], []
    for s in stems:
        for rec, cfg, out in ((llab[s], lcfg, errs_l), (rlab[s], rcfg, errs_r)):
            e = None
            for o in rec["objects"]:
                z = estimate_all(o, cfg["intrinsics"],
                                 cfg["camera_transform"]["location"]["z"])[args.estimator]
                if np.isfinite(z):
                    e = z - o["true_range_m"]
                    break
            out.append(e)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = None

    for i, s in enumerate(stems):
        li, _ = annotate(ld, s, llab[s], lcfg, args.estimator)
        ri, _ = annotate(rd, s, rlab[s], rcfg, args.estimator)
        if li is None or ri is None:
            continue

        li = header(li, args.left, f"{args.estimator} vs ground truth")
        ri = header(ri, args.right, f"{args.estimator} vs ground truth")
        pair = np.hstack([li, np.full((li.shape[0], 3, 3), 40, np.uint8), ri])
        frame = np.vstack(
            [pair, error_strip(errs_l, errs_r, i, pair.shape[1])])

        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
            if not writer.isOpened():
                raise SystemExit("VideoWriter failed to open. Try --out with .avi "
                                 "and codec XVID if mp4v is unavailable.")
        writer.write(frame)

    if writer:
        writer.release()
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
