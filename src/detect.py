#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# COCO classes that correspond to CARLA vehicle blueprints.
COCO_VEHICLE = {2: "car", 5: "bus", 7: "truck", 3: "motorcycle", 1: "bicycle"}

IOU_MATCH_THRESHOLD = 0.45


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def match(gt_objects, dets):
    """
    Greedy IoU matching, highest overlap first.

    Returns (matched, n_missed). A matched record keeps the GT's true_range and
    identity but takes bbox geometry from the DETECTOR -- that substitution is
    the whole point, since detector box error is what propagates into range.

    Misses are counted, not discarded quietly: a detector that finds nothing in
    heavy rain has a range error of "undefined", and reporting only the error on
    frames where it succeeded would flatter it badly.
    """
    pairs = []
    for gi, g in enumerate(gt_objects):
        for di, d in enumerate(dets):
            v = iou(g["bbox"], d["bbox"])
            if v >= IOU_MATCH_THRESHOLD:
                pairs.append((v, gi, di))
    pairs.sort(reverse=True)

    used_g, used_d, out = set(), set(), []
    for v, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        g, d = gt_objects[gi], dets[di]
        x1, y1, x2, y2 = d["bbox"]
        out.append({
            **g,                                   # keeps true_range_m, type_id
            "bbox": [x1, y1, x2, y2],              # DETECTOR box
            "bbox_height_px": y2 - y1,
            "bbox_width_px": x2 - x1,
            "det_confidence": d["conf"],
            "det_iou": v,
            "gt_bbox": g["bbox"],                  # kept for error analysis
            "gt_bbox_height_px": g["bbox_height_px"],
        })
    return out, len(gt_objects) - len(used_g)


def run_condition(model, run_dir, conf_threshold, imgsz):
    with open(os.path.join(run_dir, "labels.json")) as f:
        labels = json.load(f)

    stems = sorted(labels.keys())
    paths = [os.path.join(run_dir, "rgb", s + ".png") for s in stems]

    results = model.predict(paths, conf=conf_threshold, imgsz=imgsz,
                            verbose=False, stream=True)

    out = {}
    n_matched = n_missed = n_spurious = 0

    for stem, res in zip(stems, results):
        dets = []
        if res.boxes is not None:
            for b in res.boxes:
                cls = int(b.cls.item())
                if cls not in COCO_VEHICLE:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                dets.append({"bbox": [x1, y1, x2, y2],
                             "conf": float(b.conf.item()),
                             "coco_class": COCO_VEHICLE[cls]})

        gt = labels[stem]["objects"]
        matched, missed = match(gt, dets)
        n_matched += len(matched)
        n_missed += missed
        n_spurious += max(0, len(dets) - len(matched))

        out[stem] = {
            "ego_speed_mps": labels[stem]["ego_speed_mps"],
            "sim_timestamp": labels[stem]["sim_timestamp"],
            "objects": matched,
            "n_gt": len(gt),
            "n_detections": len(dets),
            "n_missed": missed,
        }

    with open(os.path.join(run_dir, "detections.json"), "w", newline="\n") as f:
        json.dump(out, f)

    recall = n_matched / max(n_matched + n_missed, 1)
    return n_matched, n_missed, n_spurious, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="yolo11s.pt",
                    help="Downloaded on first use. yolo11n is faster, "
                         "yolo11m more accurate.")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence floor. Lower finds more in bad "
                         "conditions but adds false positives -- both effects "
                         "are part of what is being measured, so keep it fixed "
                         "across conditions.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None,
                    help="'0' for GPU, 'cpu' to force.")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    if args.device:
        model.to(args.device)

    from .config import sort_by_severity
    conds = sort_by_severity([
        d for d in os.listdir(args.dataset)
        if os.path.exists(os.path.join(args.dataset, d, "labels.json"))
    ])
    if not conds:
        raise SystemExit("No labelled runs found. Run src/labels.py first.")

    print(f"{args.model}  conf={args.conf}  imgsz={args.imgsz}\n")
    print(f"{'condition':16s} {'matched':>8s} {'missed':>7s} "
          f"{'spurious':>9s} {'recall':>8s}")

    recalls = {}
    for cond in conds:
        m, miss, sp, rec = run_condition(
            model, os.path.join(args.dataset, cond), args.conf, args.imgsz)
        recalls[cond] = rec
        print(f"{cond:16s} {m:8d} {miss:7d} {sp:9d} {rec:8.1%}")

    print("\nRecall spread across conditions: "
          f"{max(recalls.values()) - min(recalls.values()):.1%}")
    if max(recalls.values()) - min(recalls.values()) < 0.05:
        print("  Under 5% -- the detector is barely affected by these weather")
        print("  presets. The CARLA presets may be too mild; consider custom")
        print("  WeatherParameters with higher fog_density and precipitation.")
    else:
        print("  The detector responds to condition. Range error computed from")
        print("  these boxes will now vary across the matrix.")

    print("\nNext: python -m src.evaluate --dataset <dataset> --labels detections.json")


if __name__ == "__main__":
    main()
