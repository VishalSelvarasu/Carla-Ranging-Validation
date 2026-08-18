from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np

# CARLA 0.9.14+ splits vehicles across six semantic classes. Tag 10 is Terrain
# in this tag set. Do NOT verify against carla.CityObjectLabel -- that enum is
# for 3D bbox queries and uses different numbering; checking it returns 10 and
# falsely confirms the old value.
# Car, Truck, Bus, Train, Motorcycle, Bicycle
VEHICLE_TAGS = (14, 15, 16, 17, 18, 19)
_VEHICLE_TAGS_ARR = np.array(VEHICLE_TAGS, dtype=np.uint8)
SEM_TAG_VEHICLE = VEHICLE_TAGS[0]         # back-compat for diagnostics

DEPTH_SCALE_CM = 100.0
DEPTH_MAX_CM = 65535

# A box smaller than this is not something a monocular ranging method can be
# fairly evaluated on -- quantisation of the pixel height dominates the error.
MIN_BOX_HEIGHT_PX = 16
MIN_BOX_AREA_PX = 400
MIN_VISIBILITY = 0.30
MAX_RANGE_M = 100.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def transform_matrix(tf: dict) -> np.ndarray:
    """
    Build a 4x4 actor->world matrix from a serialised CARLA transform.

    CARLA/Unreal uses a LEFT-handed coordinate system: x forward, y right,
    z up, with rotations in degrees. This reproduces carla.Transform.get_matrix()
    exactly -- do not substitute a scipy right-handed rotation here, the y-axis
    sign convention differs and yaw will come out mirrored.
    """
    loc = tf["location"]
    rot = tf["rotation"]
    cy, sy = math.cos(math.radians(rot["yaw"])), math.sin(
        math.radians(rot["yaw"]))
    cr, sr = math.cos(math.radians(rot["roll"])), math.sin(
        math.radians(rot["roll"]))
    cp, sp = math.cos(math.radians(rot["pitch"])), math.sin(
        math.radians(rot["pitch"]))

    m = np.identity(4)
    m[0, 3], m[1, 3], m[2, 3] = loc["x"], loc["y"], loc["z"]
    m[0, 0] = cp * cy
    m[0, 1] = cy * sp * sr - sy * cr
    m[0, 2] = -cy * sp * cr - sy * sr
    m[1, 0] = sy * cp
    m[1, 1] = sy * sp * sr + cy * cr
    m[1, 2] = -sy * sp * cr + cy * sr
    m[2, 0] = sp
    m[2, 1] = -cp * sr
    m[2, 2] = cp * cr
    return m


def bbox_corners_world(actor: dict) -> np.ndarray:
    """Eight bbox corners in world coordinates, as a 4xN homogeneous array."""
    e = actor["bbox_extent"]
    o = actor["bbox_offset"]
    signs = np.array([[sx, sy, sz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float64)
    local = signs * np.array([e["x"], e["y"], e["z"]])
    local += np.array([o["x"], o["y"], o["z"]])  # bbox offset in actor frame
    homog = np.concatenate([local, np.ones((8, 1))], axis=1).T  # 4x8
    return transform_matrix(actor["transform"]) @ homog


def world_to_camera(points_world: np.ndarray, cam_tf: dict) -> np.ndarray:
    """
    World -> standard camera frame (x right, y down, z forward).

    Two steps, and the second is the one people forget: CARLA's sensor frame
    is still UE-convention (x forward, y right, z up), so after applying the
    inverse camera transform you must still remap axes.
    """
    world_to_sensor = np.linalg.inv(transform_matrix(cam_tf))
    p = world_to_sensor @ points_world              # UE sensor frame
    return np.stack([p[1], -p[2], p[0]], axis=0)    # -> standard camera frame


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """3xN camera-frame points -> 2xN pixel coordinates."""
    uvw = K @ points_cam
    return uvw[:2] / uvw[2]


def intrinsic_matrix(intr: dict) -> np.ndarray:
    return np.array([[intr["fx"], 0.0, intr["cx"]],
                     [0.0, intr["fy"], intr["cy"]],
                     [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

def visibility_fraction(box, depth_m, semantic, true_range, tol_frac=0.10):
    """
    Fraction of the projected box that is genuinely this vehicle's surface.

    A pixel counts as visible when it is BOTH tagged as vehicle AND at a depth
    consistent with this actor's range. The semantic check alone is not enough:
    two overlapping cars are both tagged vehicle, and without the depth test the
    nearer one steals the farther one's pixels and both look fully visible.

    Tolerance scales with range because depth error and vehicle extent both
    grow with distance -- a fixed metre tolerance rejects the far side of a
    nearby car and accepts a different car entirely at 80 m.
    """
    x1, y1, x2, y2 = box
    d = depth_m[y1:y2, x1:x2]
    s = semantic[y1:y2, x1:x2]
    if d.size == 0:
        return 0.0

    tol = max(1.5, true_range * tol_frac)
    is_vehicle = np.isin(s, _VEHICLE_TAGS_ARR)
    depth_ok = np.abs(d - true_range) < tol
    return float((is_vehicle & depth_ok).mean())


def clip_box(u, v, width, height):
    """Clip to image bounds. Returns None if nothing survives."""
    x1 = int(np.clip(np.floor(u.min()), 0, width - 1))
    x2 = int(np.clip(np.ceil(u.max()), 0, width - 1))
    y1 = int(np.clip(np.floor(v.min()), 0, height - 1))
    y2 = int(np.clip(np.ceil(v.max()), 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Per-frame label generation
# ---------------------------------------------------------------------------

def labels_for_frame(meta, depth_m, semantic, K, intr):
    width, height = intr["width"], intr["height"]
    cam_tf = meta["camera_world_transform"]
    out = []

    for actor in meta.get("actors", []):
        corners_w = bbox_corners_world(actor)
        corners_c = world_to_camera(corners_w, cam_tf)

        # Reject anything not fully in front of the image plane. Partial
        # rejection is deliberate: a box straddling z=0 projects to garbage
        # coordinates that would silently poison the label set.
        if np.any(corners_c[2] < 0.5):
            continue

        # True range = forward distance to the NEAREST face of the box, not
        # its centre.
        #
        # This is not a stylistic choice. The projected 2D box is sized by the
        # near face (it is closer, so it projects larger), so any estimator
        # reading that box is measuring the near face. Scoring it against the
        # centre injects a constant bias of half the vehicle length -- about
        # 2.4 m for a car, which is LARGER than the weather effect this
        # project exists to measure. It would appear as every estimator
        # under-reading in every condition, and would look like a bad height
        # prior rather than a definitional error.
        #
        # It is also the right quantity physically: TTC and collision are
        # about the nearest point of the obstacle, not its centroid.
        true_range = float(corners_c[2].min())
        centre_range = float(corners_c[2].mean())
        if not (2.0 < true_range < MAX_RANGE_M):
            continue

        uv = project(corners_c, K)
        box = clip_box(uv[0], uv[1], width, height)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        if (y2 - y1) < MIN_BOX_HEIGHT_PX or (x2 - x1) * (y2 - y1) < MIN_BOX_AREA_PX:
            continue

        vis = visibility_fraction(box, depth_m, semantic, true_range)
        if vis < MIN_VISIBILITY:
            continue

        out.append({
            "actor_id": actor["id"],
            "type_id": actor["type_id"],
            "bbox": [x1, y1, x2, y2],
            "bbox_height_px": y2 - y1,
            "bbox_width_px": x2 - x1,
            "true_range_m": true_range,        # nearest face -- use this
            "centre_range_m": centre_range,     # centroid, for reference only
            "visibility": vis,
            # Real-world extents, needed by the height-prior estimator. Extent
            # is a half-size, hence the factor of two.
            "real_height_m": 2.0 * actor["bbox_extent"]["z"],
            "real_width_m": 2.0 * actor["bbox_extent"]["y"],
            "target_speed_mps": actor["speed_mps"],
        })

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_depth(path):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    if d.dtype != np.uint16:
        raise ValueError(
            f"{path} is {d.dtype}, expected uint16. cv2 silently downcasts "
            "16-bit PNGs unless you pass IMREAD_UNCHANGED at write AND read.")
    return d.astype(np.float32) / DEPTH_SCALE_CM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. ./dataset/ClearNoon")
    ap.add_argument("--out", default=None, help="Default: <run>/labels.json")
    ap.add_argument("--debug-frame", type=int, default=None,
                    help="Write an annotated PNG for this frame and exit.")
    args = ap.parse_args()

    with open(os.path.join(args.run, "run_config.json")) as f:
        cfg = json.load(f)
    intr = cfg["intrinsics"]
    K = intrinsic_matrix(intr)

    meta_dir = os.path.join(args.run, "meta")
    frames = sorted(os.listdir(meta_dir))
    all_labels = {}
    n_obj = 0

    for fn in frames:
        stem = os.path.splitext(fn)[0]
        if args.debug_frame is not None and int(stem) != args.debug_frame:
            continue

        with open(os.path.join(meta_dir, fn)) as f:
            meta = json.load(f)
        depth_m = load_depth(os.path.join(args.run, "depth", stem + ".png"))
        semantic = cv2.imread(os.path.join(args.run, "semantic", stem + ".png"),
                              cv2.IMREAD_UNCHANGED)
        if semantic.ndim == 3:
            semantic = semantic[:, :, 2]  # tags live in the R channel

        objs = labels_for_frame(meta, depth_m, semantic, K, intr)
        all_labels[stem] = {
            "ego_speed_mps": meta["ego_speed_mps"],
            "sim_timestamp": meta["sim_timestamp"],
            "objects": objs,
        }
        n_obj += len(objs)

        if args.debug_frame is not None:
            img = cv2.imread(os.path.join(args.run, "rgb", stem + ".png"))
            for o in objs:
                x1, y1, x2, y2 = o["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img,
                            f"{o['true_range_m']:.1f}m v={o['visibility']:.2f}",
                            (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 0), 1)
            dbg = os.path.join(args.run, f"debug_{stem}.png")
            cv2.imwrite(dbg, img)
            print(f"Wrote {dbg} with {len(objs)} objects. "
                  "Open it. If boxes float above or below the cars, the bbox "
                  "offset or the axis remap is wrong -- fix that before "
                  "generating the full label set.")
            return

    out = args.out or os.path.join(args.run, "labels.json")
    with open(out, "w", newline="\n") as f:
        json.dump(all_labels, f)

    n_frames = len(all_labels)
    print(f"{out}: {n_obj} objects across {n_frames} frames "
          f"({n_obj / max(n_frames, 1):.1f} per frame)")
    if n_obj == 0:
        print("Zero objects. Check SEM_TAG_VEHICLE first -- that is the "
              "usual cause, and it fails silently.")


if __name__ == "__main__":
    main()
