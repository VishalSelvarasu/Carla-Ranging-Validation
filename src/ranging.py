from __future__ import annotations

import numpy as np

# Class height priors in metres. A real system does not know the exact height
# of the car in front of it -- it knows a class average. Using each actor's
# TRUE extent here would leak ground truth into the estimator and produce an
# error curve that no real vehicle could reproduce.
CLASS_HEIGHT_PRIOR_M = {
    "car": 1.50,
    "truck": 3.20,
    "van": 2.10,
    "bus": 3.30,
    "motorcycle": 1.10,
    "bicycle": 1.10,
    "default": 1.55,
}

# Spread of true heights within each class, metres (1 sigma). Drives the
# uncertainty estimate used for fusion.
CLASS_HEIGHT_SIGMA_M = {
    "car": 0.18,
    "truck": 0.55,
    "van": 0.25,
    "bus": 0.30,
    "motorcycle": 0.15,
    "bicycle": 0.15,
    "default": 0.30,
}


def classify(type_id: str) -> str:
    """
    Crude class from a CARLA blueprint id, e.g. 'vehicle.audi.tt' -> 'car'.

    Deliberately crude: a deployed system classifies from pixels and gets it
    wrong sometimes. Refine this only if you also measure how misclassification
    propagates into range error -- that is a finding, not a bug to hide.
    """
    t = type_id.lower()
    for key in ("truck", "bus", "van"):
        if key in t:
            return key
    if any(k in t for k in ("harley", "yamaha", "kawasaki", "vespa", "motorcycle")):
        return "motorcycle"
    if any(k in t for k in ("bike", "bicycle", "crossbike", "omafiets", "century")):
        return "bicycle"
    if any(k in t for k in ("carlacola", "firetruck", "ambulance", "sprinter")):
        return "truck"
    return "car"


def height_prior_range(box_height_px, fy, cls="car"):
    """Z = fy * H_real / h_px."""
    if box_height_px <= 0:
        return np.nan
    return fy * CLASS_HEIGHT_PRIOR_M.get(cls, CLASS_HEIGHT_PRIOR_M["default"]) \
        / box_height_px


def ground_plane_range(v_bottom_px, fy, cy, camera_height_m):
    """
    Z = fy * h_cam / (v_bottom - cy).

    Returns NaN when the box bottom sits at or above the horizon line: that
    implies a target on or above the camera plane, where the flat-ground
    assumption is meaningless. Returning a huge number instead of NaN here is
    a classic way to poison your mean error with a handful of samples.
    """
    dv = v_bottom_px - cy
    if dv <= 1.0:
        return np.nan
    return fy * camera_height_m / dv


def height_prior_sigma(z, box_height_px, cls="car", box_sigma_px=2.0):
    """
    1-sigma range uncertainty for the height-prior method.

    Z = fy*H/h, so relative errors add in quadrature:
        sigma_Z / Z = sqrt( (sigma_H/H)^2 + (sigma_h/h)^2 )
    """
    h_real = CLASS_HEIGHT_PRIOR_M.get(cls, CLASS_HEIGHT_PRIOR_M["default"])
    s_real = CLASS_HEIGHT_SIGMA_M.get(cls, CLASS_HEIGHT_SIGMA_M["default"])
    if box_height_px <= 0 or not np.isfinite(z):
        return np.inf
    rel = np.hypot(s_real / h_real, box_sigma_px / box_height_px)
    return abs(z) * rel


def ground_plane_sigma(z, v_bottom_px, cy, box_sigma_px=2.0):
    """
    Z = fy*h/(v-cy), so sigma_Z/Z = sigma_v / (v-cy).

    Note the quadratic blow-up with range: (v-cy) shrinks as targets recede,
    so this method's uncertainty grows as Z^2. Expect it to beat the height
    prior up close and lose badly past ~40 m.
    """
    dv = v_bottom_px - cy
    if dv <= 1.0 or not np.isfinite(z):
        return np.inf
    return abs(z) * (box_sigma_px / dv)


def fuse(z_a, s_a, z_b, s_b):
    """
    Inverse-variance fusion. Optimal ONLY if the two errors are independent,
    which they are not when both estimators read the same bad bounding box.
    Report this caveat with the result rather than quietly assuming it away.
    """
    vals, sigmas = [], []
    for z, s in ((z_a, s_a), (z_b, s_b)):
        if np.isfinite(z) and np.isfinite(s) and s > 0:
            vals.append(z)
            sigmas.append(s)
    if not vals:
        return np.nan, np.inf
    w = np.array([1.0 / s ** 2 for s in sigmas])
    return float(np.sum(w * np.array(vals)) / w.sum()), float(np.sqrt(1.0 / w.sum()))


def estimate_all(obj, intr, camera_height_m):
    """
    All three estimates for one detection.

    `obj` needs: bbox [x1,y1,x2,y2], bbox_height_px, type_id.
    Uses only what a real pipeline would have -- never obj['true_range_m'].
    """
    fy, cy = intr["fy"], intr["cy"]
    cls = classify(obj["type_id"])
    h_px = obj["bbox_height_px"]
    v_bottom = obj["bbox"][3]

    z_h = height_prior_range(h_px, fy, cls)
    s_h = height_prior_sigma(z_h, h_px, cls)

    z_g = ground_plane_range(v_bottom, fy, cy, camera_height_m)
    s_g = ground_plane_sigma(z_g, v_bottom, cy)

    z_f, s_f = fuse(z_h, s_h, z_g, s_g)

    return {
        "class": cls,
        "height_prior": z_h, "height_prior_sigma": s_h,
        "ground_plane": z_g, "ground_plane_sigma": s_g,
        "fused": z_f, "fused_sigma": s_f,
    }


ESTIMATORS = ("height_prior", "ground_plane", "fused")
