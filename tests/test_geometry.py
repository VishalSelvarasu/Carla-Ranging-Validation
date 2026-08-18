from src.ranging import estimate_all, height_prior_range, ground_plane_range
from src.labels import (bbox_corners_world, world_to_camera, project,
                        intrinsic_matrix, transform_matrix)
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


INTR = {"fx": 400.0, "fy": 400.0, "cx": 400.0, "cy": 300.0,
        "width": 800, "height": 600, "fov_deg": 90.0}
CAM_H = 1.6
CAM = {"location": {"x": 0.0, "y": 0.0, "z": CAM_H},
       "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}}


def make_actor(x, y=0.0, yaw=0.0, half_len=2.4, half_wid=1.0, half_hgt=0.75):
    return {"id": 1, "type_id": "vehicle.audi.tt", "speed_mps": 0.0,
            "transform": {"location": {"x": x, "y": y, "z": 0.0},
                          "rotation": {"pitch": 0.0, "yaw": yaw, "roll": 0.0}},
            "bbox_extent": {"x": half_len, "y": half_wid, "z": half_hgt},
            "bbox_offset": {"x": 0.0, "y": 0.0, "z": half_hgt}}


def projected(actor):
    corners_c = world_to_camera(bbox_corners_world(actor), CAM)
    return corners_c, project(corners_c, intrinsic_matrix(INTR))


def test_identity_transform():
    np.testing.assert_allclose(
        transform_matrix({"location": {"x": 0, "y": 0, "z": 0},
                          "rotation": {"pitch": 0, "yaw": 0, "roll": 0}}),
        np.identity(4), atol=1e-12)


def test_forward_axis_maps_to_camera_z():
    """CARLA +x (forward) must become camera +z (depth), not +x."""
    cc, _ = projected(make_actor(50.0))
    assert cc[2].min() > 40.0
    assert abs(cc[0].mean()) < 2.0   # centred laterally


def test_lateral_sign_convention():
    """CARLA +y is RIGHT. A target at +y must project right of cx."""
    _, uv = projected(make_actor(30.0, y=5.0))
    assert uv[0].mean() > INTR["cx"] + 20
    _, uv = projected(make_actor(30.0, y=-5.0))
    assert uv[0].mean() < INTR["cx"] - 20


def test_box_bottom_sits_on_ground_plane():
    """v_bottom = cy + fy*h_cam/Z_near, from the flat-ground model."""
    for dist in (10.0, 30.0, 60.0):
        cc, uv = projected(make_actor(dist))
        z_near = cc[2].min()
        expected = INTR["cy"] + INTR["fy"] * CAM_H / z_near
        assert abs(uv[1].max() - expected) < 0.5


def test_range_definition_is_nearest_face():
    """
    Regression test for a real bug.

    The 2D box is sized by the NEAR face of the 3D box. Defining true range as
    the centroid instead injects a bias of ~half the vehicle length -- about
    2.4 m, larger than the weather effect being measured -- and it presents as
    a uniform under-read that looks like a bad height prior.
    """
    actor = make_actor(40.0, half_len=2.4)
    cc, _ = projected(actor)
    near, centre = cc[2].min(), cc[2].mean()

    assert abs(centre - near - 2.4) < 0.1, "near/centre gap should be half-length"

    est = estimate_all(
        {"bbox": [0, 0, 0, int(INTR["cy"] + INTR["fy"] * CAM_H / near)],
         "bbox_height_px": INTR["fy"] * 1.5 / near,
         "type_id": "vehicle.audi.tt"}, INTR, CAM_H)

    assert abs(est["height_prior"] -
               near) < 0.5, "estimator tracks the near face"
    assert abs(est["height_prior"] - centre) > 1.5, "and NOT the centroid"


@pytest.mark.parametrize("dist", [10.0, 20.0, 40.0, 60.0, 80.0])
def test_estimators_accurate_under_ideal_conditions(dist):
    actor = make_actor(dist)
    cc, uv = projected(actor)
    true_range = cc[2].min()
    obj = {"bbox": [int(uv[0].min()), int(uv[1].min()),
                    int(uv[0].max()), int(uv[1].max())],
           "bbox_height_px": uv[1].max() - uv[1].min(),
           "type_id": "vehicle.audi.tt"}
    est = estimate_all(obj, INTR, CAM_H)
    for name in ("height_prior", "ground_plane", "fused"):
        rel = abs(est[name] - true_range) / true_range
        assert rel < 0.05, f"{name} off by {rel:.1%} at {true_range:.0f} m"


def test_ground_plane_degrades_faster_with_distance():
    """
    ground_plane uncertainty grows as Z^2, height_prior as Z. The crossover is
    the single most useful finding the estimator comparison can produce, so
    lock in the ordering.
    """
    sig_g, sig_h = [], []
    for dist in (10.0, 80.0):
        cc, uv = projected(make_actor(dist))
        obj = {"bbox": [int(uv[0].min()), int(uv[1].min()),
                        int(uv[0].max()), int(uv[1].max())],
               "bbox_height_px": uv[1].max() - uv[1].min(),
               "type_id": "vehicle.audi.tt"}
        est = estimate_all(obj, INTR, CAM_H)
        sig_g.append(est["ground_plane_sigma"])
        sig_h.append(est["height_prior_sigma"])

    assert sig_g[0] < sig_h[0], "ground_plane should win up close"
    assert (sig_g[1] / sig_g[0]) > (sig_h[1] / sig_h[0]), \
        "ground_plane uncertainty should grow faster with range"


def test_invalid_inputs_return_nan_not_garbage():
    """
    A box bottom above the horizon breaks the flat-ground assumption. It must
    return NaN, not a huge finite number -- a handful of 10^6 m values will
    silently dominate any mean error you compute.
    """
    assert np.isnan(ground_plane_range(
        INTR["cy"] - 10, 400.0, INTR["cy"], CAM_H))
    assert np.isnan(ground_plane_range(INTR["cy"], 400.0, INTR["cy"], CAM_H))
    assert np.isnan(height_prior_range(0, 400.0, "car"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
