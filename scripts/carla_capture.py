#!/usr/bin/env python3

import argparse
import json
import math
import os
import queue
import random
import sys
import time

import numpy as np

try:
    import carla
except ImportError:
    sys.exit("carla module not found. Use a Python 3.8 env: pip install carla==0.9.15")


# --------------------------------------------------------------------------
# Sensor plumbing
# --------------------------------------------------------------------------

class FrameMatchedSensors:
    """
    Queues every sensor's output and pulls until the frame number matches the
    world tick. Callbacks fire asynchronously even in synchronous mode, so
    reading "the latest image" silently mixes frames and misaligns depth
    against RGB by a tick or two.
    """

    def __init__(self, world, timeout=5.0):
        self.world = world
        self.timeout = timeout
        self.queues = {}
        self.sensors = {}

    def register(self, name, sensor):
        q = queue.Queue()
        sensor.listen(q.put)
        self.queues[name] = q
        self.sensors[name] = sensor

    def tick(self):
        """Advance one step, return {name: data} all on the same frame."""
        frame = self.world.tick()
        out = {}
        for name, q in self.queues.items():
            while True:
                data = q.get(timeout=self.timeout)
                if data.frame == frame:
                    out[name] = data
                    break
                if data.frame > frame:
                    raise RuntimeError(
                        f"sensor '{name}' overshot: got frame {data.frame}, "
                        f"expected {frame}. Sensor tick rate is likely mismatched "
                        f"with fixed_delta_seconds."
                    )
                # data.frame < frame -> stale, discard and keep draining
        return frame, out

    def destroy(self):
        for s in self.sensors.values():
            if s.is_alive:
                s.stop()
                s.destroy()


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------

def to_bgra_array(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    return arr.reshape((image.height, image.width, 4))


def rgb_to_array(image):
    """HxWx3 uint8, BGR order (OpenCV convention)."""
    return to_bgra_array(image)[:, :, :3].copy()


def depth_to_metres(image):
    """
    CARLA packs depth into RGB as a 24-bit int, far plane 1000 m:

        normalized = (R + G*256 + B*256*256) / (256**3 - 1)
        metres     = normalized * 1000

    LogarithmicDepth is a visualisation converter and is not metric -- don't
    use it for evaluation.
    """
    bgra = to_bgra_array(image).astype(np.float32)
    b, g, r = bgra[:, :, 0], bgra[:, :, 1], bgra[:, :, 2]
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    return (normalized * 1000.0).astype(np.float32)


DEPTH_SCALE_CM = 100.0        # metres -> centimetres
DEPTH_MAX_CM = 65535          # uint16 ceiling == 655.35 m


def depth_metres_to_uint16(depth_m):
    """
    float32 metres -> uint16 centimetres for 16-bit PNG storage. 1 cm
    precision, 655.35 m range, ~10x smaller than float32 .npy. Values at
    DEPTH_MAX_CM are clamped and should be treated as invalid/sky.
    """
    cm = np.round(depth_m * DEPTH_SCALE_CM)
    return np.clip(cm, 0, DEPTH_MAX_CM).astype(np.uint16)


def uint16_to_depth_metres(depth_u16):
    """Inverse of depth_metres_to_uint16."""
    return depth_u16.astype(np.float32) / DEPTH_SCALE_CM


def semantic_to_labels(image):
    """Raw tags live in the R channel. Don't apply CityScapesPalette."""
    return to_bgra_array(image)[:, :, 2].copy()


def camera_intrinsics(width, height, fov_deg):
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return {
        "fx": f, "fy": f,
        "cx": width / 2.0, "cy": height / 2.0,
        "width": width, "height": height, "fov_deg": fov_deg,
    }


def transform_to_dict(t):
    return {
        "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
        "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw,
                     "roll": t.rotation.roll},
    }


def actor_gt(actor):
    """
    Per-actor ground truth for label generation. bbox_offset is in the actor's
    local frame, not world -- vehicles carry a non-zero z offset (about half
    the body height) and dropping it puts every projected box a metre low.
    """
    bb = actor.bounding_box
    v = actor.get_velocity()
    return {
        "id": actor.id,
        "type_id": actor.type_id,
        "transform": transform_to_dict(actor.get_transform()),
        "bbox_extent": {"x": bb.extent.x, "y": bb.extent.y, "z": bb.extent.z},
        "bbox_offset": {"x": bb.location.x, "y": bb.location.y, "z": bb.location.z},
        "velocity": {"x": v.x, "y": v.y, "z": v.z},
        "speed_mps": math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--map", default="Town10HD_Opt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30,
                    help="Ticks to discard before capture (physics settling).")
    ap.add_argument("--traffic", type=int, default=20)
    ap.add_argument("--lead-distances", default="90",
                    help="Metres ahead of ego to place stationary targets. One "
                         "far target is usually better than several near ones: "
                         "as the ego closes, range sweeps through every "
                         "evaluation bin on its own.")
    ap.add_argument("--lead-mode", choices=["stationary", "moving"],
                    default="stationary",
                    help="stationary: ego approaches a standing vehicle at "
                         "constant speed (Euro NCAP CCRs shape). moving: "
                         "targets drive slower than the ego. Moving gives "
                         "closing speeds around 1 m/s, so TTC never approaches "
                         "the brake threshold and no events are generated.")
    ap.add_argument("--lead-slowdown", type=float, default=25.0,
                    help="Percent slower than the speed limit, moving mode only.")
    ap.add_argument("--ego-ignores-vehicles", action="store_true", default=True,
                    help="Stop the ego slowing for the target. Autopilot holds "
                         "a safe gap, which is right for a driver and useless "
                         "for measuring when a brake should have fired. The "
                         "collision at the end is expected -- the frames before "
                         "it are the TTC events.")
    ap.add_argument("--stop-on-collision", action="store_true", default=True,
                    help="End capture on impact. Post-collision frames are "
                         "wreckage, not measurements.")
    ap.add_argument("--weather", default="ClearNoon",
                    help="Any carla.WeatherParameters preset name.")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--delta", type=float, default=0.05,
                    help="Fixed timestep (s).")
    ap.add_argument("--no-rendering", action="store_true",
                    help="Disable the GPU pipeline entirely. No cameras spawned, "
                         "trajectory metadata only. Runs 10-50x faster and uses "
                         "no VRAM. Use this for scenario sweeps and closed-loop "
                         "phases where you don't need pixels.")
    ap.add_argument("--strip-layers", action="store_true", default=True,
                    help="Unload foliage/decals/parked vehicles to cut VRAM.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.join(args.out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "depth"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "semantic"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "meta"), exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.load_world(args.map)

    original_settings = world.get_settings()
    tm = client.get_trafficmanager(args.tm_port)
    sensors = FrameMatchedSensors(world)
    actors = []
    collisions = []

    try:
        # ---- determinism block -------------------------------------------
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.delta
        # substepping must divide delta cleanly or physics drifts run-to-run
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)

        if args.strip_layers:
            for layer in (carla.MapLayer.Foliage,
                          carla.MapLayer.Decals,
                          carla.MapLayer.ParkedVehicles,
                          carla.MapLayer.Particles):
                try:
                    world.unload_map_layer(layer)
                except RuntimeError:
                    pass  # _Opt maps only; plain maps have no layers to unload

        # forgetting this breaks reproducibility
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(args.seed)
        world.set_pedestrians_seed(args.seed)
        # ------------------------------------------------------------------

        weather = getattr(carla.WeatherParameters, args.weather, None)
        if weather is None:
            sys.exit(f"Unknown weather preset: {args.weather}")
        world.set_weather(weather)

        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        # sort for stable ordering -- spawn point order is not guaranteed
        spawn_points.sort(key=lambda sp: (
            sp.location.x, sp.location.y, sp.location.z))
        rng = random.Random(args.seed)
        rng.shuffle(spawn_points)

        # ---- ego -----------------------------------------------------------
        ego_bp = bp_lib.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "ego")
        ego = world.spawn_actor(ego_bp, spawn_points[0])
        actors.append(ego)
        ego.set_autopilot(True, args.tm_port)
        tm.ignore_lights_percentage(ego, 100.0)   # no stopping at reds mid-run
        tm.auto_lane_change(ego, False)
        try:
            tm.set_route(ego, ['Straight'] * 50)
        except Exception:
            pass  # set_route may not exist on older TM builds

        if args.ego_ignores_vehicles:
            tm.ignore_vehicles_percentage(ego, 100.0)
            tm.distance_to_leading_vehicle(ego, 0.0)

        vehicle_bps = sorted(bp_lib.filter("vehicle.*"), key=lambda b: b.id)

        # ---- targets -------------------------------------------------------
        # Placed on the ego's own lane by waypoint. Random town-wide traffic
        # almost never lands a vehicle in the forward view -- 30 NPCs once gave
        # 19 behind the camera and nothing in frame.
        ego_wp = world.get_map().get_waypoint(spawn_points[0].location)
        lead_distances = [float(d)
                          for d in args.lead_distances.split(",") if d]
        n_lead = 0
        for dist in lead_distances:
            nxt = ego_wp.next(dist)
            if not nxt:
                continue
            tf = nxt[0].transform
            tf.location.z += 0.3          # keep it out of the road mesh
            bp = vehicle_bps[rng.randrange(len(vehicle_bps))]
            lead = world.try_spawn_actor(bp, tf)
            if lead is None:
                continue
            actors.append(lead)

            if args.lead_mode == "moving":
                lead.set_autopilot(True, args.tm_port)
                tm.vehicle_percentage_speed_difference(
                    lead, args.lead_slowdown)
                tm.auto_lane_change(lead, False)
                try:
                    tm.set_route(lead, ['Straight'] * 50)
                except Exception:
                    pass
            else:
                # No autopilot, no control input: it stands still. Closing
                # speed then equals ego speed, which is what puts TTC in
                # range of the brake threshold.
                lead.apply_control(carla.VehicleControl(
                    throttle=0.0, brake=1.0, hand_brake=True))
            n_lead += 1
        print(f"targets: {n_lead}/{len(lead_distances)} spawned "
              f"({args.lead_mode})")

        # ---- background traffic --------------------------------------------
        for sp in spawn_points[1:1 + args.traffic]:
            bp = vehicle_bps[rng.randrange(len(vehicle_bps))]
            npc = world.try_spawn_actor(bp, sp)
            if npc is not None:
                npc.set_autopilot(True, args.tm_port)
                actors.append(npc)

        # ---- sensors (all share one transform -> pixel-aligned) -------------
        cam_tf = carla.Transform(carla.Location(x=1.5, z=1.6))
        intrinsics = camera_intrinsics(args.width, args.height, args.fov)

        if not args.no_rendering:
            def make_cam(bp_name):
                bp = bp_lib.find(bp_name)
                bp.set_attribute("image_size_x", str(args.width))
                bp.set_attribute("image_size_y", str(args.height))
                bp.set_attribute("fov", str(args.fov))
                bp.set_attribute("sensor_tick", "0.0")  # every tick
                return world.spawn_actor(bp, cam_tf, attach_to=ego)

            cam_rgb = make_cam("sensor.camera.rgb")
            cam_depth = make_cam("sensor.camera.depth")
            cam_sem = make_cam("sensor.camera.semantic_segmentation")
            actors.extend([cam_rgb, cam_depth, cam_sem])

            sensors.register("rgb", cam_rgb)
            sensors.register("depth", cam_depth)
            sensors.register("semantic", cam_sem)
        else:
            print("no-rendering mode: cameras skipped, trajectory metadata only.")

        # Collision sensor is event-driven, not per-tick, so it stays out of
        # FrameMatchedSensors -- that class waits for data every frame and
        # would block forever on a sensor that only fires on impact.
        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego)
        collision_sensor.listen(lambda e: collisions.append(e.frame))
        actors.append(collision_sensor)

        # newline="\n" on the JSON writes: Windows text mode emits CRLF, which
        # makes a dataset captured here byte-different from the same dataset on
        # Linux. Doesn't affect determinism on one machine, but keeps runs
        # comparable across them.
        run_cfg = {
            "seed": args.seed, "map": args.map, "weather": args.weather,
            "fixed_delta_seconds": args.delta, "frames": args.frames,
            "warmup": args.warmup, "traffic": args.traffic,
            "lead_distances": lead_distances,
            "lead_mode": args.lead_mode,
            "lead_slowdown_pct": args.lead_slowdown,
            "lead_spawned": n_lead,
            "ego_ignores_vehicles": bool(args.ego_ignores_vehicles),
            "intrinsics": intrinsics,
            "camera_transform": transform_to_dict(cam_tf),
            "carla_version": client.get_server_version(),
        }
        with open(os.path.join(args.out, "run_config.json"), "w", newline="\n") as f:
            json.dump(run_cfg, f, indent=2, sort_keys=True)

        import cv2  # imported late so --help works without opencv

        for _ in range(args.warmup):
            sensors.tick()

        n_written = 0
        for i in range(args.frames):
            frame, data = sensors.tick()
            stem = f"{i:06d}"

            v = ego.get_velocity()
            meta = {
                "index": i,
                "carla_frame": frame,
                "sim_timestamp": world.get_snapshot().timestamp.elapsed_seconds,
                "ego_transform": transform_to_dict(ego.get_transform()),
                "ego_speed_mps": math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),
            }

            if not args.no_rendering:
                # Sample the camera pose on this tick. The mount transform is
                # relative to the ego; projection needs the world pose.
                meta["camera_world_transform"] = transform_to_dict(
                    cam_rgb.get_transform())

            # All vehicles except ego, for label generation downstream.
            meta["actors"] = [
                actor_gt(a) for a in world.get_actors().filter("vehicle.*")
                if a.id != ego.id
            ]

            if not args.no_rendering:
                rgb = rgb_to_array(data["rgb"])
                depth_m = depth_to_metres(data["depth"])
                depth_u16 = depth_metres_to_uint16(depth_m)
                sem = semantic_to_labels(data["semantic"])

                cv2.imwrite(os.path.join(args.out, "rgb", stem + ".png"), rgb)
                cv2.imwrite(os.path.join(
                    args.out, "semantic", stem + ".png"), sem)
                # 16-bit single-channel PNG; cv2 preserves uint16 on .png
                cv2.imwrite(os.path.join(
                    args.out, "depth", stem + ".png"), depth_u16)

                valid = depth_u16 < DEPTH_MAX_CM
                meta["sim_timestamp"] = data["rgb"].timestamp
                meta["depth_stats_m"] = {
                    "min": float(depth_m[valid].min()) if valid.any() else None,
                    "median": float(np.median(depth_m[valid])) if valid.any() else None,
                    "valid_fraction": float(valid.mean()),
                }

            with open(os.path.join(args.out, "meta", stem + ".json"), "w", newline="\n") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
            n_written = i + 1

            if i % 50 == 0:
                print(f"[{i}/{args.frames}] frame={frame} "
                      f"speed={meta['ego_speed_mps']:.1f} m/s")

            if args.stop_on_collision and collisions:
                print(f"collision at frame {i} -- stopping capture "
                      f"({n_written} frames kept)")
                break

        # Frame count varies per condition once collisions end runs early.
        # Downstream comparisons need to know, so it goes in run_config.
        run_cfg["frames_written"] = n_written
        run_cfg["collided"] = bool(collisions)
        with open(os.path.join(args.out, "run_config.json"), "w", newline="\n") as f:
            json.dump(run_cfg, f, indent=2, sort_keys=True)

        print(f"Done. Wrote {n_written} frames to {args.out}")

    finally:
        # Teardown order matters. Destroying a sensor while callbacks are still
        # in flight kills the process in CARLA's C++ layer (exit 0xC0000409)
        # with no Python traceback, so nothing can catch it. Sequence is
        # stop -> drain -> leave sync mode -> batch destroy.
        for s in list(sensors.sensors.values()):
            try:
                s.stop()          # detach callbacks first
            except Exception:
                pass
        try:
            collision_sensor.stop()
        except Exception:
            pass
        time.sleep(0.5)           # let in-flight callbacks drain

        # Leave sync mode before destroying: in sync mode the server blocks
        # waiting for a tick that isn't coming once teardown starts.
        try:
            tm.set_synchronous_mode(False)
        except Exception as e:
            print(f"  (tm restore: {e})")
        try:
            world.apply_settings(original_settings)
        except Exception as e:
            print(f"  (settings restore: {e})")
        time.sleep(0.5)

        # One batch RPC instead of N individual destroys, which avoids the
        # per-actor race.
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in actors])
            time.sleep(0.5)
        except Exception as e:
            print(f"  (batch destroy: {e})")

        print("Cleaned up; server restored to async mode.")


if __name__ == "__main__":
    main()
