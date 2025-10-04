# sync_vehicles_only.py
# CARLA 0.9.14 — spawn vehicles and run in synchronous mode only.

import argparse
import random
import time
from typing import List
import carla

# ---------- helpers ----------

def safe_lane_center_spawns(world: carla.World) -> List[carla.Transform]:
    """Lane-centered, non-junction spawn transforms aligned with driving lanes."""
    m = world.get_map()
    out: List[carla.Transform] = []
    for sp in m.get_spawn_points():
        wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not wp or wp.is_junction:
            continue
        tr = carla.Transform(wp.transform.location, wp.transform.rotation)
        fwd = tr.get_forward_vector()
        tr.location += carla.Location(x=fwd.x * 1.5, y=fwd.y * 1.5, z=0.3)  # nudge forward, slight lift
        out.append(tr)
    if not out:  # fallback
        out = m.get_spawn_points()
        for tr in out:
            wp = m.get_waypoint(tr.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                tr.rotation.yaw = wp.transform.rotation.yaw
    return out

def spawn_vehicles_sync(world: carla.World, tm: carla.TrafficManager, n: int, tm_port: int, seed: int) -> List[int]:
    lib = world.get_blueprint_library()
    v_bps = [bp for bp in lib.filter("vehicle.*")
             if int(bp.get_attribute("number_of_wheels").as_int()) >= 4
             and all(x not in bp.id for x in ["bus", "firetruck", "ambulance", "carlacola", "t2"])]
    spawns = safe_lane_center_spawns(world)
    random.shuffle(spawns)
    count = min(n, len(spawns))

    ids: List[int] = []
    for i in range(count):
        bp = random.choice(v_bps)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

        actor = world.try_spawn_actor(bp, spawns[i])
        if not actor:
            continue

        actor.set_autopilot(True, tm_port)

        # Conservative, stable behavior (tweak as you like)
        try:
            tm.auto_lane_change(actor, False)
            tm.random_left_lanechange_percentage(actor, 0.0)
            tm.random_right_lanechange_percentage(actor, 0.0)
            tm.ignore_signs_percentage(actor, 0.0)
            tm.ignore_lights_percentage(actor, 0.0)
            tm.vehicle_percentage_speed_difference(actor, float(random.uniform(10.0, 25.0)))  # 10–25% slower
            tm.distance_to_leading_vehicle(actor, 8.0)
        except Exception:
            pass

        ids.append(actor.id)
        tr = actor.get_transform()
        print(f"[veh] {actor.type_id} id={actor.id} @ "
              f"({tr.location.x:.1f},{tr.location.y:.1f}) yaw={tr.rotation.yaw:.1f}")

    print(f"[veh] spawned {len(ids)}/{count}")
    return ids

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=5000)
    ap.add_argument("--vehicles", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dt", type=float, default=0.05, help="fixed delta seconds for sync mode (e.g., 0.05 = 20 FPS)")
    args = ap.parse_args()

    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"[world] current map: {world.get_map().name}")

    # ---- enable SYNC world ----
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(args.dt)
    world.apply_settings(settings)

    # ---- Traffic Manager in SYNC ----
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    tm.set_hybrid_physics_mode(True)              # smoother at scale; set False if you want full physics always
    tm.set_respawn_dormant_vehicles(True)
    tm.set_global_distance_to_leading_vehicle(10.0)
    tm.global_percentage_speed_difference(20.0)   # slower globally

    # One warm-up tick so TM and world are aligned
    world.tick()

    # ---- spawn vehicles ----
    veh_ids = spawn_vehicles_sync(world, tm, args.vehicles, args.tm_port, args.seed)

    try:
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            # advance exactly one frame
            world.tick()

            # (optional) quick telemetry
            if veh_ids:
                sample = veh_ids[:10]
                speeds = []
                for vid in sample:
                    a = world.get_actor(vid)
                    if not a:
                        continue
                    v = a.get_velocity()
                    kmh = (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5 * 3.6
                    speeds.append(kmh)
                if speeds:
                    print(f"[stat] avg_speed≈{sum(speeds)/len(speeds):.1f} km/h (veh={len(veh_ids)})")

    except KeyboardInterrupt:
        print("\n[run] interrupted by user.")
    finally:
        print("[teardown] destroying vehicles...")
        for vid in veh_ids:
            a = world.get_actor(vid)
            if a:
                try:
                    a.destroy()
                except Exception:
                    pass

        # restore world + TM to async defaults
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        world.apply_settings(original)
        print("[teardown] done.")

if __name__ == "__main__":
    main()
