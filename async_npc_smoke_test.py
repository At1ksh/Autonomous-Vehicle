# async_npc_smoke_test.py
# Current map only. Force ASYNC world. Spawn vehicles + walkers reliably.
# Walkers are kinematic (physics OFF), controllers started, goals assigned w/ min distance.

import argparse
import random
import time
from typing import List, Tuple

import carla
from carla import command as carla_cmd


# ---------- helpers ----------

def safe_lane_center_spawns(world: carla.World, min_ahead: float = 8.0) -> List[carla.Transform]:
    """Lane-centered spawns; avoid obvious junction starts; fall back if too few."""
    m = world.get_map()
    out: List[carla.Transform] = []

    raw = m.get_spawn_points()
    for sp in raw:
        wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not wp:
            continue
        if wp.is_junction:
            continue
        tr = carla.Transform(wp.transform.location, wp.transform.rotation)
        fwd = tr.get_forward_vector()
        tr.location += carla.Location(x=fwd.x * 1.5, y=fwd.y * 1.5, z=0.3)
        out.append(tr)

    if len(out) == 0:
        out = raw
        for tr in out:
            wp = m.get_waypoint(tr.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                tr.rotation.yaw = wp.transform.rotation.yaw
    return out


def spawn_vehicles(world: carla.World, tm: carla.TrafficManager, n: int, tm_port: int) -> List[int]:
    lib = world.get_blueprint_library()
    v_bps = [bp for bp in lib.filter("vehicle.*")
             if int(bp.get_attribute("number_of_wheels").as_int()) >= 4
             and all(bad not in bp.id for bad in ["bus", "firetruck", "ambulance", "carlacola", "t2"])]

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

        tr = spawns[i]
        actor = world.try_spawn_actor(bp, tr)
        if not actor:
            continue

        actor.set_autopilot(True, tm_port)
        try:
            tm.auto_lane_change(actor, False)
            tm.random_left_lanechange_percentage(actor, 0.0)
            tm.random_right_lanechange_percentage(actor, 0.0)
            tm.ignore_signs_percentage(actor, 0.0)
            tm.ignore_lights_percentage(actor, 0.0)
            tm.vehicle_percentage_speed_difference(actor, float(random.uniform(10.0, 25.0)))
            tm.distance_to_leading_vehicle(actor, 8.0)
        except Exception:
            pass

        ids.append(actor.id)
        print(f"[veh] {actor.type_id} id={actor.id} @ ({tr.location.x:.1f},{tr.location.y:.1f}) yaw={tr.rotation.yaw:.1f}")

    print(f"[veh] spawned {len(ids)}/{count}")
    return ids


def spawn_walkers(world: carla.World, client: carla.Client, n: int, min_goal_dist: float = 6.0) -> Tuple[List[int], List[int]]:
    """Batch-spawn walkers + controllers; walkers physics OFF; start → 2 ticks → assign far-enough goals."""
    lib = world.get_blueprint_library()
    walker_bps = lib.filter("walker.pedestrian.*")
    controller_bp = lib.find("controller.ai.walker")

    # oversample nav locations; keep first n valid
    spawn_points: List[carla.Transform] = []
    for _ in range(n * 3):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))
        if len(spawn_points) >= n:
            break
    spawn_points = spawn_points[:n]

    # batch spawn walkers
    batch = []
    for sp in spawn_points:
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "walker")
        batch.append(carla_cmd.SpawnActor(bp, sp))
    results = client.apply_batch_sync(batch, True)
    walker_ids = [r.actor_id for r in results if not r.error and r.actor_id != 0]

    # IMPORTANT: AI walkers should be kinematic → physics OFF (prevents spinning/floating)
    for wid in walker_ids:
        w = world.get_actor(wid)
        if not w:
            continue
        try:
            w.set_simulate_physics(False)
        except Exception:
            pass

    # batch spawn controllers attached to walkers
    batch = [carla_cmd.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
    results = client.apply_batch_sync(batch, True)
    controller_ids = [r.actor_id for r in results if not r.error and r.actor_id != 0]

    # start controllers
    for cid in controller_ids:
        c = world.get_actor(cid)
        if not c:
            continue
        try:
            c.start()
            c.set_max_speed(1.4 + random.random())
        except Exception:
            pass

    # give controllers 2 frames to boot
    try:
        world.wait_for_tick(); world.wait_for_tick()
    except Exception:
        time.sleep(0.1)

    # assign first destination far enough away
    for cid in controller_ids:
        c = world.get_actor(cid)
        if not c:
            continue
        walker = c.get_parent() if hasattr(c, "get_parent") else None
        origin = walker.get_location() if walker else None
        for _ in range(10):
            dest = world.get_random_location_from_navigation()
            if dest and origin and origin.distance(dest) >= min_goal_dist:
                try:
                    c.go_to_location(dest)
                except Exception:
                    pass
                break

    print(f"[walk] spawned {len(walker_ids)} walkers + {len(controller_ids)} controllers")
    return walker_ids, controller_ids


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=5000)
    ap.add_argument("--vehicles", type=int, default=20)
    ap.add_argument("--walkers", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cross", type=float, default=0.8, help="Pedestrian road-crossing factor [0..1]")
    ap.add_argument("--sleep", type=float, default=0.05, help="Wall-clock sleep per loop (sec)")
    args = ap.parse_args()

    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"[world] current map: {world.get_map().name}")

    # Force ASYNC
    s = world.get_settings()
    s.synchronous_mode = False
    s.fixed_delta_seconds = None
    world.apply_settings(s)

    # Pedestrian behavior (must be set before spawning)
    try:
        world.set_pedestrians_cross_factor(float(args.cross))
        world.set_pedestrians_seed(int(args.seed))
    except Exception:
        pass

    # Traffic Manager in async + conservative global behavior
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(False)
    tm.set_random_device_seed(args.seed)
    tm.set_hybrid_physics_mode(True)
    tm.set_respawn_dormant_vehicles(True)
    tm.set_global_distance_to_leading_vehicle(10.0)
    tm.global_percentage_speed_difference(20.0)

    # spawn actors
    v_ids = spawn_vehicles(world, tm, args.vehicles, args.tm_port)
    w_ids, c_ids = spawn_walkers(world, client, args.walkers)

    # optional: focus spectator once on first walker (helps you see motion)
    did_focus = False

    print("[run] ASYNC mode — sim is running. Ctrl+C to stop early.")
    t_end = time.time() + args.seconds
    last_retarget = 0.0

    try:
        while time.time() < t_end:
            # pump one snapshot so client-side sees fresh states
            try:
                world.wait_for_tick()
            except Exception:
                pass

            # set spectator once (optional)
            if not did_focus and w_ids:
                w0 = world.get_actor(w_ids[0])
                if w0:
                    tr = w0.get_transform()
                    spec = world.get_spectator()
                    spec.set_transform(carla.Transform(tr.location + carla.Location(z=15),
                                                       carla.Rotation(pitch=-90)))
                    did_focus = True

            # quick vehicle speed sample every loop
            speeds = []
            for vid in v_ids[:10]:
                a = world.get_actor(vid)
                if not a:
                    continue
                v = a.get_velocity()
                speeds.append((v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5 * 3.6)
            if speeds:
                print(f"[stat] avg_speed≈{sum(speeds)/len(speeds):.1f} km/h  (veh={len(v_ids)} walk={len(w_ids)})")

            # retarget walkers every ~10s so stragglers get unstuck
            now = time.time()
            if now - last_retarget > 10.0:
                last_retarget = now
                for cid in c_ids:
                    c = world.get_actor(cid)
                    if not c:
                        continue
                    try:
                        dest = world.get_random_location_from_navigation()
                        if dest:
                            c.go_to_location(dest)
                    except Exception:
                        pass

            time.sleep(max(0.0, float(args.sleep)))
    except KeyboardInterrupt:
        print("\n[run] interrupted by user.")
    finally:
        print("[teardown] destroying spawned NPCs...")
        # stop walker controllers first
        for cid in c_ids:
            c = world.get_actor(cid)
            if c and c.type_id.startswith("controller.ai.walker"):
                try:
                    c.stop()
                except Exception:
                    pass
        # destroy walkers then vehicles
        for aid in w_ids + c_ids + v_ids:
            a = world.get_actor(aid)
            if a:
                try:
                    a.destroy()
                except Exception:
                    pass
        print("[teardown] done.")


if __name__ == "__main__":
    main()
