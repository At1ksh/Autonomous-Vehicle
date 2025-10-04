# sync_npc_diag.py
# One-file diagnostic: load a map, enable synchronous mode, spawn vehicles & walkers,
# and keep ticking in this process (no external ticker). Conservative TM settings.
#
# Usage examples:
#   python sync_npc_diag.py                  # Town03, 25 vehicles, 15 walkers
#   python sync_npc_diag.py --town Town04 --vehicles 40 --walkers 20
#   python sync_npc_diag.py --sleep 0.0      # run at max speed (wall-clock) while staying sync

import argparse
import random
import time
from typing import List, Tuple

import carla


def _safe_lane_center_spawns(world: carla.World, min_ahead: float = 12.0) -> List[carla.Transform]:
    """Return lane-centered spawn transforms, away from junctions, with a small forward nudge."""
    m = world.get_map()
    safe: List[carla.Transform] = []
    for sp in m.get_spawn_points():
        wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        if wp.is_junction:
            continue
        nxt = wp.next(min_ahead)
        if not nxt or nxt[0].is_junction:
            continue
        tr = carla.Transform(wp.transform.location, wp.transform.rotation)
        # nudge 2m forward & 0.3m up to avoid scraping curbs/furniture
        fwd = tr.get_forward_vector()
        tr.location += carla.Location(x=fwd.x * 2.0, y=fwd.y * 2.0, z=0.3)
        safe.append(tr)
    return safe


def _spawn_vehicles(world: carla.World,
                    tm: carla.TrafficManager,
                    n: int,
                    tm_port: int,
                    lights_on: bool = True) -> List[int]:
    lib = world.get_blueprint_library()
    # Start with regular 4+ wheel vehicles, avoid bulky ones while debugging tight towns
    v_bps = [bp for bp in lib.filter("vehicle.*")
             if int(bp.get_attribute("number_of_wheels").as_int()) >= 4
             and all(bad not in bp.id for bad in ["bus", "firetruck", "ambulance", "carlacola", "t2"])]

    spawns = _safe_lane_center_spawns(world)
    random.shuffle(spawns)
    count = min(n, len(spawns))
    ids: List[int] = []

    for i in range(count):
        bp = random.choice(v_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")

        tr = spawns[i]
        actor = world.try_spawn_actor(bp, tr)
        if not actor:
            continue

        # Register with TM (autopilot) and apply conservative per-vehicle behavior
        actor.set_autopilot(True, tm_port)
        try:
            tm.auto_lane_change(actor, False)  # HARD off while debugging
            tm.random_left_lanechange_percentage(actor, 0.0)
            tm.random_right_lanechange_percentage(actor, 0.0)
            tm.ignore_signs_percentage(actor, 0.0)
            tm.ignore_lights_percentage(actor, 0.0)
            tm.vehicle_percentage_speed_difference(actor, float(random.uniform(10.0, 25.0)))  # slower
        except Exception:
            pass

        if lights_on:
            try:
                actor.set_light_state(
                    carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
                )
            except Exception:
                pass

        ids.append(actor.id)
        print(f"[veh] {actor.type_id} id={actor.id} @ ({tr.location.x:.1f},{tr.location.y:.1f}) yaw={tr.rotation.yaw:.1f}")

    print(f"[veh] spawned {len(ids)}/{count}")
    return ids


def _spawn_walkers(world: carla.World, n: int) -> Tuple[List[int], List[int]]:
    lib = world.get_blueprint_library()
    walker_bps = lib.filter("walker.pedestrian.*")
    controller_bp = lib.find("controller.ai.walker")

    spawn_points: List[carla.Transform] = []
    for _ in range(n):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))

    walkers: List[carla.Actor] = []
    controllers: List[carla.Actor] = []

    for sp in spawn_points:
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "walker")
        w = world.try_spawn_actor(bp, sp)
        if w:
            walkers.append(w)

    for w in walkers:
        try:
            c = world.spawn_actor(controller_bp, carla.Transform(), w)  # parent = walker
            controllers.append(c)
        except Exception:
            pass

    for c in controllers:
        try:
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(1.4 + random.random())  # ~1.4–2.4 m/s
        except Exception:
            pass

    # In sync mode, give controllers two ticks to boot, then retarget once
    world.tick()
    world.tick()
    for c in controllers:
        try:
            c.go_to_location(world.get_random_location_from_navigation())
        except Exception:
            pass

    print(f"[walk] spawned {len(walkers)} walkers + {len(controllers)} controllers")
    return [w.id for w in walkers], [c.id for c in controllers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=5000)
    ap.add_argument("--town", default="Town03", help="Different environment from your setup")
    ap.add_argument("--vehicles", type=int, default=25)
    ap.add_argument("--walkers", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed_dt", type=float, default=0.05, help="Simulation step (s)")
    ap.add_argument("--sleep", type=float, default=0.002, help="Wall-clock sleep each tick to ease CPU")
    ap.add_argument("--seconds", type=float, default=120.0, help="Run time for this diagnostic")
    args = ap.parse_args()

    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    # Load a different map/environment (fresh world)
    world = client.load_world(args.town)
    print(f"[world] loaded {world.get_map().name}")

    # Apply synchronous mode + fixed delta
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(args.fixed_dt)
    world.apply_settings(settings)

    # Keep pedestrians to sidewalks/crossings only (must be set before spawning)
    try:
        world.set_pedestrians_cross_factor(0.0)
        world.set_pedestrians_seed(args.seed)
    except Exception:
        pass

    # Weather (optional: comment out if you want default)
    try:
        world.set_weather(carla.WeatherParameters.ClearNoon)
    except Exception:
        pass

    # Traffic Manager (sync + conservative global behavior)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    tm.set_hybrid_physics_mode(True)
    tm.set_respawn_dormant_vehicles(True)
    tm.set_global_distance_to_leading_vehicle(10.0)     # big headway globally
    tm.global_percentage_speed_difference(20.0)         # 20% slower globally

    # Spawn actors
    v_ids = _spawn_vehicles(world, tm, args.vehicles, args.tm_port, lights_on=True)
    w_ids, c_ids = _spawn_walkers(world, args.walkers)

    # Simple live diagnostics loop (this script ticks the world)
    print(f"[run] sync=True, dt={settings.fixed_delta_seconds}s | Ctrl+C to stop earlier")
    t_end = time.time() + args.seconds
    tick_i = 0
    try:
        while time.time() < t_end:
            frame = world.tick()
            tick_i += 1
            if tick_i % 40 == 0:  # print every ~2s at 20 Hz
                # sample a few vehicles for speed
                speeds = []
                for vid in v_ids[:10]:
                    a = world.get_actor(vid)
                    if not a:
                        continue
                    v = a.get_velocity()
                    spd = (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5 * 3.6  # km/h
                    speeds.append(spd)
                avg = sum(speeds)/len(speeds) if speeds else 0.0
                print(f"[tick] frame={frame} avg_speed≈{avg:4.1f} km/h  veh={len(v_ids)} walk={len(w_ids)}")
            if args.sleep > 0:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n[run] interrupted by user.")
    finally:
        # Clean teardown (destroy controllers first)
        actors = world.get_actors()
        for cid in c_ids:
            c = world.get_actor(cid)
            if c and c.type_id.startswith("controller.ai.walker"):
                try:
                    c.stop()
                except Exception:
                    pass
        # Destroy everything we spawned
        for aid in v_ids + w_ids + c_ids:
            a = world.get_actor(aid)
            if a:
                try:
                    a.destroy()
                except Exception:
                    pass
        print("[teardown] destroyed spawned NPCs & controllers.")


if __name__ == "__main__":
    main()
