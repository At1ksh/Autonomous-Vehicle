# spawn_traffic.py
# Spawns vehicles + walkers AND runs the sync tick loop in this same process.
# Designed for CARLA 0.9.14. Keeps your existing config structure intact.

import argparse
import random
import time
from typing import Dict, List, Tuple

import carla

from utils import (
    load_config, set_seed, get_client, get_world_and_tm
)

# ---------------- helpers ----------------

def _get_safe_spawns(world: carla.World) -> List[carla.Transform]:
    """Lane-centered, non-junction spawns aligned with lane; nudge forward & lift slightly."""
    m = world.get_map()
    safe: List[carla.Transform] = []
    for sp in m.get_spawn_points():
        wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        nxt = wp.next(12.0)
        if nxt and nxt[0].is_junction:
            continue
        tr = carla.Transform(wp.transform.location, wp.transform.rotation)
        fwd = tr.get_forward_vector()
        tr.location += carla.Location(x=fwd.x * 1.5, y=fwd.y * 1.5, z=0.3)
        safe.append(tr)
    return safe

def _per_vehicle_tm_settings(tm: carla.TrafficManager, veh: carla.Actor, tr_cfg: dict):
    """Apply per-vehicle TM behavior pulled from your config's 'traffic' block."""
    try:
        tm.auto_lane_change(veh, bool(tr_cfg.get("auto_lane_change", True)))
        tm.random_left_lanechange_percentage(veh, float(tr_cfg.get("lane_change_random_left", 20.0)))
        tm.random_right_lanechange_percentage(veh, float(tr_cfg.get("lane_change_random_right", 20.0)))

        tm.ignore_signs_percentage(veh, float(tr_cfg.get("ignore_signs_percentage", 0.0)))
        tm.ignore_lights_percentage(veh, float(tr_cfg.get("ignore_lights_percentage", 0.0)))

        # Add some per-vehicle speed offset on top of TM global
        per_min = float(tr_cfg.get("per_vehicle_speed_diff_min", 5.0))
        per_max = float(tr_cfg.get("per_vehicle_speed_diff_max", 20.0))
        tm.vehicle_percentage_speed_difference(veh, float(random.uniform(per_min, per_max)))

        tm.distance_to_leading_vehicle(veh, float(max(5.0, tr_cfg.get("min_distance", 8.0))))
    except Exception:
        pass

# ---------------- spawners ----------------

def _spawn_vehicles(world: carla.World, tm: carla.TrafficManager, vehicles: int, tm_port: int, tr_cfg: dict) -> List[int]:
    lib = world.get_blueprint_library()
    v_bps = [bp for bp in lib.filter("vehicle.*")
             if int(bp.get_attribute("number_of_wheels").as_int()) >= 4
             and all(x not in bp.id for x in ["bus", "firetruck", "ambulance", "carlacola", "t2"])]

    spawns = _get_safe_spawns(world)
    random.shuffle(spawns)
    count = min(vehicles, len(spawns))

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
        _per_vehicle_tm_settings(tm, actor, tr_cfg)

        tr = actor.get_transform()
        print(f"[veh] {actor.type_id} id={actor.id} @ ({tr.location.x:.1f},{tr.location.y:.1f}) yaw={tr.rotation.yaw:.1f}")
        ids.append(actor.id)

    print(f"[veh] spawned {len(ids)}/{count}")
    return ids

def _spawn_pedestrians(world: carla.World, client: carla.Client, walkers: int, ped_cfg: dict
                       ) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, carla.Location]]:
    """Batch-spawn walkers + controllers with staged ticks. Returns walker_ids, controller_ids, ctrl->walker, current_goals."""
    lib = world.get_blueprint_library()
    walker_bps = lib.filter("walker.pedestrian.*")
    controller_bp = lib.find("controller.ai.walker")

    # oversample nav points a bit
    spawn_points: List[carla.Transform] = []
    for _ in range(walkers * 3):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))
        if len(spawn_points) >= walkers:
            break
    spawn_points = spawn_points[:walkers]

    if not spawn_points:
        print("[walk] no navmesh spawn points found")
        return [], [], {}, {}

    # 1) spawn walkers (batch) + force a frame
    w_batch = []
    for sp in spawn_points:
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "walker")
        w_batch.append(carla.command.SpawnActor(bp, sp))
    w_results = client.apply_batch_sync(w_batch, True)
    walker_ids = [r.actor_id for r in w_results if not r.error and r.actor_id != 0]
    if not walker_ids:
        return [], [], {}, {}
    world.tick()

    # 2) spawn controllers (batch) + force a frame
    c_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
    c_results = client.apply_batch_sync(c_batch, True)
    controller_ids = [r.actor_id for r in c_results if not r.error and r.actor_id != 0]
    if not controller_ids:
        client.apply_batch([carla.command.DestroyActor(w) for w in walker_ids])
        print("[walk] controller spawn failed; cleaned walkers.")
        return [], [], {}, {}
    world.tick()

    # map controller -> walker (by spawn order)
    ctrl2walker: Dict[int, int] = {cid: wid for cid, wid in zip(controller_ids, walker_ids)}

    # 3) start controllers & initial goal (staged)
    speed = float(ped_cfg.get("speed", 1.3))
    current_goals: Dict[int, carla.Location] = {}

    for cid in controller_ids:
        c = world.get_actor(cid)
        if c:
            try:
                c.start()
                c.set_max_speed(speed)
            except Exception:
                pass
    world.tick()

    for cid in controller_ids:
        c = world.get_actor(cid)
        if c:
            try:
                dest = world.get_random_location_from_navigation()
                if dest:
                    c.go_to_location(dest)
                    current_goals[cid] = dest
            except Exception:
                pass
    world.tick()

    print(f"[walk] spawned {len(walker_ids)} walkers + {len(controller_ids)} controllers")
    return walker_ids, controller_ids, ctrl2walker, current_goals

# ---------------- ticker+run loop (merged here) ----------------

def _run_loop(world: carla.World,
              veh_ids: List[int],
              walk_ids: List[int],
              ctrl_ids: List[int],
              ctrl2walker: Dict[int, int],
              current_goals: Dict[int, carla.Location],
              cfg: dict,
              run_seconds: float = None):
    """Authoritative sync tick loop that replaces ticker.py."""

    sync = world.get_settings().synchronous_mode
    if not sync:
        print("[run] WARNING: world is async; this loop expects sync mode.")
    else:
        print("[run] SYNC mode: this process is advancing frames.")

    # optional runtime
    t_end = (time.time() + run_seconds) if run_seconds else None

    # walker re-target settings (only when they reach their current goal)
    ped_cfg = cfg.get("pedestrians", {})
    retarget_radius = float(ped_cfg.get("retarget_radius", 2.0))

    try:
        while (t_end is None) or (time.time() < t_end):
            # advance exactly one frame
            if sync:
                world.tick()
            else:
                world.wait_for_tick()

            # (optional) quick vehicle stat
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
                    print(f"[stat] avg_v≈{sum(speeds)/len(speeds):.1f} km/h (veh={len(veh_ids)} walk={len(walk_ids)})")

            # walkers: retarget only when they reach current goal
            for cid in ctrl_ids:
                ctrl = world.get_actor(cid)
                if not ctrl:
                    continue
                wid = ctrl2walker.get(cid)
                if not wid:
                    continue
                walker = world.get_actor(wid)
                if not walker:
                    continue

                goal = current_goals.get(cid)
                if goal is None:
                    # assign once if missing
                    dest = world.get_random_location_from_navigation()
                    if dest:
                        try:
                            ctrl.go_to_location(dest)
                            current_goals[cid] = dest
                        except Exception:
                            pass
                    continue

                try:
                    loc = walker.get_location()
                    if loc.distance(goal) <= retarget_radius:
                        dest = world.get_random_location_from_navigation()
                        if dest:
                            ctrl.go_to_location(dest)
                            current_goals[cid] = dest
                except Exception:
                    continue

    except KeyboardInterrupt:
        print("\n[run] interrupted by user.")

# ---------------- main ----------------

def main(args):
    cfg = load_config(args.config)
    set_seed(cfg)

    client = get_client(cfg)
    world, tm = get_world_and_tm(client, cfg)

    # Make sure TM mirrors world sync (bootstrap sets this already; this is belt & suspenders)
    try:
        tm.set_synchronous_mode(world.get_settings().synchronous_mode)
    except Exception:
        pass

    # Warm-up a frame so TM<->world sync latches BEFORE spawns
    if world.get_settings().synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick()

    # Traffic config and counts
    tr = cfg.get("traffic", {})
    ped_cfg = cfg.get("pedestrians", {})
    tm_port = int(cfg.get("tm_port", 5000))
    vehicles = int(cfg.get("spawn", {}).get("vehicles", 30))
    walkers  = int(cfg.get("spawn", {}).get("walkers", 30))
    run_seconds = args.seconds

    # --- spawn ---
    veh_ids = _spawn_vehicles(world, tm, vehicles, tm_port, tr)
    walk_ids, ctrl_ids, ctrl2walker, current_goals = _spawn_pedestrians(world, client, walkers, ped_cfg)

    # Optional spectator focus
    spec = world.get_spectator()
    focus_id = (veh_ids[0] if veh_ids else (walk_ids[0] if walk_ids else None))
    if focus_id is not None:
        a = world.get_actor(focus_id)
        if a:
            trf = a.get_transform()
            spec.set_transform(carla.Transform(trf.location + carla.Location(z=20), carla.Rotation(pitch=-90)))

    print(f"[spawn_traffic] vehicles={len(veh_ids)} walkers={len(walk_ids)} | driving now...")
    _run_loop(world, veh_ids, walk_ids, ctrl_ids, ctrl2walker, current_goals, cfg, run_seconds=run_seconds)

    # --- teardown ---
    print("[teardown] stopping walkers + destroying all actors...")
    # stop walker controllers first
    for cid in ctrl_ids:
        c = world.get_actor(cid)
        if c and c.type_id.startswith("controller.ai.walker"):
            try:
                c.stop()
            except Exception:
                pass

    # destroy walkers, controllers, vehicles
    for aid in (walk_ids + ctrl_ids + veh_ids):
        a = world.get_actor(aid)
        if a:
            try:
                a.destroy()
            except Exception:
                pass
    print("[teardown] done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--seconds", type=float, default=None, help="run for N seconds; omit for infinite until Ctrl+C")
    main(ap.parse_args())
