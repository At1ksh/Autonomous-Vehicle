import argparse
import random 
from typing import List 

import carla 

from utils import load_config, get_client, get_world_and_tm, set_seed 

def _get_safe_spawns(world: carla.World) -> List[carla.Transform]:
    m = world.get_map()
    safe = []
    for sp in m.get_spawn_points():
        wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        nxt = wp.next(12.0)
        if nxt and nxt[0].is_junction:
            continue

        # align to lane, then nudge forward & lift slightly
        tr = carla.Transform(wp.transform.location, wp.transform.rotation)
        fwd = tr.get_forward_vector()
        tr.location += carla.Location(x=fwd.x * 1.5, y=fwd.y * 1.5, z=0.3)
        safe.append(tr)
    return safe

def _per_vehicle_tm_settings(tm: carla.TrafficManager, veh: carla.Actor, tr_cfg: dict):
    # obey your config.json -> "traffic"
    tm.auto_lane_change(veh, bool(tr_cfg.get("auto_lane_change", True)))
    tm.random_left_lanechange_percentage(veh, float(tr_cfg.get("lane_change_random_left", 20.0)))
    tm.random_right_lanechange_percentage(veh, float(tr_cfg.get("lane_change_random_right", 20.0)))

    tm.ignore_signs_percentage(veh, float(tr_cfg.get("ignore_signs_percentage", 0.0)))
    tm.ignore_lights_percentage(veh, float(tr_cfg.get("ignore_lights_percentage", 0.0)))

    # per-vehicle speed offset (on top of TM global)
    tm.vehicle_percentage_speed_difference(veh, float(random.uniform(10.0, 25.0)))

    # extra cushion behind leader (also have a global headway via utils.set_tm_global)
    tm.distance_to_leading_vehicle(veh, float(max(5.0, tr_cfg.get("min_distance", 8.0))))

    
def _spawn_vehicles(world: carla.World, tm:carla.TrafficManager, n:int, lights_on: bool, tm_port: int, tr_cfg:dict) -> List[int]:
    lib = world.get_blueprint_library()
    v_bps = [bp for bp in lib.filter("vehicle.*") if int(bp.get_attribute("number_of_wheels").as_int())>=4 and all(bad not in bp.id for bad in ["ambulance","firetruck","bus","carlacola","t2"])]
    #v_bps = [bp for bp in lib.filter("vehicle.*")] #spawn 2 wheelers also using this 
    
    
    spawns = _get_safe_spawns(world)
    random.shuffle(spawns)
    count = min(n, len(spawns))
    ids = []
    
    for i in range(count):
        bp = random.choice(v_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color" , random.choice(bp.get_attribute("color").recommended_values))
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name","autopilot")
        
        actor = world.try_spawn_actor(bp, spawns[i])
        if not actor:
            continue 
        
        actor.set_autopilot(True, tm_port)
        tr = actor.get_transform()
        print(f"[diag] auto ON tm:{tm_port} id={actor.id} yaw={tr.rotation.yaw:.1f} @ ({tr.location.x:.1f},{tr.location.y:.1f})")

        
        if lights_on:
            try:
                actor.set_light_state(
                    carla.VehicleLightState(
                        carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
                    )
                )
            except Exception:
                pass
            
        _per_vehicle_tm_settings(tm,actor,tr_cfg)
        
        ids.append(actor.id)
        print(f"[traffic] spawned {actor.type_id} {actor.id} at {spawns[i].location}")
        


    print(f"[traffic] spawned {len(ids)}/{count} vehicles")
    return ids 

def _spawn_pedestrians(world: carla.World,client:carla.Client, n:int)-> List[int]:
    lib = world.get_blueprint_library()
    walker_bps = lib.filter("walker.pedestrian.*")
    controller_bp = lib.find("controller.ai.walker")
    
    spawn_points = []
    for _ in range(n * 3):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))
        if len(spawn_points) >= n:
            break
    spawn_points = spawn_points[:n]
    
    if not spawn_points:
        print("[traffic] no navmesh spawn points for walkers")
        return []
    
    w_batch = []
    for sp in spawn_points:
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible","false")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name","walker")
        w_batch.append(carla.command.SpawnActor(bp,sp))
        
    w_results = client.apply_batch_sync(w_batch, True)
    walker_ids = [r.actor_id for r in w_results if not r.error and r.actor_id != 0]
    if not walker_ids:
        return []
    
    world.wait_for_tick()
    
    c_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
    c_results = client.apply_batch_sync(c_batch, True)
    controller_ids = [r.actor_id for r in c_results if not r.error and r.actor_id != 0]
    
    
    if not controller_ids:
        client.apply_batch([carla.command.DestroyActor(w) for w in walker_ids]) 
        print("[traffic] controller spawn failed; cleaned walkers")
        return []
    
    world.wait_for_tick()
    
    for cid in controller_ids:
        c= world.get_actor(cid)
        if c:
            try:
                c.start()
                c.set_max_speed(1.3+random.random()*0.5)
            except Exception:
                pass
            
    world.wait_for_tick()
    
    for cid in controller_ids:
        c= world.get_actor(cid)
        if c:
            try:
                dest = world.get_random_location_from_navigation()
                if dest:
                    c.go_to_location(dest)
            except Exception:
                pass
    
    world.wait_for_tick()
    
    print(f"[traffic] spawned {len(walker_ids)} walkers and {len(controller_ids)} controllers")
    return walker_ids + controller_ids

  
    # walkers, controllers= [],[]
    
    # for sp in spawn_points:
    #     bp = random.choice(walker_bps)
    #     if bp.has_attribute("is_invincible"):
    #         bp.set_attribute("is_invincible","false")
    #     if bp.has_attribute("role_name"):
    #         bp.set_attribute("role_name","walker")
    #     w=world.try_spawn_actor(bp,sp)
    #     if w:
    #         walkers.append(w)
            
    # for w in walkers:
    #     try:
    #         c = world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
    #         controllers.append(c)
    #     except Exception:
    #         pass
    
    # for c in controllers:
    #     try:
    #         c.start()
    #         c.go_to_location(world.get_random_location_from_navigation())
    #         c.set_max_speed(1.4 + random.random())
    #     except Exception:
    #         pass
    
    # world.wait_for_tick()
    # world.wait_for_tick()
    # for c in controllers:
    #     try:
    #         c.go_to_location(world.get_random_location_from_navigation())
    #     except Exception:
    #         pass
        
    


    # print(f"[traffic] spawned {len(walkers)} walkers and {len(controllers)} controllers")
    # return [a.id for a in walkers] + [a.id for a in controllers]

def main(args):
    cfg = load_config(args.config)
    set_seed(cfg)
    client = get_client(cfg)
    world, tm = get_world_and_tm(client, cfg)
    s = world.get_settings()
    print(f"[diag] sync={s.synchronous_mode} fds={s.fixed_delta_seconds} tm_port={tm.get_port() if hasattr(tm,'get_port') else 'unknown'}")
    try:
        print(f"[diag] tm_sync={tm.get_synchronous_mode()}")
    except Exception:
        print("[diag] tm_sync=getter not available")

    
    tr = cfg.get("traffic", {})
    vehicles = int(args.vehicles) if args.vehicles is not None else int(tr.get("vehicles", 80))
    walkers = int(args.walkers) if args.walkers is not None else int(tr.get("walkers", 40))
    lights_on = bool(tr.get("vehicle_lights_on", True))
    tm_port = int(cfg.get("tm_port", 5000))
    
    v_ids = _spawn_vehicles(world,tm,vehicles,lights_on,tm_port,tr)
    w_ids = _spawn_pedestrians(world,client,walkers)
    world.wait_for_tick()
    world.wait_for_tick()
    veh_actors = [world.get_actor(i) for i in v_ids][:10]
    for a in veh_actors:
        v = a.get_velocity()
        speed = (v.x**2+v.y**2+v.z**2) ** 0.5 * 3.6  # km/h
        print(f"[diag] veh {a.id} speed≈{speed:.1f} km/h")
    
    walker_actors = [a for a in world.get_actors() if a.id in w_ids and a.type_id.startswith("walker.pedestrian")]
    print(f"[diag] walkers_spawned={len(walker_actors)}")
    
    print(f"[traffic] Done. vehicles = {len(v_ids)} walkers+ctrls={len(w_ids)}")
    
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--vehicles", type=int, default=None, help="Number of vehicles to spawn (overrides config)")
    ap.add_argument("--walkers", type=int, default=None, help="Number of walkers to spawn (overrides config)")
    main(ap.parse_args())