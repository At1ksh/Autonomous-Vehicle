import argparse
import random
import time 
import math
from typing import List, Tuple, Dict
import carla

def safe_lane_center_spawns(world:carla.World) -> List[carla.Transform]:
    m = world.get_map()
    out: List[carla.Transform] = []
    for sp in m.get_spawn_points():
        wp = m.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not wp or wp.is_junction:
            continue
        tr=carla.Transform(wp.transform.location, wp.transform.rotation)
        fwd = tr.get_forward_vector()
        tr.location += carla.Location(x=fwd.x * 1.5, y=fwd.y * 1.5,z=0.3)
        out.append(tr)
        
    if not out:
        out = m.get_spawn_points()
        for tr in out:
            wp = m.get_waypoint(tr.location, project_to_road=True, lane_type = carla.LaneType.Driving)
            if wp:
                tr.rotation.yaw = wp.transform.rotation.yaw
                
    return out 

def euclid(a: carla.Location, b: carla.Location) -> float:
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def get_actor_with_retry(world:carla.World, actor_id:int, max_tries: int = 60):
    for _ in range(max_tries):
        actor = world.get_actor(actor_id)
        if actor is not None:
            return actor
        world.tick()
        
    return None

def spawn_vehicles_sync(world:carla.World, tm:carla.TrafficManager, n:int, tm_port:int, seed:int) -> List[int]:
    lib = world.get_blueprint_library()
    v_bps = [bp for bp in lib.filter("vehicle.*") if int(bp.get_attribute("number_of_wheels").as_int())>=4 and all(x not in bp.id for x in ["bus","firetruck","ambulance","carlacola","t2"])]
    
    spawns = safe_lane_center_spawns(world)
    random.shuffle(spawns)
    count = min(n, len(spawns))
    
    ids: List[int] = []
    for i in range(count):
        bp = random.choice(v_bps)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name","autopilot")
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            
        actor = world.try_spawn_actor(bp, spawns[i])
        if not actor:
            continue
        
        actor.set_autopilot(True, tm_port)
        
        try:
            tm.auto_lane_change(actor, True)
            tm.random_left_lanechange_percentage(actor, 30.0)
            tm.random_right_lanechange_percentage(actor, 30.0)
            tm.ignore_signs_percentage(actor, 0.0)
            tm.ignore_lights_percentage(actor, 0.0)
            tm.vehicle_percentage_speed_difference(actor, float(random.uniform(10.0,25.0)))
            tm.distance_to_leading_vehicle(actor, 8.0)
        except Exception:
            pass
        
        ids.append(actor.id)
        tr=actor.get_transform()
        print(f"[veh] {actor.type_id} id={actor.id} @ ({tr.location.x:.1f},{tr.location.y:.1f}) yaw={tr.rotation.yaw:.1f}")
    
    print(f"[veh] spawned {len(ids)}/{count}")
    return ids

def spawn_walkers_sync(world:carla.World, client:carla.Client, n:int, speed:float, retarget_radius:float) -> Tuple[List[int], List[int], Dict[int,int],Dict[int,carla.Location]]:
    bp = world.get_blueprint_library()
    walker_bps = bp.filter('walker.pedestrian.*')
    ctrl_bp = bp.find('controller.ai.walker')
    
    spawns: List[carla.Transform] = []
    for _ in range(n*3):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawns.append(carla.Transform(loc))
        if len(spawns) >= n:
            break
    
    spawns = spawns[:n]
    if not spawns:
        print("[walk] no valid navmesh spawns found.")
        return [], [], {}
    
    spawn_cmds = []
    for tr in spawns:
        wbp = random.choice(walker_bps)
        if wbp.has_attribute('is_invincible'):
            wbp.set_attribute('is_invincible','false')
        spawn_cmds.append(carla.command.SpawnActor(wbp, tr))
    results = client.apply_batch_sync(spawn_cmds, True)
    walker_ids = [r.actor_id for r in results if not r.error and r.actor_id != 0]
    if not walker_ids:
        return [], [], {}
    
    world.tick()
    
    ctrl_cmds = [carla.command.SpawnActor(ctrl_bp,carla.Transform(), wid) for wid in walker_ids]
    results = client.apply_batch_sync(ctrl_cmds, True)
    ctrl_ids = [r.actor_id for r in results if not r.error and r.actor_id != 0]
    if not ctrl_ids:
        print("[walk] failed to spawn controllers.")
        client.apply_batch([carla.command.DestroyActor(wid) for wid in walker_ids])
        return [], [], {}
    
    world.tick()
    
    ctrl_to_walker = {}
    for i,cid in enumerate(ctrl_ids):
        if i < len(walker_ids):
            ctrl_to_walker[cid] = walker_ids[i]
            
    controllers=[]
    for cid in ctrl_ids:
        c = get_actor_with_retry(world,cid, max_tries=30)
        if c:
            controllers.append(c)
    
    current_goals: Dict[int, carla.Location] = {}
            
    for ctrl in controllers:
        try:
            ctrl.start()
            ctrl.set_max_speed(speed)
        except Exception:
            pass
        
    world.tick()
    
    for ctrl in controllers:
        try:
            dest = world.get_random_location_from_navigation()
            if dest:
                ctrl.go_to_location(dest)
                current_goals[ctrl.id] = dest
        except Exception:
            pass
        
    world.tick()
    print(f"[walk] spawned {len(walker_ids)} walkers + {len(ctrl_ids)} controllers")
    return walker_ids, ctrl_ids, ctrl_to_walker, current_goals
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=5000)
    ap.add_argument("--vehicles", type=int, default=30)
    ap.add_argument("--walkers", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dt", type=float, default=0.05, help="fixed delta seconds (e.g., 0.05 = 20 FPS)")
    ap.add_argument("--walker_speed", type=float, default=1.3)
    ap.add_argument("--retarget_radius", type=float, default=2.0)
    ap.add_argument("--retarget_interval", type=float, default=0.5)
    ap.add_argument("--cross", type=float, default=0.8, help="Pedestrian road-crossing factor [0..1]")
    args = ap.parse_args()
    
    random.seed(args.seed)
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"[world] current map: {world.get_map().name}")

    # --- enable SYNC world ---
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(args.dt)
    world.apply_settings(settings)
    
    try:
        world.set_pedestrians_cross_factor(float(args.cross))
        world.set_pedestrians_seed(int(args.seed))
    except Exception:
        pass
    
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    tm.set_hybrid_physics_mode(True)
    tm.set_respawn_dormant_vehicles(True)
    tm.set_global_distance_to_leading_vehicle(10.0)
    tm.global_percentage_speed_difference(20.0)
    
    world.tick()
    veh_ids = spawn_vehicles_sync(world, tm, args.vehicles, args.tm_port, args.seed)
    walk_ids, ctrl_ids, ctrl_to_walker,current_goals = spawn_walkers_sync(world, client, args.walkers, speed= args.walker_speed, retarget_radius=args.retarget_radius)
    
    spec = world.get_spectator()
    if veh_ids:
        a = world.get_actor(veh_ids[0])
        if a:
            tr = a.get_transform()
            spec.set_transform(carla.Transform(tr.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
    elif walk_ids:
        a = world.get_actor(walk_ids[0])
        if a:
            tr = a.get_transform()
            spec.set_transform(carla.Transform(tr.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
            
    t_end = time.time() + args.seconds
    last_retarget = time.time()
    
    try:
        while time.time() < t_end:
            world.tick()
            
            if veh_ids:
                sample = veh_ids[:10]
                speeds = []
                for vid in sample:
                    a= world.get_actor(vid)
                    if not a:
                        continue
                    
                    v= a.get_velocity()
                    kmh = (v.x*v.x + v.y*v.y + v.z*v.z) ** 0.5 * 3.6
                    speeds.append(kmh)
                if speeds:
                    print(f"[stat] avg_speed≈{sum(speeds)/len(speeds):.1f} km/h (veh={len(veh_ids)})")
            for cid in ctrl_ids:
                ctrl = world.get_actor(cid)
                if not ctrl:
                    continue
                wid = ctrl_to_walker.get(cid)
                if not wid:
                    continue
                walker = world.get_actor(wid)
                if not walker:
                    continue
                
                try:
                    goal = current_goals.get(cid)
                    if goal is None:
                        dest = world.get_random_location_from_navigation()
                        if dest:
                            ctrl.go_to_location(dest)
                            current_goals[cid] = dest
                        continue
                    
                    loc = walker.get_location()
                    if euclid(loc, goal) <= float(args.retarget_radius):
                        dest = world.get_random_location_from_navigation()
                        if dest:
                            ctrl.go_to_location(dest)
                            current_goals[cid] = dest
                except Exception:
                    continue
            now = time.time()
            # do_periodic = (now - last_retarget) >= float(args.retarget_interval)
            
            # if ctrl_ids:
            #     for cid in ctrl_ids:
            #         ctrl = world.get_actor(cid)
            #         if not ctrl:
            #             continue
            #         wid = ctrl_to_walker.get(cid,None)
            #         if not wid:
            #             continue 
            #         walker = world.get_actor(wid)
            #         if not walker:
            #             continue 
                    
            #         try:
            #             if do_periodic:
            #                 dest = world.get_random_location_from_navigation()
            #                 if dest:
            #                     ctrl.go_to_location(dest)
            #             else:
            #                 pass
            #         except Exception:
            #             continue
                
            # if do_periodic:
            #     last_retarget = now
    except KeyboardInterrupt:
        print("\n[run] interrupted by user.")
    finally:
        print("[teardown] destroying actors...")
        
        for cid in ctrl_ids:
            c= world.get_actor(cid)
            if c and c.type_id.startswith("controller.ai.walker"):
                try:
                    c.stop()
                except Exception:
                    pass
                
        for aid in (walk_ids + ctrl_ids + veh_ids):
            a = world.get_actor(aid)
            if a:
                try:
                    a.destroy()
                except Exception:
                    pass
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        world.apply_settings(original)
        print("[teardown] done.")

if __name__ == "__main__":
    main()        
            
        