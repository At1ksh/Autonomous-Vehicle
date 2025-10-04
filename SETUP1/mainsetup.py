import argparse
import random
import json
import time
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import carla
import numpy as np


def load_config(path: Optional[str])-> dict:
    if path and os.path.exists(path):
        with open(path,'r') as f:
            return json.load(f)
    return {}

def make_run_dir(base="run"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = f"{base}_{ts}"
    os.makedirs(root, exist_ok=True)
    return root

def to_transform(dct: dict) -> carla.Transform:
    loc = carla.Location(x=float(dct.get("x",0.0)), y=float(dct.get("y",0.0)),z=float(dct.get("z",0.0)))
    rot = carla.Rotation(pitch=float(dct.get("pitch",0.0)), yaw=float(dct.get("yaw",0.0)), roll=float(dct.get("roll",0.0)))
    return carla.Transform(location=loc, rotation=rot)

def safe_lane_center_spawns(world: carla.World) -> List[carla.Transform]:
    m = world.get_map()
    out: List[carla.Transform] = []
    for sp in m.get_spawn_points():
        wp= m.get_waypoint(sp.location, project_to_road = True, lane_type = carla.LaneType.Driving)
        if not wp or wp.is_junction:
            continue
        nxt = wp.next(12.0)
        if nxt and nxt[0].is_junction:
            continue
        tr = carla.Transform(wp.transform.location, wp.transform.rotation)
        fwd = tr.get_forward_vector()
        tr.location += carla.Location(x=fwd.x * 1.5, y=fwd.y * 1.5, z=0.3)
        out.append(tr)
    if not out:
        out = m.get_spawn_points()
    return out

def euclid(a: carla.Location, b:carla.Location) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return (dx*dx + dy*dy + dz*dz)**0.5

def configure_world_and_tm(client: carla.Client, cfg: dict)-> Tuple[carla.World, carla.TrafficManager, dict]:
    world = client.get_world()
    settings = world.get_settings()
    sync = bool(cfg.get("synchronous_mode", True))
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = float(cfg.get("fixed_delta_seconds", 0.05)) if sync else None
    world.apply_settings(settings)
    
    tm= client.get_trafficmanager(int(cfg.get("tm_port",5000)))
    tm.set_synchronous_mode(sync)
    tm.set_random_device_seed(int(cfg.get("seed",42)))
    tm.set_hybrid_physics_mode(True)
    tm.set_respawn_dormant_vehicles(True)
    
    tr = cfg.get("traffic",{})
    tm.set_global_distance_to_leading_vehicle(float(tr.get("min_distance",8.0)))
    tm.global_percentage_speed_difference(float(tr.get("global_speed_perc_diff",20)))
    
    peds = cfg.get("pedestrians",{})
    try:
        world.set_pedestrians_cross_factor(float(peds.get("cross_factor",0.5)))
        world.set_pedestrians_seed(int(cfg.get("seed",42)))
    except Exception:
        pass
    
    if sync:
        world.tick()
    else:
        world.wait_for_tick()
    return world, tm, tr


def per_vehicle_tm_settings(tm: carla.TrafficManager, veh:carla.Actor, tr_cfg: dict):
    tm.auto_lane_change(veh, bool(tr_cfg.get("auto_lane_change", True)))
    tm.random_left_lanechange_percentage(veh, float(tr_cfg.get("lane_change_random_left", 20.0)))
    tm.random_right_lanechange_percentage(veh, float(tr_cfg.get("lane_change_random_right", 20.0)))
    tm.ignore_signs_percentage(veh, float(tr_cfg.get("ignore_signs_percentage", 0.0)))
    tm.ignore_lights_percentage(veh, float(tr_cfg.get("ignore_lights_percentage", 0.0)))
    per_min = float(tr_cfg.get("per_vehicle_speed_perc_diff_min",5.0))
    per_max = float(tr_cfg.get("per_vehicle_speed_perc_diff_max",20.0))
    tm.vehicle_percentage_speed_difference(veh, float(random.uniform(per_min, per_max)))
    tm.distance_to_leading_vehicle(veh, float(max(5.0, tr_cfg.get("min_distance",8.0))))
    
def spawn_vehicles(world: carla.World, tm: carla.TrafficManager, n:int, tm_port: int, tr_cfg:dict, reserved_first: int = 1)-> List[int]:
    lib = world.get_blueprint_library()
    v_bps = [bp for bp in lib.filter("vehicle.*") if int(bp.get_attribute("number_of_wheels").as_int())>=4 and all(bad not in bp.id for bad in ["ambulance","firetruck","bus","carlacola","t2"])]
    
    spawns = safe_lane_center_spawns(world)
    random.shuffle(spawns)
    count = min(n, len(spawns))
    ids: List[int] = []
    
    if reserved_first > 0 and len(spawns)>0:
        spawns = spawns[reserved_first:] 
        
    
    
    for i in range(count):
        bp = random.choice(v_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color" , random.choice(bp.get_attribute("color").recommended_values))
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        
        actor = world.try_spawn_actor(bp, spawns[i])
        if not actor:
            continue 
        actor.set_autopilot(True, tm_port)
        per_vehicle_tm_settings(tm, actor, tr_cfg)
        ids.append(actor.id)
    print(f"[veh] spawned {len(ids)}/{count}")
    return ids


def spawn_walkers(world: carla.World, client: carla.Client, n: int, speed: float
                  ) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, carla.Location]]:
    bp = world.get_blueprint_library()
    walker_bps = bp.filter('walker.pedestrian.*')
    ctrl_bp = bp.find('controller.ai.walker')

    # pick navmesh points
    spawns: List[carla.Transform] = []
    for _ in range(n * 3):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawns.append(carla.Transform(loc))
        if len(spawns) >= n:
            break
    spawns = spawns[:n]
    if not spawns:
        print("[walk] no navmesh spawns")
        return [], [], {}, {}

    w_batch = []
    for tr in spawns:
        wbp = random.choice(walker_bps)
        if wbp.has_attribute('is_invincible'):
            wbp.set_attribute('is_invincible', 'false')
        w_batch.append(carla.command.SpawnActor(wbp, tr))
    w_res = client.apply_batch_sync(w_batch, True)
    walker_ids = [r.actor_id for r in w_res if not r.error and r.actor_id != 0]
    if not walker_ids:
        return [], [], {}, {}
    world.tick()

    c_batch = [carla.command.SpawnActor(ctrl_bp, carla.Transform(), wid) for wid in walker_ids]
    c_res = client.apply_batch_sync(c_batch, True)
    ctrl_ids = [r.actor_id for r in c_res if not r.error and r.actor_id != 0]
    if not ctrl_ids:
        client.apply_batch([carla.command.DestroyActor(w) for w in walker_ids])
        print("[walk] controller spawn failed")
        return [], [], {}, {}
    world.tick()

    ctrl2walker = {cid: wid for cid, wid in zip(ctrl_ids, walker_ids)}
    current_goals: Dict[int, carla.Location] = {}

    for cid in ctrl_ids:
        c = world.get_actor(cid)
        if c:
            try:
                c.start()
                c.set_max_speed(speed)
            except Exception:
                pass
    world.tick()

    for cid in ctrl_ids:
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
    print(f"[walk] spawned {len(walker_ids)} walkers + {len(ctrl_ids)} controllers")
    return walker_ids, ctrl_ids, ctrl2walker, current_goals


def spawn_ego(world:carla.World,ego_cfg:dict)-> Optional[carla.Actor]:
    model = ego_cfg.get("blueprint","vehicle.tesla.model3")
    lib = world.get_blueprint_library()
    try:
        bp = lib.find(model)
    except Exception:
        print(f"[ego] no bp {model}")
        return None

    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name","ego")
        
    spawns = safe_lane_center_spawns(world)
    if not spawns:
        print("[ego] no spawns")
        return None
    ego = world.try_spawn_actor(bp, spawns[0])
    if not ego:
        for tr in spawns[1:5]:
            ego = world.try_spawn_actor(bp, tr)
            if ego:
                break
    if not ego:
        print("[ego] spawn failed")
        return None
    
    if bool(ego_cfg.get("autopilot",False)):
        tm_port = int(ego_cfg.get("tm_port",5000))
        ego.set_autopilot(True, tm_port)
    print(f"[ego] spawned {ego.type_id} id={ego.id}")
    return ego

def attach_sensors(world:carla.World, ego: carla.Actor, sensors_cfg: List[dict], out_root: str) -> List[carla.Actor]:
    os.makedirs(out_root, exist_ok=True)
    lib = world.get_blueprint_library()
    sensors: List[carla.Actor] = []
    
    coll_f = open(os.path.join(out_root,"collisions.csv"),"w",buffering = 1)
    coll_f.write("frame,other_type,x,y,z,nx,ny,nz\n")
    lane_f = open(os.path.join(out_root,"lane_invasions.csv"),"w",buffering = 1)
    lane_f.write("frame,lane_type\n")
    imu_f = open(os.path.join(out_root,"imu.csv"),"w",buffering = 1)
    imu_f.write("frame,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
    gnss_f = open(os.path.join(out_root,"gnss.csv"),"w",buffering = 1)
    gnss_f.write("frame,lat,lon,alt\n")
    
    def cam_cb(image: carla.Image, subdir: str, sid: str):
        path = os.path.join(out_root, subdir)
        os.makedirs(path, exist_ok=True)
        image.save_to_disk(os.path.join(path, f"{sid}_{image.frame:08d}.png"))
    
    def lidar_cb(meas:carla.LidarMeasurement, subdir: str, sid: str):
        if np is None:
            return
        pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1,4)[:,:3]
        # Save points to disk or process them
        path = os.path.join(out_root, subdir)
        os.makedirs(path, exist_ok=True)
        fn= os.path.join(path, f"{sid}_{meas.frame:08d}.ply")
        with open(fn,"w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex %d\n" % pts.shape[0])
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for x,y,z in pts:
                f.write(f"{x} {y} {z}\n")
                
    def collision_cb(event: carla.CollisionEvent):
        other = event.other_actor.type_id if event.other_actor else "Unknown"
        imp = event.normal_impulse
        p = event.transform.location
        coll_f.write(f"{event.frame},{other},{p.x:.2f},{p.y:.2f},{p.z:.2f},{imp.x:.2f},{imp.y:.2f},{imp.z:.2f}\n")
        
    def lane_cb(event: carla.LaneInvasionEvent):
        types = "+".join([str(x).split('.')[-1] for x in event.crossed_lane_markings])
        lane_f.write(f"{event.frame},{types}\n")

    def imu_cb(m: carla.IMUMeasurement):
        imu_f.write(f"{m.frame},{m.accelerometer.x:.6f},{m.accelerometer.y:.6f},{m.accelerometer.z:.6f},"
                    f"{m.gyroscope.x:.6f},{m.gyroscope.y:.6f},{m.gyroscope.z:.6f},{m.compass:.6f}\n")

    def gnss_cb(m: carla.GnssMeasurement):
        gnss_f.write(f"{m.frame},{m.latitude:.8f},{m.longitude:.8f},{m.altitude:.3f}\n")

    for s in sensors_cfg or []:
        sid = s.get("id",s.get("type","sensor"))
        stype = s.get("type","")
        tr = to_transform(s.get("transform",{}))
        attrs = s.get("attributes",{})
        
        try:
            bp = lib.find(stype)
        except:
            print(f"[sensor] unknown type {stype}, skipping")
            continue
        
        for k,v in attrs.items():
            if bp.has_attribute(k):
                bp.set_attribute(k,str(v))

        sensor = world.try_spawn_actor(bp, tr, attach_to=ego)
        sensors.append(sensor)
        
        if "sensor.camera" in stype:
            sub = f"{sid}_images"
            sensor.listen(lambda img, sub=sub, sid=sid: cam_cb(img, sub, sid))
        elif "sensor.lidar" in stype:
            sub = f"{sid}_lidar"
            sensor.listen(lambda m, sub=sub, sid=sid: lidar_cb(m, sub, sid))
        elif stype == "sensor.other.collision":
            sensor.listen(lambda ev: collision_cb(ev))
        elif stype == "sensor.other.lane_invasion":
            sensor.listen(lambda ev: lane_cb(ev))
        elif stype == "sensor.other.imu":
            sensor.listen(lambda m: imu_cb(m))
        elif stype == "sensor.other.gnss":
            sensor.listen(lambda m: gnss_cb(m))
        else:
            # generic passthrough; no-op
            sensor.listen(lambda _msg: None)
        print(f"[sensor] {stype} id={sensor.id} attached to ego")
        
    return sensors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    cfg= load_config(args.config)
    random.seed(int(cfg.get("seed",42)))

    client = carla.Client(cfg.get("host","127.0.0.1"), cfg.get("port",2000))
    client.set_timeout(10.0)
    
    world, tm, tr_cfg = configure_world_and_tm(client, cfg)
    
    vehicles_n = int(cfg.get("traffic",{}).get("vehicles",50))
    walkers_n = int(cfg.get("traffic",{}).get("walkers",50))
    ped_cfg = cfg.get("pedestrians",{})
    tm_port = int(cfg.get("tm_port",5000))
    
    ego = spawn_ego(world,cfg.get("ego",{}))
    world.tick()
    
    veh_ids = spawn_vehicles(world,tm,vehicles_n,tm_port,tr_cfg,reserved_first=0 if ego else 1)
    walk_ids, ctrl_ids, ctrl_to_walker, current_goals = spawn_walkers(world,client,walkers_n, float(ped_cfg.get("speed",1.3)))
    
    spec = world.get_spectator()
    focus = ego 
    if focus:
        trf = focus.get_transform()
        spec.set_transform(carla.Transform(trf.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
    
    run_dir = make_run_dir("run")
    sensors =[]
    if ego:
        sensors = attach_sensors(world,ego,cfg.get("sensors",[]), out_root=run_dir)
        
    print(f"[run] driving... (veh)={len(veh_ids)} walk ={len(walk_ids)} ego={'yes' if ego else 'no'})")
    t_end = (time.time()+float(args.seconds)) if args.seconds else None
    retarget_radius = float(ped_cfg.get("retarget_radius",2.0))
    try:
        while(t_end is None) or(time.time()<t_end):
            world.tick()
            
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
                
                goal = current_goals.get(cid)
                if goal is None:
                    dest = world.get_random_location_from_navigation()
                    if dest:
                        try:
                            ctrl.go_to_location(dest)
                            current_goals[cid] = dest
                        except Exception as e:
                            pass
                    continue
                
                try:
                    loc = walker.get_location()
                    if euclid(loc,goal)<retarget_radius:
                        dest = world.get_random_location_from_navigation()
                        if dest:
                            ctrl.go_to_location(dest)
                            current_goals[cid] = dest
                except Exception:
                    continue
                
    except KeyboardInterrupt:
        print("[run]cancelled by user")
    finally:
        print("[teardown] destroying actors...")
        
        # stop sensor streams
        for s in sensors:
            try:
                s.stop()
            except Exception:
                pass

        # stop walkers first
        for cid in ctrl_ids:
            c = world.get_actor(cid)
            if c and c.type_id.startswith("controller.ai.walker"):
                try:
                    c.stop()
                except Exception:
                    pass

        # destroy in order: sensors -> ego -> walkers/controllers -> vehicles
        for s in sensors:
            try:
                s.destroy()
            except Exception:
                pass
        if ego:
            try:
                ego.destroy()
            except Exception:
                pass
        for aid in (walk_ids + ctrl_ids + veh_ids):
            a = world.get_actor(aid)
            if a:
                try:
                    a.destroy()
                except Exception:
                    pass

        # return TM to async if you like (optional)
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass

        print("[teardown] done. Logs at:", run_dir)

if __name__ == "__main__":
    main()
