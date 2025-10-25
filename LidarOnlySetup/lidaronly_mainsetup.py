import argparse
import random
import json
import time
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import threading 
import queue
import traceback

import carla
import numpy as np



class AsyncDiskWriter:
    """background file writer so that sensor callbacks can stay fast"""
    
    def __init__(self,name:str,max_queue:int=256):
        self.name = name
        self.q = queue.Queue(max_queue)
        self._drops = 0
        self._alive=True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        
    @property
    def dropped(self)-> int:
        return self._drops 
    
    def submit(self, fn):
        try:
            self.q.put_nowait(fn)
        except queue.Full:
            self._drops += 1
    
    def _run(self):
        while True:
            job = self.q.get()
            if job is None:
                break
            try:
                job()
            except Exception as e:
                print(f"[writer:{self.name}] job failed: {e}")
                traceback.print_exc()
            finally:
                self.q.task_done()
    
    def stop(self):
        if not self._alive:
            return 
        self._alive = False
        self.q.join()
        self.q.put(None)
        self._t.join(timeout=10)

def carla_transform_to_mat(tr: carla.Transform):
    import math
    loc, rot = tr.location, tr.rotation
    cy,sy = math.cos(math.radians(rot.yaw)) , math.sin(math.radians(rot.yaw))
    cp,sp = math.cos(math.radians(rot.pitch)), math.sin(math.radians(rot.pitch))
    cr,sr = math.cos(math.radians(rot.roll)), math.sin(math.radians(rot.roll))
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr +sy*sr, 0.0],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, 0.0],
        [ -sp, cp*sr, cp*cr, 0.0],
        [0.0,0.0,0.0,1.0]
    ], dtype = np.float32)
    T = np.eye(4,dtype = np.float32)
    T[:3, 3] = [loc.x, loc.y, loc.z]
    return T @ R

def invert_tf(M):
    R = M[:3,:3]; t= M[:3,3]
    Minv = np.eye(4, dtype = np.float32)
    Minv[:3,:3] = R.T
    Minv[:3,3] = -R.T @ t
    return Minv

def save_lidar_ply_binary(filename: str, pts: "np.ndarray"):
    """
    Save points as binary little-endian PLY.
    pts shape: (N,3) for XYZ or (N,4) for XYZI (float32)
    """
    assert np is not None, "NumPy required"
    pts = np.asarray(pts, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] in (3, 4), "pts must be (N,3) or (N,4)"
    N, C = pts.shape

    with open(filename, "wb") as f:
        # header
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if C == 4:
            header.append("property float intensity")
        header.append("end_header\n")
        f.write(("\n".join(header)).encode("ascii"))

        # body
        pts.tofile(f)

        
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
        
    s = world.get_settings()
    print("[world] sync =",s.synchronous_mode," | fixed_dt=",s.fixed_delta_seconds)
    
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
    v_bps = [bp for bp in lib.filter("vehicle.*") if int(bp.get_attribute("number_of_wheels").as_int())>=4 and all(bad not in bp.id for bad in ["ambulance","firetruck","bus","carlacola","t2","tesla.model3"])]
    
    spawns = safe_lane_center_spawns(world)
    random.shuffle(spawns)
    
    ids: List[int] = []
    
    if reserved_first > 0 and len(spawns)>0:
        spawns = spawns[reserved_first:] 
        
    count = min(n, len(spawns))
        
    
    
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
def attach_sensors_async(world:carla.World, ego: carla.Actor, sensors_cfg: list, out_root:str):
    os.makedirs(out_root, exist_ok=True)
    lib = world.get_blueprint_library()
    sensor_actors = []
    writers = []
    
    class LabelsWriter(AsyncDiskWriter):
        pass
    
    
    #img_writer = AsyncDiskWriter("camera", max_queue=2048)
    lidar_writer = AsyncDiskWriter("lidar", max_queue=512)
    event_writer = AsyncDiskWriter("events", max_queue=64)
    #radar_writer = AsyncDiskWriter("radar",max_queue=1024)
    writers.extend([lidar_writer, event_writer])
    
    cam_dirs = {}
    lidar_dirs = {}
    events_dirs = os.path.join(out_root, "events")
    os.makedirs(events_dirs, exist_ok=True)
    
    def ensure_dir(d):
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)   
            
    # def make_cam_cb(subdir: str, sid: str):
    #     path_dir = os.path.join(out_root, subdir)
    #     ensure_dir(path_dir)

    #     def cb(image: carla.Image):
    #         try:
    #             frame = image.frame
    #             ext = ".jpg" if USE_JPEG else ".png"
    #             out_path = os.path.join(path_dir, f"{sid}_{frame:06d}{ext}")

    #             if Image is not None and np is not None:
    #                 # copy raw BGRA bytes now; encode in writer thread
    #                 arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    #                 rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB (+ copy to own memory)

    #                 def job(rgb=rgb, out_path=out_path):
    #                     if USE_JPEG:
    #                         Image.fromarray(rgb).save(out_path, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING, optimize=True)
    #                     else:
    #                         Image.fromarray(rgb).save(out_path, format="PNG", optimize=False)
    #                 img_writer.submit(job)
    #             else:
    #                 # fallback: direct CARLA save (sync/slow, but safe)
    #                 image.save_to_disk(out_path)

    #         except Exception as e:
    #             print(f"[sensor:{sid}] camera cb error: {e}")
    #             traceback.print_exc()
    #     return cb

    
    def make_lidar_cb_ego(subdir: str, sid: str, sensor: carla.Actor, ego: carla.Actor):
        path_dir = os.path.join(out_root, subdir)
        ensure_dir(path_dir)

        def cb(meas: carla.LidarMeasurement):
            try:
                frame = meas.frame
                if np is not None:
                    # === read XYZI ===
                    pts4 = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4).copy()  # [x,y,z,intensity]
                    xyz = pts4[:, :3]
                    inten = pts4[:, 3:4]  # keep as column

                    # sensor->world & ego->world
                    S_w = carla_transform_to_mat(sensor.get_transform())
                    E_w = carla_transform_to_mat(ego.get_transform())
                    E_w_inv = invert_tf(E_w)

                    # transform XYZ to ego: ego^-1 * (sensor_w * p)
                    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
                    xyz_h = np.concatenate([xyz, ones], axis=1)
                    xyz_ego_h = (E_w_inv @ (S_w @ xyz_h.T)).T
                    xyz_ego = xyz_ego_h[:, :3]

                    # === reattach intensity ===
                    pts_ego4 = np.concatenate([xyz_ego, inten], axis=1).astype(np.float32)

                    out_path = os.path.join(path_dir, f"{sid}_{frame:06d}.ply")
                    npy_out = os.path.join(path_dir, f"{sid}_{frame:06d}.npy")
                    lidar_writer.submit(lambda p=pts_ego4, fn=npy_out: np.save(fn, p))
                else:
                    # fallback raw dump
                    raw = bytes(meas.raw_data)
                    out_path = os.path.join(path_dir, f"{sid}_{frame:06d}.bin")
                    lidar_writer.submit(lambda r=raw, fn=out_path: open(fn, "wb").write(r))
            except Exception as e:
                print(f"[sensor:{sid}] lidar cb error: {e}")
                traceback.print_exc()
        return cb

    def make_lidar_cb(subdir: str, sid: str):
        path_dir = os.path.join(out_root, subdir)
        ensure_dir(path_dir)

        def cb(meas: carla.LidarMeasurement):
            try:
                frame = meas.frame
                if np is not None:
                    pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3].copy()
                    out_path = os.path.join(path_dir, f"{sid}_{frame:06d}.ply")
                    lidar_writer.submit(lambda pts=pts, out_path=out_path: save_lidar_ply_binary(out_path, pts))
                else:
                    # fallback: store raw bytes quickly (no per-point loop)
                    raw = bytes(meas.raw_data)  # copy
                    out_path = os.path.join(path_dir, f"{sid}_{frame:06d}.bin")
                    lidar_writer.submit(lambda raw=raw, out_path=out_path: open(out_path, "wb").write(raw))
            except Exception as e:
                print(f"[sensor:{sid}] lidar cb error: {e}")
                traceback.print_exc()
        return cb

    # def make_radar_cb(subdir: str, sid:str):
    #     path_dir = os.path.join(out_root, subdir)
    #     ensure_dir(path_dir)
        
    #     def cb(meas: carla.RadarMeasurement):
    #         try:
    #             frame = meas.frame
    #             if np is not None:
    #                 data = np.array([[d.depth, d.azimuth, d.altitude, d.velocity]for d in meas],dtype=np.float32)
    #                 out_path = os.path.join(path_dir, f"{sid}_{frame:06d}.npy")
    #                 radar_writer.submit(lambda data= data, out_path = out_path: np.save(out_path,data))
    #             else:
    #                 out_path = os.path.join(path_dir, f"{sid}_{frame:06d}.txt")
    #                 lines = "".join(f"{d.depth:.3f},{d.azimuth:.3f},{d.altitude:.3f},{d.velocity:.3f}\n" for d in meas)
    #                 radar_writer.submit(lambda lines = lines, out_path=out_path: open(out_path,"w").write(lines))
    #         except Exception as e:
    #             print(f"[sensor:{sid}] radar cb error: {e}")
    #             traceback.print_exc()
    #     return cb

    def make_event_writer(name: str, header: str):
        csv_path = os.path.join(events_dirs, f"{name}.csv")
        # create once with header
        with open(csv_path, "w", buffering=1) as f:
            f.write(header)

        def write_line(line: str, csv_path=csv_path):
            def job(line=line, csv_path=csv_path):
                with open(csv_path, "a", buffering=1) as f:
                    f.write(line)
            event_writer.submit(job)

        return write_line  
    
    write_collision = make_event_writer("collisions", "frame,other_type,x,y,z,nx,ny,nz\n")
    write_laneinv = make_event_writer("lane_invasions", "frame,types\n")
    write_imu = make_event_writer("imu", "frame,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,compass\n")
    write_gnss = make_event_writer("gnss", "frame,lat,lon,alt\n")
    
    for s in sensors_cfg or []:
        sid = s.get("id", s.get("type","sensor"))
        stype = s.get("type")
        trcfg = s.get("transform",{})
        attrs = s.get("attributes",{})
        
        try:
            bp = lib.find(stype)
        except Exception:
            print(f"[sensor] unknown type {stype}, skipping")
            continue 
        
        for k,v in attrs.items():
            if bp.has_attribute(k):
                bp.set_attribute(k,str(v))
                
        tr = carla.Transform(
            carla.Location(float(trcfg.get("x",0)), float(trcfg.get("y",0)), float(trcfg.get("z",0))),
            carla.Rotation(float(trcfg.get("pitch",0)), float(trcfg.get("yaw",0)),float(trcfg.get("roll",0)))
        )
        
        sensor = world.try_spawn_actor(
            bp,tr,attach_to=ego,attachment_type=carla.AttachmentType.Rigid
        )
        if not sensor:
            print(f"[sensor:{sid}] failed to spawn; skipping")
            continue 
        sensor_actors.append(sensor)
        
        # if "sensor.camera" in stype:
        #     sub = f"{sid}_images"
        #     cb = make_cam_cb(sub,sid)
        #     sensor.listen(cb)
        #     print(f"[sensor] attached {sid} ({stype}) -> {sub}/")
        if "sensor.lidar" in stype:
            sub = f"{sid}_lidar"
            cb = make_lidar_cb_ego(sub, sid,sensor,ego)
            sensor.listen(cb)
            print(f"[sensor] attached {sid} ({stype}) -> {sub}/ (ego-frame)")
        
        # elif "sensor.other.radar" in stype:
        #     sub = f"{sid}_radar"
        #     cb = make_radar_cb(sub,sid)
        #     sensor.listen(cb)
        #     print(f"[sensor] attached {sid} ({stype}) -> {sub}/")

        # Events/others (keep super light)
        elif stype == "sensor.other.collision":
            sensor.listen(lambda ev: (
                write_collision(f"{ev.frame},{(ev.other_actor.type_id if ev.other_actor else 'unknown')},"
                                f"{ev.transform.location.x:.3f},{ev.transform.location.y:.3f},{ev.transform.location.z:.3f},"
                                f"{ev.normal_impulse.x:.3f},{ev.normal_impulse.y:.3f},{ev.normal_impulse.z:.3f}\n")
            ))
            print(f"[sensor] attached {sid} ({stype}) -> events/collision.csv")
        elif stype == "sensor.other.lane_invasion":
            sensor.listen(lambda ev: (
                write_laneinv(f"{ev.frame}," +
                              "+".join([str(x).split('.')[-1] for x in ev.crossed_lane_markings]) + "\n")
            ))
            print(f"[sensor] attached {sid} ({stype}) -> events/lane_invasion.csv")
        elif stype == "sensor.other.imu":
            sensor.listen(lambda m: (
                write_imu(f"{m.frame},{m.accelerometer.x:.6f},{m.accelerometer.y:.6f},{m.accelerometer.z:.6f},"
                          f"{m.gyroscope.x:.6f},{m.gyroscope.y:.6f},{m.gyroscope.z:.6f},{m.compass:.6f}\n")
            ))
            print(f"[sensor] attached {sid} ({stype}) -> events/imu.csv")
        elif stype == "sensor.other.gnss":
            sensor.listen(lambda m: (
                write_gnss(f"{m.frame},{m.latitude:.8f},{m.longitude:.8f},{m.altitude:.3f}\n")
            ))
            print(f"[sensor] attached {sid} ({stype}) -> events/gnss.csv")
        else:
            # generic no-op listener so sensor is alive
            sensor.listen(lambda _msg: None)
            print(f"[sensor] attached {sid} ({stype})") 
            
    return sensor_actors, writers

def map_folder(world: carla.World) -> str:
    raw = world.get_map().name
    return raw.replace("\\","/").split("/")[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed",type= int, default = -1 , help ="-1 = auto")
    ap.add_argument("--warmup_ticks", type=int, default=0)
    ap.add_argument("--dataset_root", default="dataset", help="where to write runs")
    args = ap.parse_args()
    cfg= load_config(args.config)
    random.seed(int(cfg.get("seed",42)))
    
    

    seed = args.seed if args.seed >=0 else (int(time.time()*1000)& 0xffffffff)
    cfg['seed']=seed
    random.seed(seed)
    if np is not None:
        np.random.seed(seed & 0xffffffff)
    print (f"[seed] using {seed}")
    client = carla.Client(cfg.get("host","127.0.0.1"), cfg.get("port",2000))
    client.set_timeout(10.0)
    
    
    
    world, tm, tr_cfg = configure_world_and_tm(client, cfg)
    town = map_folder(world)
    run_dir = make_run_dir(os.path.join(args.dataset_root,town,f"seed_{cfg['seed']}","run"))
    
    labels_dir = os.path.join(run_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    labels_writer= AsyncDiskWriter("labels",max_queue=256)
    
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
    
    #run_dir = make_run_dir("run")
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "seed": cfg["seed"],
            "map": world.get_map().name,
            "vehicles": len(veh_ids),
            "walkers": len(walk_ids),
            "fixed_delta_seconds": world.get_settings().fixed_delta_seconds,
            "weather": str(world.get_weather())
        }, f, indent=2)
    sensors =[]
    writers = []
    if ego:
        sensors, writers = attach_sensors_async(world,ego,cfg.get("sensors",[]), out_root=run_dir)
        
    print(f"[run] driving... (veh={len(veh_ids)}) (walk={len(walk_ids)}) (ego={'yes' if ego else 'no'})")

    t_end = (time.time()+float(args.seconds)) if args.seconds else None
    retarget_radius = float(ped_cfg.get("retarget_radius",2.0))
    try:
        while(t_end is None) or(time.time()<t_end):
            world.tick()
            try:
                if ego:
                    frame_id = world.get_snapshot().frame
                    ego_mat  = carla_transform_to_mat(ego.get_transform())
                    ego_inv  = invert_tf(ego_mat)

                    labels = []

                    # Vehicles
                    for a in world.get_actors().filter("*vehicle*"):
                        if a.id == ego.id:
                            continue
                        bb = a.bounding_box
                        if not bb:
                            continue
                        aw = carla_transform_to_mat(a.get_transform())
                        c_world = np.array([bb.location.x, bb.location.y, bb.location.z, 1.0], dtype=np.float32)
                        c_ego   = ego_inv @ (aw @ c_world)

                        w = bb.extent.y * 2.0
                        l = bb.extent.x * 2.0
                        h = bb.extent.z * 2.0

                        yaw_deg = a.get_transform().rotation.yaw - ego.get_transform().rotation.yaw
                        v = a.get_velocity()
                        v_world = np.array([v.x, v.y, v.z, 0.0], dtype=np.float32)
                        v_ego   = ego_inv @ v_world
                        vx, vy  = float(v_ego[0]), float(v_ego[1])

                        labels.append({
                            "class": "vehicle",
                            "x": float(c_ego[0]), "y": float(c_ego[1]), "z": float(c_ego[2]),
                            "w": float(w), "l": float(l), "h": float(h),
                            "yaw_deg": float(yaw_deg),
                            "vx": vx, "vy": vy, "score": 1.0
                        })

                    # Pedestrians  (UNINDENTED, separate loop)
                    for a in world.get_actors().filter("walker.pedestrian.*"):
                        bb = a.bounding_box
                        if not bb:
                            continue
                        aw = carla_transform_to_mat(a.get_transform())
                        c_world = np.array([bb.location.x, bb.location.y, bb.location.z, 1.0], dtype=np.float32)
                        c_ego   = ego_inv @ (aw @ c_world)
                        w = bb.extent.y * 2.0
                        l = bb.extent.x * 2.0
                        h = bb.extent.z * 2.0
                        yaw_deg = a.get_transform().rotation.yaw - ego.get_transform().rotation.yaw
                        v = a.get_velocity()
                        v_world = np.array([v.x, v.y, v.z, 0.0], dtype=np.float32)
                        v_ego   = ego_inv @ v_world
                        vx, vy  = float(v_ego[0]), float(v_ego[1])

                        labels.append({
                            "class": "pedestrian",
                            "x": float(c_ego[0]), "y": float(c_ego[1]), "z": float(c_ego[2]),
                            "w": float(w), "l": float(l), "h": float(h),
                            "yaw_deg": float(yaw_deg),
                            "vx": vx, "vy": vy, "score": 1.0
                        })

                    # Single write per frame
                    
                    outp = os.path.join(labels_dir, f"labels_{frame_id:06d}.json")
                    payload = {
                        "frame": int(frame_id),
                        "ego_id": int(ego.id),
                        "timestamp": float(world.get_snapshot().timestamp.elapsed_seconds),
                        "labels": labels
                    }
                    labels_writer.submit(lambda p=payload, fn=outp: open(fn, "w").write(json.dumps(p)))
            except Exception as e:
                print("[labels] error:", e)
    
                                    
            
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
        
        try:
            world.tick()
            world.tick()
            world.tick()
        except Exception:
            pass
        
        for w in writers:
            try:
                w.stop()
            except Exception:
                pass
        print( 
              f"lidar dropped = {writers[0].dropped if len(writers)>0 else 0}"
              )

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
        
        try:
            labels_writer.stop()
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
