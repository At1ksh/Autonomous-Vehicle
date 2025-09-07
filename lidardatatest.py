import carla
import numpy as np
import os
import time


client = carla.Client("localhost",2000)
client.set_timeout(10.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True 
settings.fixed_delta_seconds = 0.033 
world.apply_settings(settings)

blueprints = world.get_blueprint_library()

spawn_point = carla.Transform(carla.Location(x=-113.648178, y=-14.281184, z=0.600000), carla.Rotation(pitch=0.000000, yaw=90.642235, roll=0.000000))
vehicle_bp = blueprints.find("vehicle.audi.tt")
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

out_dir = r"E:\FusionRL\data\testlidardata"
os.makedirs(out_dir, exist_ok=True)

lidar_bp = blueprints.find("sensor.lidar.ray_cast_semantic")
lidar_bp.set_attribute("range", "50")
lidar_bp.set_attribute("rotation_frequency", "10")
lidar_bp.set_attribute("sensor_tick", "0.1")
lidar_bp.set_attribute("points_per_second", "50000")

lidar_transform = carla.Transform(carla.Location(x=0.0,y=0,z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

frame_counter = {"n":0}

def save_lidar(data):
    dtype= np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('cos_incidence', np.float32),
        ('object_idx',  np.uint32),
        ('object_tag',  np.uint32)
    ])
    pts = np.frombuffer(data.raw_data, dtype=dtype)
    
    fname = os.path.join(out_dir, f"frame_{frame_counter['n']:06d}.npy")
    np.save(fname,pts)
    print(f"Saved {fname} with {len(pts)} points")
    frame_counter['n']+=1
    
lidar.listen(save_lidar)

tm=client.get_trafficmanager(8000)
tm.set_synchronous_mode(True)
vehicle.set_autopilot(True)

try:
    for i in range(3000):
        world.tick()
        #time.sleep(0.05)
finally:
    print("Clearning up")
    # lidar.stop()
    # lidar.destroy()
    # vehicle.destroy()
    # settings.synchronous_mode = False
    # world.apply_settings(settings)

    # 1) Stop incoming callbacks
    try:
        lidar.stop()
        world.tick()
    except Exception as e:
        print(f"Could not stop lidar: {e}")
    
    # 2) Destroy actorss
    try:
        lidar.destroy()
    except Exception as e:
        print(f"Could not destroy lidar: {e}")
        
    try:
        vehicle.destroy()
    except Exception as e:
        print(f"Could not destroy vehicle: {e}")
        
    # 3) Unsync tm adsasd
    try:
        tm.set_synchronous_mode(False)
    except Exception as e:
        print(f"Could not unsync tm: {e}")
    
    try:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        world.tick()
    except Exception as e:
        print(f"Could not unsync world: {e}")
        
    time.sleep(1)

    print("All done")