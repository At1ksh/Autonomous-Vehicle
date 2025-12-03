import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import cv2

class CarlaGymCameraLiDAREnv(gym.Env):
    def __init__(self, host="127.0.0.1", port=2000, town="Town10HD_Opt", max_steps=300):
        super().__init__()
        
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()    
        
        if self.world.get_map().name.split("/")[-1] != town:
            self.world = self.client.load_world(town)
            
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        self.bp_lib = self.world.get_blueprint_library()
        
        self.vehicle = None
        self.camera = None
        self.lidar = None
        self.actor_list = []
        
        self.rgb_image = None 
        self.lidar_data = None
        
        self.step_count = 0
        self.max_steps = max_steps 
        
        self.observation_space = spaces.Dict({
            "camera": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "lidar": spaces.Box(low=0.0, high = 50.0, shape = (64,), dtype=np.float32),
        })
        
        self.action_space = spaces.Box(
            low = np.array([-1.0, 0.0,0.0],dtype=np.float32),
            high = np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self._spawn_vehicle()
        self._attach_camera()
        self._attach_lidar()
        self._prime_sensors()
    def _spawn_vehicle(self):
        vehicle_bp = self.bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = carla.Transform(
            carla.Location(x=-113.6, y=-14.2, z=0.6),
            carla.Rotation(pitch=0.0, yaw=90.6, roll=0.0)
        )

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)

        for _ in range(5):
            self.world.tick()
    
    def _attach_camera(self):
        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "256")
        cam_bp.set_attribute("image_size_y", "256")
        cam_bp.set_attribute("fov", "90")
        
        cam_transform = carla.Transform(
            carla.Location(x=1.5, z=2.0),
            carla.Rotation(pitch=-5.0)
        )
        self.camera = self.world.spawn_actor(
            cam_bp, cam_transform, attach_to=self.vehicle
        )
        
        self.actor_list.append(self.camera)
        self.camera.listen(self._camera_callback)
        
    def _camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:,:,:3]
        self.rgb_image = array
    
    def _attach_lidar(self):
        lidar_bp = self.bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("points_per_second", "56000")
        lidar_bp.set_attribute("rotation_frequency", "20")
        
        lidar_transform = carla.Transform(
            carla.Location(x=0.0, z=2.5)
        )     
        
        self.lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle
        )
        
        self.actor_list.append(self.lidar)
        self.lidar.listen(self._lidar_callback)
        
    def _lidar_callback(self, data):
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = np.reshape(points, (-1, 4))
        distances = np.linalg.norm(points[:, :3], axis=1)

        hist, _ = np.histogram(distances, bins=64, range=(0, 50))
        self.lidar_data = hist.astype(np.float32)
        
    def _prime_sensors(self):
        for _ in range(10):
            self.world.tick()
            if self.rgb_image is not None and self.lidar_data is not None:
                break
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        spawn_point = carla.Transform(
            carla.Location(x=-113.6, y=-14.2, z=0.6),
            carla.Rotation(pitch=0.0, yaw=90.6, roll=0.0)
        )
        
        self.vehicle.set_transform(spawn_point)
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
        
        for _ in range(5):
            self.world.tick()
        
        obs = {
            "camera": cv2.resize(self.rgb_image, (128, 128)),
            "lidar": self.lidar_data
        }
        
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])
        
        if throttle > 0.1:
            brake = 0.0
        
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )
        
        self.world.tick()
        
        obs = {
            "camera": cv2.resize(self.rgb_image, (128, 128)),
            "lidar": self.lidar_data
        }
        
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) 
        reward = 1.0 + 0.05 * speed
        
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        return obs, float(reward), terminated, truncated, {}
    
    def close(self):
        if self.camera:
            self.camera.stop()
        if self.lidar:
            self.lidar.stop()
            
        for a in self.actor_list:
            try:
                a.destroy()
            except:
                pass
        
        self.world.apply_settings(self.original_settings)