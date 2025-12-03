import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time


class CarlaGymEnvMinimal(gym.Env):
    def __init__(self, host="127.0.0.1", port=2000,max_steps=300):
        self.host = host
        self.port = port
        self.town = "Town10HD_Opt"
        self.max_steps = max_steps
        
        
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        if self.world.get_map().name.split("/")[-1] != self.town:
            self.client.load_world(self.town)
            time.sleep(2)  # wait for the world to load
            self.world = self.client.get_world()
        
        self.original_settings = self.world.get_settings()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        self.blueprint_lib= self.world.get_blueprint_library()
        
        self.vehicle = None
        self.actor_list = []
        
        
        self.observation_space = spaces.Box(
            low = -np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low = np.array([-1.0, 0.0,0.0],dtype=np.float32),
            high = np.array([1.0, 1.0,1.0],dtype=np.float32),
            dtype=np.float32
        )
        
        self.step_count = 0
        
        self._spawn_vehicle()
        
    def _spawn_vehicle(self):
        if self.vehicle is not None:
            return 
        
        vehicle_bp = self.blueprint_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = carla.Transform(carla.Location(x=-113.648178, y=-14.281184, z=0.600000), carla.Rotation(pitch=0.000000, yaw=90.642235, roll=0.000000))
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)
        
        for _ in range(5):
            self.world.tick()
    
    def _destroy_actors(self):
        for a in self.actor_list:
            if a is not None:
                a.destroy()
        self.actor_list = []
        self.vehicle = None
        
    
    #Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        
        self._destroy_actors()
        self._spawn_vehicle()
        
        obs = np.zeros((1,), dtype=np.float32)
        return obs, {}
    
    def step(self, action):
        self.step_count+=1
        
        action = np.clip(action,self.action_space.low, self.action_space.high)
        
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])
        
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        
        self.vehicle.apply_control(control) 
        self.world.tick()
        
        #Dummy observation
        obs = np.zeros((1,), dtype=np.float32)
        
        #speed based reward
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) 
        
        reward = 1.0 +0.05*speed
        
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        return obs, float(reward), terminated, truncated, {}
    def close(self):
        self._destroy_actors()
        self.world.apply_settings(self.original_settings)
        