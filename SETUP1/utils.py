import json
import os
import random
from typing import Any, Dict, List

import carla

def load_config(path: str="config.json") -> Dict[str,Any]:
    with open(path,'r') as f:
        return json.load(f)
    
def get_client(cfg) -> carla.Client:
    client = carla.Client(cfg['host'],int(cfg['port']))
    client.set_timeout(float(cfg.get("timeout",10.0)))
    return client

def get_world_and_tm(client: carla.Client, cfg):
    world = client.get_world()
    tm = client.get_trafficmanager(int(cfg.get("tm_port",8000)))
    return world, tm

def set_seed(cfg):
    seed = int(cfg.get("seed",42))
    random.seed(seed)
    
def to_transform(t: Dict[str, float]) -> carla.Transform:
    # dict -> carla.Transform
    loc = carla.Location(x=t.get("x", 0.0), y=t.get("y", 0.0), z=t.get("z", 0.0))
    rot = carla.Rotation(pitch=t.get("pitch", 0.0), yaw=t.get("yaw", 0.0), roll=t.get("roll", 0.0))
    return carla.Transform(loc, rot)

def find_actor_by_role(world: carla.World, role_name: str) -> carla.Actor:
    actors = world.get_actors().filter("*")
    hits = [a for a in actors if a.attributes.get("role_name")== role_name]
    return hits[0] if hits else None

def ensure_map(client: carla.Client, map_name: str) -> carla.World:
    world = client.get_world()
    if map_name and map_name != world.get_map().name:
        world = client.load_world(map_name)
    return world

def configure_world(world: carla.World, cfg) -> None:
    settings = world.get_settings()
    sync = bool(cfg.get("synchronous_mode", False))
    settings.synchronous_mode = sync
    if sync:
        settings.fixed_delta_seconds = float(cfg.get("fixed_delta_seconds", 0.05))
    else:
        settings.fixed_delta_seconds = None
    world.apply_settings(settings)

def apply_weather(world: carla.World, preset: str):
    # Supported common presets (extend as needed)
    presets = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "ClearNight": carla.WeatherParameters.ClearNight,
    "CloudyNight": carla.WeatherParameters.CloudyNight,
    "WetNight": carla.WeatherParameters.WetNight,
    "SoftRainNight": carla.WeatherParameters.SoftRainNight,
    "MidRainyNight": carla.WeatherParameters.MidRainyNight,
    "HardRainNight": carla.WeatherParameters.HardRainNight,
    }
    wp = presets.get(preset, carla.WeatherParameters.ClearNoon)
    world.set_weather(wp)

def set_tm_global(world: carla.World, tm: carla.TrafficManager, cfg):
    tr = cfg.get("traffic", {})
    tm.set_synchronous_mode(world.get_settings().synchronous_mode)
    tm.set_hybrid_physics_mode(bool(tr.get("hybrid_physics_mode", True)))
    tm.set_global_distance_to_leading_vehicle(float(tr.get("min_distance", 2.0)))
    tm.global_percentage_speed_difference(float(tr.get("global_speed_perc_diff", -10)))
    tm.set_respawn_dormant_vehicles(True)

    # NOTE: Lane-change, ignore_lights/signs, per-vehicle speed diffs, etc.
    # are PER-ACTOR settings in 0.9.14. We'll apply those after vehicles are spawned.
    # (So nothing else to set here globally.)
      
def make_dir(path: str):
    os.makedirs(path,exist_ok=True)
    
def get_spawn_point(world:carla.World, choice: str= "random") -> carla.Transform:
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        raise RuntimeError("No spawn points available in the map")
    if choice == "random":
        return random.choice(spawns)
    if choice.startswith("idx:"):
        idx = int(choice.split(":", 1)[1])
        return spawns[idx % len(spawns)]
    return spawns[0]