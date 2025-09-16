#sets map, world sync or async, tfmanager globals, and initial weather

import argparse
from utils import load_config, get_client, ensure_map, configure_world, apply_weather, set_seed, set_tm_global, get_world_and_tm

def main(args):
    cfg = load_config(args.config)
    set_seed(cfg)
    client = get_client(cfg)
    
    world = ensure_map(client,cfg.get("map",""))
    configure_world(world, cfg)
    world, tm = get_world_and_tm(client, cfg)
    set_tm_global(world,tm,cfg)

    weather = cfg.get("weather", {}).get("preset","ClearNoon")
    apply_weather(world,weather)
    
    print("[bootsrap] Done. World is {} | sync ={}| weather={}".format(world.get_map().name, world.get_settings().synchronous_mode, weather))
    
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    main(ap.parse_args())
    
    