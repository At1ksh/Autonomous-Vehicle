import argparse
import time

from utils import load_config, get_client 

def main(args):
    cfg = load_config(args.config)
    client = get_client(cfg)
    world = client.get_world()
    settings = world.get_settings()
    
    if not settings.synchronous_mode:
        print("[ticker] world is not in synchronous model; nothing to do"
              "Set 'synchronous_mode': true in config.json and run boostrap_world.py")
        return 
    
    fds = settings.fixed_delta_seconds 
    print(f"[ticker] Starting. sync = True, fixed_Delta_seconds = {fds}. Ctrl + C to stop")
    print_every = max(0,int(args.print_every))
    sleep_s = max(0.0, float(args.sleep))
    
    i = 0
    try:
        while True:
            frame = world.tick()
            i += 1
            if print_every and (i%print_every == 0):
                sim_time = frame * (fds or 0.0)
                print(f"[ticker] frame = {frame}, sim_time = {sim_time:.2f}s")
            if sleep_s > 0.0:
                time.sleep(sleep_s)
                
    except KeyboardInterrupt:
        print("\n[ticker] stopped (by user)")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--print_every", default = 30, help = "Log every N ticks(0 to disable)")
    ap.add_argument("--sleep", default = 0.0, help = "Optional sleep per loop to reduce CPU")
    main(ap.parse_args())