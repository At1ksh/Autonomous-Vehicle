# async_walkers.py
# CARLA 0.9.14 — robust walker wander script using controller->walker ID mapping

import argparse
import random
import time
import math
import carla

def euclid(a: carla.Location, b: carla.Location) -> float:
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def wait_or_tick(world: carla.World, sync: bool, seconds: float = 0.05, ticks: int = 1):
    if sync:
        for _ in range(max(1, ticks)):
            world.tick()
    else:
        time.sleep(max(0.0, seconds))

def get_actor_with_retry(world: carla.World, actor_id: int, sync: bool, max_tries: int = 60):
    for _ in range(max_tries):
        actor = world.get_actor(actor_id)
        if actor is not None:
            return actor
        wait_or_tick(world, sync, seconds=0.05, ticks=1)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--num-walkers', type=int, default=30)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--speed', type=float, default=1.3)
    parser.add_argument('--retarget-radius', type=float, default=2.0)
    parser.add_argument('--retarget-interval', type=float, default=0.5)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    original_settings = world.get_settings()
    walkers_ids = []
    ctrl_ids = []
    ctrl_to_walker = {}   # controller_id -> walker_id
    current_goals = {}    # controller_id -> carla.Location

    try:
        # --- optional sync ---
        if args.sync:
            s = world.get_settings()
            s.synchronous_mode = True
            world.apply_settings(s)

        bp = world.get_blueprint_library()

        # --- choose spawn points on navmesh ---
        spawns = []
        for _ in range(args.num_walkers):
            loc = world.get_random_location_from_navigation()
            if loc:
                spawns.append(carla.Transform(loc))
        if not spawns:
            print("No navmesh locations found.")
            return

        # --- spawn walkers ---
        walker_bps = bp.filter('walker.pedestrian.*')
        spawn_cmds = []
        for tr in spawns:
            wbp = random.choice(walker_bps)
            if wbp.has_attribute('is_invincible'):
                wbp.set_attribute('is_invincible', 'false')
            spawn_cmds.append(carla.command.SpawnActor(wbp, tr))
        results = client.apply_batch_sync(spawn_cmds, args.sync)
        for r in results:
            if not r.error:
                walkers_ids.append(r.actor_id)

        if not walkers_ids:
            print("Failed to spawn walkers.")
            return

        # Allow walkers to register
        wait_or_tick(world, args.sync, seconds=0.1, ticks=2)

        # --- spawn controllers, preserving order to build a mapping ---
        ctrl_bp = bp.find('controller.ai.walker')
        ctrl_cmds = [carla.command.SpawnActor(ctrl_bp, carla.Transform(), wid) for wid in walkers_ids]
        results = client.apply_batch_sync(ctrl_cmds, args.sync)

        # Build controller list and controller->walker mapping by index
        j = 0
        for r in results:
            if not r.error:
                cid = r.actor_id
                ctrl_ids.append(cid)
                # mapping: controller just spawned for walkers_ids[j]
                if j < len(walkers_ids):
                    ctrl_to_walker[cid] = walkers_ids[j]
                j += 1

        if not ctrl_ids:
            print("Failed to spawn controllers; cleaning up walkers.")
            client.apply_batch([carla.command.DestroyActor(w) for w in walkers_ids])
            return

        # Let controllers register
        wait_or_tick(world, args.sync, seconds=0.1, ticks=2)

        # --- get controller actor handles with retries ---
        controllers = []
        for cid in ctrl_ids:
            actor = get_actor_with_retry(world, cid, args.sync, max_tries=60)
            if actor:
                controllers.append(actor)

        if not controllers:
            print("No controller actors resolved; aborting.")
            return

        # --- start controllers & initial goals ---
        for ctrl in controllers:
            try:
                ctrl.start()
                ctrl.set_max_speed(args.speed)
            except Exception as e:
                print(f"Controller {ctrl.id} start failed: {e}")

        for ctrl in controllers:
            dest = world.get_random_location_from_navigation()
            if dest:
                try:
                    ctrl.go_to_location(dest)
                    current_goals[ctrl.id] = dest
                except Exception:
                    pass

        # --- wander loop ---
        start = time.time()
        last_check = start

        def reassign(ctrl):
            d = world.get_random_location_from_navigation()
            if d:
                try:
                    ctrl.go_to_location(d)
                    current_goals[ctrl.id] = d
                except Exception:
                    pass

        while (time.time() - start) < args.duration:
            if args.sync:
                world.tick()
            else:
                time.sleep(0.03)

            now = time.time()
            if (now - last_check) >= args.retarget_interval:
                for ctrl in controllers:
                    try:
                        # Look up the walker via our mapping
                        wid = ctrl_to_walker.get(ctrl.id)
                        if wid is None:
                            continue
                        w = world.get_actor(wid)
                        if not w:
                            continue

                        goal = current_goals.get(ctrl.id)
                        if goal is None:
                            reassign(ctrl)
                            continue

                        loc = w.get_location()
                        if euclid(loc, goal) <= args.retarget_radius:
                            reassign(ctrl)
                    except RuntimeError:
                        continue
                last_check = now

    finally:
        # --- teardown ---
        try:
            for cid in ctrl_ids:
                actor = world.get_actor(cid)
                if actor:
                    try:
                        actor.stop()
                    except Exception:
                        pass
            if ctrl_ids:
                client.apply_batch([carla.command.DestroyActor(x) for x in ctrl_ids])
            if walkers_ids:
                client.apply_batch([carla.command.DestroyActor(x) for x in walkers_ids])

            if args.sync:
                world.apply_settings(original_settings)
        except Exception as e:
            print("Teardown issue:", e)

        print("Wander session complete. Teardown done.")

if __name__ == '__main__':
    main()
