import carla
import time

def main():
    host = "127.0.0.1"
    port = 2000

    print("Connecting to CARLA at {}:{} ...".format(host, port))
    client = carla.Client(host, port)
    client.set_timeout(10.0)  # seconds

    # Get world
    world = client.get_world()
    world_map = world.get_map()
    print("Connected. Current map:", world_map.name)

    # Show some basic world info
    settings = world.get_settings()
    print("Synchronous mode:", settings.synchronous_mode)
    print("Fixed delta seconds:", settings.fixed_delta_seconds)

    # Tick a few times to ensure we can step the world
    print("Ticking world 5 times...")
    for i in range(5):
        # If async mode:
        #   world.wait_for_tick()
        # If sync mode:
        #   world.tick()
        # We’ll handle both safely:
        try:
            if settings.synchronous_mode:
                world.tick()
            else:
                world.wait_for_tick()
            print("Tick", i + 1, "OK")
        except Exception as e:
            print("Error during tick:", e)
            break

    print("Test complete. Closing client.")

if __name__ == "__main__":
    main()
