import carla

client=carla.Client("localhost",2000)
client.set_timeout(5)

world=client.load_world("Town10HD")

print("Now Running map:",world.get_map().name)

