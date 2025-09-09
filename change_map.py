import carla

client=carla.Client("localhost",2000)
client.set_timeout(5)

mapname= input("Enter the map name you want")
world=client.load_world(mapname)

print("Now Running map:",world.get_map().name)

