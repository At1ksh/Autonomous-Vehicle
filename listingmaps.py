import carla

client = carla.Client("localhost",2000)
client.set_timeout(5)

maps = client.get_available_maps()
print("Available Maps:")
for m in maps:
    print(" -", m)