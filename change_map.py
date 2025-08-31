import carla

client=carla.get_client("localhost",2000)
client.setTimeout(5)
world=client.get_world()

