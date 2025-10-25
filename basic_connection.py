import carla
import random
import time 

#Connect to server
client = carla.Client("127.0.0.1",2000)
client.set_timeout(5.0)

world=client.get_world()
print("Loaded map:",world.get_map().name)