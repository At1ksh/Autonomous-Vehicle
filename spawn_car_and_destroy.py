import carla
import random
import time
#print("hello pop")
#Connect to server
client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.get_world()

blueprint_library = world.get_blueprint_library() 
#print("Blueprint Library:", blueprint_library.filter("vehicle"))

#choose a vehicle
#vehicle_bp = random.choice(blueprint_library.filter("vehicle"))
vehicle_bp = blueprint_library.find("vehicle.audi.tt")
print("Choosen Vehicle:", vehicle_bp)

#spawn point
#spawn_point = random.choice(world.get_map().get_spawn_points())
spawn_point = carla.Transform(carla.Location(x=-113.648178, y=-14.281184, z=0.600000), carla.Rotation(pitch=0.000000, yaw=90.642235, roll=0.000000))
#spawn_point = carla.Transform(carla.Location(x=-113.648178, y=-14.281184, z=0.600000), carla.Rotation(pitch=0.000000, yaw=270.642235, roll=0.000000))
#spawn_point = carla.Transform(carla.Location(x=-103.648178, y=-14.281184, z=0.600000), carla.Rotation(pitch=0.000000, yaw=270.642235, roll=0.000000))
print("Spawn point:", spawn_point)

#spawn the vehicle
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print("Spawned vehicle:", vehicle.id)

#let it  drive a bit
vehicle.set_autopilot(True)
time.sleep(30)

#destroywaw
vehicle.destroy()
print("Vehicle destroyed, script fully done")
