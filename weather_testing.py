import carla,random,time

client = carla.Client("localhost",2000)
client.set_timeout(5.0)
world=client.get_world()

#carla presets
from carla import WeatherParameters as W
print("Map:", world.get_map().name)
print("Before:", world.get_weather())
time.sleep(5)
world.set_weather(carla.WeatherParameters(cloudiness=100.0,precipitation=100.0,precipitation_deposits=100.0,wind_intensity=80.0,sun_altitude_angle=-15.0,sun_azimuth_angle=200.0))
print("After :", world.get_weather())
# print(W)
# PRESETS = [
#     W.ClearNoon, W.CloudyNoon, W.WetNoon, W.WetCloudyNoon, 
#     W.SoftRainNoon, W.MidRainyNoon, W.HardRainNoon,
#     W.ClearSunset, W.CloudySunset, W.WetSunset, W.WetCloudySunset,
#     W.SoftRainSunset, W.MidRainSunset,  W.HardRainSunset
# ]
# world.set_weather(carla.WeatherParameters.HardRainSunset)
# #choose a random one
# choice = random.choice(PRESETS)
# world.set_weather(choice)
# print("Applied Weather Condition:",choice)
