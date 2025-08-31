import carla
import time 

#connect firstly man
client = carla.Client("localhost",2000)
client.set_timeout(5)
world=client.get_world()

#check light manager availability
if not hasattr(world,"get_lightmanager"):
    raise RuntimeError("This CARLA build does not expose LightManager (0.9.14+ required)")

#light manager
lm= world.get_lightmanager()

#function to change group light intensity
def set_group_intensity(group:carla.LightGroup,intensity: float):
    lights = [L for L in lm.get_all_lights() if L.get_light_group()==group]
    for L in lights:
        lm.set_intensity(L, intensity)
        
#Turning off street/city lights
#Street light
set_group_intensity(carla.LightGroup.Street,0)
print("street lights set to 0 intensity")

# building lights (windows, signs etc)
set_group_intensity(carla.LightGroup.Building,0)
print("building lights set to 0 intensity")

#turn of decorative light
set_group_intensity(carla.LightGroup.Other, 0)
print("decorative lights set to 0 intensity")\
    
#light vehicle light
def set_vehicle_lights_off(vehicle: carla.Vehicle):
    vehicle.set_light_state(carla.VehicleLightState.NONE)
    
#example usage (spanwned/ego vehicle)
#set_vehicle_lights_off(ego_vehicle)

def restore_city_lights(street=10.0,building=5.0, other=3.0):
    set_group_intensity(carla.LightGroup.Steet, street)
    set_group_intensity(carla.LightGroup.Building, building)
    set_group_intensity(carla.LightGroup.other, other)
    print("City lights restored")
    
time.sleep(10)
restore_city_lights()