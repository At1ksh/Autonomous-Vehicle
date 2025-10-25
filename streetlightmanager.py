import carla
import time 

#connect firstly man
client = carla.Client("127.0.0.1",2000)
client.set_timeout(5)
world=client.get_world()

#check light manager availability
if not hasattr(world,"get_lightmanager"):
    raise RuntimeError("This CARLA build does not expose LightManager (0.9.14+ required)")

# #light manager
# lm= world.get_lightmanager()

# #function to change group light intensity
# def set_group_intensity(group:carla.LightGroup,intensity: float):
#     lights = [L for L in lm.get_all_lights() if L.get_light_group()==group]
#     for L in lights:
#         lm.set_intensity(L, intensity)
        
# #Turning off street/city lights
# #Street light
# set_group_intensity(carla.LightGroup.Street,0)
# print("street lights set to 0 intensity")

# # building lights (windows, signs etc)
# set_group_intensity(carla.LightGroup.Building,0)
# print("building lights set to 0 intensity")

# #turn of decorative light
# set_group_intensity(carla.LightGroup.Other, 0)
# print("decorative lights set to 0 intensity")\
    
# #light vehicle light
# def set_vehicle_lights_off(vehicle: carla.Vehicle):
#     vehicle.set_light_state(carla.VehicleLightState.NONE)
    
# #example usage (spanwned/ego vehicle)
# #set_vehicle_lights_off(ego_vehicle)

# def restore_city_lights(street=10.0,building=5.0, other=3.0):
#     set_group_intensity(carla.LightGroup.Steet, street)
#     set_group_intensity(carla.LightGroup.Building, building)
#     set_group_intensity(carla.LightGroup.other, other)
#     print("City lights restored")
    
# time.sleep(10)
# restore_city_lights()

lm=world.get_lightmanager()

all_lights=lm.get_all_lights()
print("Found", len(all_lights), "lights in this world")

# for L in all_lights:
#     try: 
#         #intensity = L.get_intensity()
#         color= L.get_color()
#         position = L.get_location()
#         print(f"Light ID={L.id} |  color = {color} | Location = {position}")
#     except Exception as e:
#         print(f"Could not query light {L.id}: {e}")   
# doesnt work in carla to first get_intensity

# for L in lm.get_all_lights():
#     try:
#         lm.set_intensity(L,100.0)
#     except Exception as e:
#         print(f"Could not set light {L.id}: {e}")

# lm.set_intensity(all_lights,0.0)

def apply_batched(func, seq, batch = 64, *args, **kwargs):
    for i in range(0, len(seq), batch):
        part = seq[i:i+batch]
        try:
            func(part, *args, **kwargs)
        except function as e:
            print(f"Batch {i//batch} failed: {e}")
        world.tick()
        
# apply_batched(lm.switch_off, all_lights, batch=54)
# time.sleep(0.1)

apply_batched(lm.set_intensity, all_lights, batch = 64, intensity = 0.0)
time.sleep(0.1)





        
print("All done")