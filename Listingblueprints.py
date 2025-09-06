import carla

client = carla.Client("localhost",2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib= world.get_blueprint_library()

# list all vehicles/sensors we have
print("Vehicles:", [bp.id for bp in bp_lib.filter("vehicle.*")])
print("Sensors:", [bp.id for bp in bp_lib.filter("sensor.*")])

#inspect on any blueprint

def show_attrs(blueprint_id):
    bp=bp_lib.find(blueprint_id)
    print(f"\n[{bp.id}] attributes")
    
    for attr in bp:
        print(f"  - {attr.id} | type ={attr.type} | recommended={attr.recommended_values}")
        
#show_attrs("vehicle.kawasaki.ninja")
show_attrs("sensor.lidar.ray_cast_semantic")
#show_attrs("sensor.camera.rgb")