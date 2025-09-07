import numpy as np

file_path = r"e:\FusionRL\data\testlidardata\frame_000165.npy"

points = np.load(file_path)

print("Number of points:", len(points))

x = points['x']
y = points['y']
z = points['z']
intensity = points['cos_incidence']
object_idx = points['object_idx']
object_tag = points['object_tag']

print("First 5 points:")
for i in range(1496):
    print(f"Point {i}: x={x[i]}, y={y[i]}, z={z[i]}, intensity={intensity[i]}, object_idx={object_idx[i]}, object_tag={object_tag[i]}")