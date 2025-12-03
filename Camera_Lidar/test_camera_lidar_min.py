from gym_camera_and_lidar.carla_gym_camera_lidar import CarlaGymCameraLiDAREnv
import cv2
import numpy as np

env = CarlaGymCameraLiDAREnv()

obs, _ = env.reset()
print("Camera obs shape:", obs["camera"].shape)
print("Lidar obs shape:", obs["lidar"].shape)

k = 1
for _ in range(400):
    k += 1
    if k <= 100:
        # Forward: accel > 0
        action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    else:
        # Reverse: accel < 0
        action = np.array([0.0, -0.5, 0.0], dtype=np.float32)

    obs, reward, terminated, truncated, _ = env.step(action)
    cv2.imshow("CARLA Camera", obs["camera"])
    if cv2.waitKey(1) & 0xFF == 27:
        break



env.close()
cv2.destroyAllWindows()