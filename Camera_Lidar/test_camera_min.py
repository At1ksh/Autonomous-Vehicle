import cv2
from stable_baselines3 import PPO
from gym_camera_only.check_carla_gym_camera_only import CarlaGymCameraEnv as CarlaGymCameraOnlyEnv
import numpy as np

env = CarlaGymCameraOnlyEnv()

obs, info = env.reset()
print("Camera obs shape:", obs.shape)

cv2.imshow("CARLA Camera", obs)
cv2.waitKey(2000)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=3000)

obs, info = env.reset()

# for i in range(500):
#     action = np.array([0.2, 0.6, 0.0], dtype=np.float32)  # slight turn + throttle
#     obs, reward, terminated, truncated, info = env.step(action)
#     print("Action:",action, "Reward:",reward)
    
#     cv2.imshow("CARLA Camera", obs)
#     cv2.waitKey(10)

# env.close()
# cv2.destroyAllWindows()
obs, info = env.reset()

for _ in range(300):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print("Action:",action, "Reward:",reward)
    cv2.imshow("CARLA Camera", obs)
    cv2.waitKey(1)
    
    if terminated or truncated:
        obs, info = env.reset()
env.close()
cv2.destroyAllWindows()