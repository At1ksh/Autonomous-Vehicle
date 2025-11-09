import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from camera_env.camera_env import CameraEnv

# --- Paths ---
image_dir = r"E:\FusionRL\Codes\SETUP1\dataset\Town10HD_Opt\seed_3526230764\run_20251011_130055\hero_cam_front_images"
events_dir = r"E:\FusionRL\Codes\SETUP1\dataset\Town10HD_Opt\seed_3526230764\run_20251011_130055\events"

# --- Init Env ---
def make_env():
    return CameraEnv(image_dir=image_dir, events_dir=events_dir)

env = DummyVecEnv([make_env])   # replaces make_vec_env

# --- PPO ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = PPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log="./ppo_camera_log/")
model.learn(total_timesteps=50000)

# --- Test rollout ---
obs = env.reset()
for _ in range(50):
    action, _ = model.predict(obs)
    result = env.step(action)

    # Handle both gym and gymnasium formats
    if len(result) == 4:
        obs, reward, done, info = result
        truncated = False
    else:
        obs, reward, done, truncated, info = result

    print(f"Action: {action}, Reward: {reward}")

    if done or truncated:
        obs = env.reset()
