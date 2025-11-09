from LidarCNN.lidar_env import LidarEnv
from stable_baselines3 import PPO


env = LidarEnv(
    lidar_dir=r"E:\FusionRL\Codes\LidarOnlySetup\dataset\Town10HD_Opt\seed_488783317\run_20251026_021934\lidar_top_lidar",
    label_dir=r"E:\FusionRL\Codes\LidarOnlySetup\dataset\Town10HD_Opt\seed_488783317\run_20251026_021934\labels",
    mode="offline"
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

obs, info = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Action: {action}, Reward: {reward:.3f}")
    if done:
        obs, info = env.reset()


for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward:.3f}")
    if terminated or truncated:
        obs, info = env.reset()