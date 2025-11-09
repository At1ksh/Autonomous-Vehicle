from stable_baselines3 import PPO
import builtins
import numpy as np
def safe_randint(low, high=None, size=None, dtype=int):
    if high is None:
        high = np.iinfo(np.int32).max   # cap upper bound
    if high > np.iinfo(np.int32).max:
        high = np.iinfo(np.int32).max
    rng = np.random.default_rng()
    return rng.integers(low, high, size=size, dtype=dtype)
# patch both numpy and global reference
np.random.randint = safe_randint
builtins.randint = safe_randint
from stable_baselines3.common.env_util import make_vec_env
from camera_env.camera_env_resnet import CameraEnvResNet

env = CameraEnvResNet(
    img_dir=r"E:\FusionRL\Codes\SETUP1\dataset\Town10HD_Opt\seed_3526230764\run_20251011_130055\hero_cam_front_images",
    events_dir=r"E:\FusionRL\Codes\SETUP1\dataset\Town10HD_Opt\seed_3526230764\run_20251011_130055\events",
    device="cpu"
)

env = make_vec_env(lambda: env, n_envs=1)
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
model.learn(total_timesteps=50000)

obs = env.reset()
for _ in range(50):
    action, _ = model.predict(obs)
    result = env.step(action)
    if len(result) == 4:
        obs, reward, done, info = result
    else:
        obs, reward, done, truncated, info = result
    print(f"Action: {action}, Reward: {reward}")
    if done:
        obs = env.reset()
