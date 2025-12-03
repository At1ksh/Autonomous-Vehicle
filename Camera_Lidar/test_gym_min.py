from stable_baselines3 import PPO
from check_gym_minimal import CarlaGymEnvMinimal

env = CarlaGymEnvMinimal()

obs, info = env.reset()
print("Obs Shape:",obs.shape)

model = PPO("MlpPolicy",env,verbose = 1)
model.learn(total_timesteps=3000)

obs,info = env.reset()

for _ in range(300):
    action, _ = model.predict(obs)
    obs,reward,terminated,truncated,info = env.step(action)
    print("Action:",action, "Reward:",reward)
    
    if terminated or truncated:
        obs,info = env.reset()
env.close()