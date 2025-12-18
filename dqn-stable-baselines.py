import gymnasium as gym
from stable_baselines3 import DQN

# init environment
env = gym.make("CartPole-v1")

# init model
model = DQN("MlpPolicy", env, verbose=1)

# train dqn model
model.learn(total_timesteps=200000, log_interval = 4)

model.save("dqn_CartPole-v1")

model = DQN.load("dqn_CartPole-v1")


# make a human viewable env
env = gym.make("CartPole-v1", render_mode = "human")

# Rollout the model
obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)

    if done:
        obs, info = env.reset()
