import gymnasium as gym
import pygame
import deque
import torch

env = gym.make("MountainCar-v0", render_mode = "human")
state0, _ = env.reset()
pygame.init()

demo_len = 20000
state0_demos = deque(maxlen = demo_len)
state1_demos = deque(maxlen = demo_len)

for i in range(5000):
    
    # get human action
    a = 1 # no-op
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        a = 0
    elif keys[pygame.K_d]:
        a = 2

    #a = env.action_space.sample()
    state1, reward, done, _, _ = env.step(a)

    trajectory.append([state0, state1, reward, a])
    
    state0 = state1
    if done:
        state0, _ = env.reset()


