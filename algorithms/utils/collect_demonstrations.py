import gymnasium as gym
import pygame
from collections import deque
import torch
from replay_buffer import Replay_Buffer
import pickle
import numpy as np
# cuda availablity
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# create expert replay buffer
expert_buffer = Replay_Buffer(5000)

# init environment
env = gym.make("MountainCarHillTop-v0", render_mode = "human")
state0, _ = env.reset()
state0 = torch.tensor(state0).to(device)
pygame.init()

# collect human demonstrations
for i in range(2000):
    
    # get human action
    a = 1 # no-op
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        a = 0
    elif keys[pygame.K_d]:
        a = 2

    #a = env.action_space.sample()
    state1, reward, done, truncation, _ = env.step(a)
    
    #print("Position: {}    Velocity: {}".format(state1[0], state1[1]))
    #print("Reward: {}".format(reward))
    height = np.sin(3* state0[0]) *.45 *.55
    print("height?: {}".format(height))
    state1 = torch.tensor(state1).to(device)
    
    # save human experience to replay buffer
    expert_buffer.append(state0, state1, a, reward, int(done))
    
    state0 = state1.detach().clone().to(device)

    if truncation:
        print("RESETTING")
        state0, _ = env.reset()
        state0 = torch.tensor(state0).to(device)


# save human data
save_human_demo = True
if save_human_demo:
    print("Saving human data...")
    with open("DISCARDexpert_data_MountainCarHillTop-v0", 'wb') as file:
        pickle.dump(expert_buffer, file)

    print("done.")

