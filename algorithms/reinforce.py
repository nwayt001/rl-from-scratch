# implement the traditional reinforce algorithm
import gymnasium as gym
import random, pygame
import torch
from torch import nn
from torchinfo import summary
import numpy as np
from collections import deque
from pathlib import Path
import pickle
from replay_buffer import Replay_Buffer
import torch.nn.functional as F


# environment
mountainCar = "MountainCar-v0"
mountainCarHillTop = "MountainCarHillTop-v0"
cartPole = "CartPole-v1"
environment = mountainCar

# training parameters
steps = 0 
gamma = 0.99
learning_rate = 0.00025
train = True
beta = 0.01
use_expert_demo = True

# define policy network
class PolicyMLP(nn.Module):
    def __init__(self, input_size, num_actions):
        super(PolicyMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_actions)
        )

    def forward(self, x):
        return self.model(x)
        
        
if train:

    # load expert data
    if use_expert_demo:
        with open("rl-from-scratch/expert_data_{}".format(environment), 'rb') as file:
            expert_buffer = pickle.load(file)

    # create environment
    env = gym.make(environment)
    env._max_episode_steps = 1000
    # create our policy 
    policy = PolicyMLP(env.observation_space.shape[0], env.action_space.n)

    # visualize the policy network
    summary(policy, input_size= env.observation_space.shape)

    # initialize optimizer and loss function
    optimizer = torch.optim.Adam(policy.parameters(), lr = learning_rate)
    
    # normalized return : G - b where b is the average return. this will help reduce variance. 
    def computeNormalizedReturn(rewards, gamma = 0.99):
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + gamma * discounted_sum

            returns.insert(0, discounted_sum)

        returns = np.array(returns, np.float32)

        return (returns - np.mean(returns)) / (np.std(returns) + 1e-9)


    # pretraining phase (using expert data)
    print("Pre-Training using expert data...")
    epochs = 20
    for e in range(epochs):
        print("Pre-training epoch: {}".format(e))
        for i in range(expert_buffer.buffer_size):
            state = expert_buffer.state0_buffer[i]
            a = expert_buffer.action_buffer[i]
            G = 2

            # compute loss (expert policy gradient)
            loss = - (F.log_softmax(policy(state), dim = 0)[a] * G)

            # update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save model
    torch.save(policy.state_dict(), "rl-from-scratch/reinforce_weights_pretrain_{}.pth".format(environment))

    episode = 1
    while False:
        
        # episode trajectory of st, at, rt...
        states = []
        actions = []
        rewards = []
        episode_loss = []
        state0, _ = env.reset()
        state0 = torch.tensor(state0)

        done = False
        terminate = False
        # run through an episode
        while not (done or terminate):

            # sample action from the policy
            logits = policy(state0)
            probs = F.softmax(logits)
            a = torch.multinomial(probs, num_samples=1).item()

            # execute action
            state1, reward, done, terminate, info = env.step(a)

            # reward engineering. add velocity to reward function to encourage learning
            #reward += (np.abs(state1[1]) * 100)
            #height = np.sin(3* state0[0]) *.45 *.55
            #reward += height

            # append sample to trajectory
            states.append(state0)
            actions.append(a)        
            rewards.append(reward)
            
            state0 = torch.tensor(state1)

        # Train model using trajectory experience
        # compute discounted return
        returns = computeNormalizedReturn(rewards) # this includes subtraction of baseline from return G - b

        episode_loss = []
        episode_reward = sum(rewards)
        loss = 0
        for t in range(len(states)):  

            # compute loss
            state = states[t]      
            a = actions[t]
            G = returns[t]

            

            # compute entropy
            probs = F.softmax(policy(state), dim = 0)
            logprobs = F.log_softmax(policy(state), dim = 0) 
            
            H = - (probs * logprobs).sum(dim = -1).mean()
            
            #. Poicy Gradient loss:  log(pi(s|a)) * G + entropy
            loss += -(F.log_softmax(policy(state), dim=0)[a] * G) - (beta * H)
            
            episode_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            

        # print training metrics
        if (episode % 30) == 0:
            print("Episode: {}, Episode Reward: {}, Episode Loss: {}".format(episode, episode_reward, sum(episode_loss)))
            
            # save model
            torch.save(policy.state_dict(), "reinforce_weights_{}_b.pth".format(environment))
        
        episode += 1

env = gym.make(environment, render_mode = "human")

policy = PolicyMLP(env.observation_space.shape[0], env.action_space.n)

policy.load_state_dict(torch.load("rl-from-scratch/reinforce_weights_pretrain_{}.pth".format(environment)))

# rollout trained policy
while True:
    state0, _ = env.reset()
    state0 = torch.tensor(state0)
    done = False
    terminate = False
    while not (done or terminate):

        #a = torch.multinomial(F.softmax(policy(state0), dim=0), num_samples=1).item()
        a = np.argmax(policy(state0).detach().numpy())
        state1, reward, done, terminate, info = env.step(a)
        state0 = torch.tensor(state1)



    





