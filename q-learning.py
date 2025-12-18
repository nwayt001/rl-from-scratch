# Implements the Q-Learning Algorithm for solving mountaincar. 
import gymnasium as gym
import random, pygame
import torch
from torch import nn
from torchinfo import summary
import numpy as np
from collections import deque
from pathlib import Path
import pickle


# environment
mountainCar = "MountainCar-v0"
cartPole = "CartPole-v1"
environment = mountainCar

# training parameters
num_episodes = 100000
num_timesteps = 10000
T = 200
epsilon = 1
start_epsilon = 1
end_epsilon = 0.01
steps = 0 
gamma = 0.99
update_freq = 500
episode_loss = 0
batch_size = 32
end_training = False
num_hidden_units = 64
learning_rate = 0.001
buffer_length = 10000
warm_start = 100
human_demo = False

# create a simple neural network for the q-learner
class Q_network_v1(nn.Module):
    def __init__(self, input_size, num_actions, num_hidden):
        super(Q_network_v1, self).__init__()
        
        # input layer
        self.fc1 = nn.Linear(input_size, num_hidden)
        # hidden layer
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        # output layer
        self.fc3 = nn.Linear(num_hidden, num_actions)

    # forward pass
    def forward(self, x):
        x = self.fc1(x) # first layer
        x = nn.functional.tanh(x) # non-linear activation
        x = self.fc2(x) # second layer
        x = nn.functional.tanh(x) # non-linear activation
        x = self.fc3(x) # second layer
        return x


class Q_network(nn.Module):
    def __init__(self, input_size, num_actions, num_hidden):
        super(Q_network, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, num_hidden),
            #nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            #nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_actions)
        )
    
    def forward(self, x):
        return self.model(x)

# Create Replay buffer consisting of s0, r, a, s1
state0_buffer = deque(maxlen = buffer_length)
state1_buffer = deque(maxlen = buffer_length)
action_buffer = deque(maxlen = buffer_length)
reward_buffer = deque(maxlen = buffer_length)
done_buffer = deque(maxlen = buffer_length)

## Seed replay buffer with human demonstrations
env = gym.make(environment, render_mode = "human")
pygame.init()
state0, _ = env.reset()
state0 = torch.tensor(state0)

# Warm start with either human demonstrations on random actions
for i in range(warm_start):
    keys = pygame.key.get_pressed()
    if human_demo:
        # get action from human
        a = 1
        if keys[pygame.K_a]:
            a = 0
        elif keys[pygame.K_d]:
            if environment == cartPole:
                a = 1
            elif environment == mountainCar:
                a = 2
    else:
        # random policy
        a = random.randint(0, env.action_space.n - 1) 
    
    if keys[pygame.K_q]:
        break

    # env step
    state1, reward, done, _, _ = env.step(a)
    state1 = torch.tensor(state1)

    # save human experience to replay buffer
    state0_buffer.append(state0)
    state1_buffer.append(state1)
    action_buffer.append(a)
    reward_buffer.append(reward)
    done_buffer.append(int(done))

    state0 = state1.detach().clone()

    if done:
        state0, _ = env.reset()
        state0 = torch.tensor(state0)

env.close()

# linearly decay epsilon for eps-greedy exploration
def decaying_epsilon(current_step: int, start_eps: float, end_eps: float, total_steps: int) -> float:
    decay_ammounnt = (start_eps - end_eps) / total_steps
    new_eps = start_eps - (decay_ammounnt * current_step)
    
    return max(new_eps, end_eps)


# initialize mountaincar env
env = gym.make(environment)
pygame.init() # for keypress

# initialize q-network
q_network = Q_network(env.observation_space.shape[0], env.action_space.n, num_hidden_units)
q_target_network = Q_network(env.observation_space.shape[0], env.action_space.n, num_hidden_units)

# visualize network architechture. 
summary(q_network, input_size=env.observation_space.shape)

# init optimizer and loss function
optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate) # use default learning rate
loss_fn = nn.MSELoss()


for episode in range(num_episodes):
    # reset env
    state0, _= env.reset()
    state0 = torch.tensor(state0)
    episode_loss = 0
    episode_reward = 0
    episode_step = 0
    for t in range(T):
        # chose action according to epsilon greedy policy
        epsilon = decaying_epsilon(steps, start_epsilon, end_epsilon, num_timesteps)
        if (random.random() < epsilon):
            a = random.randint(0, env.action_space.n - 2)  # take random action
        else:
            q_network.eval()
            a = q_network(state0.unsqueeze(0)).detach().numpy().argmax() # choose action using our q-network

        # execute action in environment
        state1, reward, done, truncated, info= env.step(a)

        # store the experience tuple in a replay buffer
        state1 = torch.tensor(state1)
        state0_buffer.append(state0)
        state1_buffer.append(state1)
        action_buffer.append(a)
        reward_buffer.append(reward)
        done_buffer.append(int(done))
         
        # Sample minibatch
        idxs = random.sample(range(len(state0_buffer)), batch_size)
        s0_batch = [state0_buffer[i] for i in idxs]
        s1_batch = [state1_buffer[i] for i in idxs]
        a_batch = [action_buffer[i] for i in idxs]
        r_batch = [reward_buffer[i] for i in idxs]
        done_batch = [done_buffer[i] for i in idxs]

        s0_batch = torch.stack(s0_batch, dim=0)
        a_batch = torch.tensor(a_batch).unsqueeze(dim=1)
        s1_batch = torch.stack(s1_batch, dim=0)
        r_batch = torch.tensor(r_batch).unsqueeze(dim=1)
        done_batch = torch.tensor(done_batch)

        # compute loss, perform a gradient update (backwards step)
        # target q-values
        with torch.no_grad():
            # get max q-value of next step
            next_q_vals = q_target_network(s1_batch).max(dim=1, keepdim=True)[0]

            y_target = r_batch + (gamma * next_q_vals * (1 - done_batch))

        #predicted q-values
        q_network.train()
        q_vals = torch.gather(q_network(s0_batch), dim=1, index=a_batch)
        
        # compute loss
        loss = loss_fn(q_vals, y_target)

        # optimize 
        optimizer.zero_grad() # Zero out gradients
        loss.backward() # compute gradient
        optimizer.step() # take a gradient step

        steps = steps + 1 # increment our global step counter
        episode_step += 1 # increment our episode step counter
        episode_loss += loss.detach().numpy()
        episode_reward += reward
        # update target q network
        if (steps % update_freq == 0):
            q_target_network.load_state_dict(q_network.state_dict())

        # make the new state be the old state
        state0 = state1.detach().clone()

        # check for user to quit and save weights.
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            torch.save(q_network.state_dict(), "q-network_weights.pth")
            end_training = True
            break
        if done:
            break
    
    if end_training:
        break
    # print training metrics
    if (episode % 50 == 0):
        avg_loss = (episode_loss / episode_step)
        print("Episode {}, Reward: {}, Loss: {}, epsilon: {}, total timesteps: {}".format(episode, episode_reward, avg_loss, epsilon, steps))
    

# load the network weights and visualize rollout
env.close()

env = gym.make(environment, render_mode = "human")

state0, _ = env.reset()
state0 = torch.tensor(state0)
for t in range(T):
    a = q_network(state0).detach().numpy().argmax()

    # execute action in environment
    state1, reward, done, truncated, info= env.step(a)

    state0 = torch.tensor(state1)

    
    

