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
from replay_buffer import Replay_Buffer
import torch.nn.functional as F

# environment
mountainCar = "MountainCar-v0"
mountainCarHillTop = "MountainCarHillTop-v0"
cartPole = "CartPole-v1"
environment = cartPole

# training parameters
max_timesteps = 6000000
num_timesteps_eps = 100000
epsilon = 1
start_epsilon = 1
end_epsilon = 0.05
steps = 0 
gamma = 0.99
update_freq = 500
episode_loss = 0
batch_size = 128
end_training = False
learning_rate = 0.00025
buffer_length = 10000
warm_start = 1000
human_demo = False
boltzman_sampling = False
checkpoint_interval = 50000
train = False
train_interval = 20
use_expert_demo = False
expert_demo_duration = 200000

# cuda availablity
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

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
    def __init__(self, input_size, num_actions):
        super(Q_network, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_actions)
        )
    
    def forward(self, x):
        return self.model(x)


# Load expert replay buffer
if use_expert_demo:
    with open("rl-from-scratch/expert_data_{}".format(environment), 'rb') as file:
        expert_buffer = pickle.load(file)

# Initialize replay buffer
replay_buffer = Replay_Buffer(buffer_length)

## Seed replay buffer 
env = gym.make(environment)
pygame.init()
state0, _ = env.reset()
state0 = torch.tensor(state0).to(device)

# Warm start with either human demonstrations on random actions
if train:
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
        state1 = torch.tensor(state1).to(device)

        # save human experience to replay buffer
        replay_buffer.append(state0, state1, a, reward, int(done))

        state0 = state1.detach().clone().to(device)

        if done:
            state0, _ = env.reset()
            state0 = torch.tensor(state0).to(device)

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
    q_network = Q_network(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target_network = Q_network(env.observation_space.shape[0], env.action_space.n).to(device)

    # visualize network architechture. 
    summary(q_network, input_size=env.observation_space.shape)

    # init optimizer and loss function
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate) # use default learning rate
    loss_fn = nn.MSELoss()

    # reset env
    state0, _= env.reset()
    state0 = torch.tensor(state0).to(device)
    episode_loss = 0
    episode_reward = 0
    episode_step = 0
    episode = 0
    for step in range(max_timesteps):
        
        # use boltzman sampling
        if boltzman_sampling:
            probs = F.softmax(q_network(state0.unsqueeze(0)), dim=1)
            a = torch.multinomial(probs, num_samples=1).item()

        else: # chose action according to epsilon greedy policy
            epsilon = decaying_epsilon(step, start_epsilon, end_epsilon, num_timesteps_eps)
            if (random.random() < epsilon):
                #a = random.randint(0, env.action_space.n - 2)  # take random action
                probs = F.softmax(q_network(state0.unsqueeze(0)), dim=1)
                a = torch.multinomial(probs, num_samples=1).item()
            else:
                q_network.eval()
                a = q_network(state0.unsqueeze(0)).detach().cpu().numpy().argmax() # choose action using our q-network

        # execute action in environment
        state1, reward, done, truncated, info= env.step(a)
        state1 = torch.tensor(state1).to(device)

        # store the experience tuple in a replay buffer
        replay_buffer.append(state0, state1, a, reward, int(done))
        
        # train weights
        if (step % train_interval) == 0:

            # Sample minibatch from agent replay buffer
            s0_batch, s1_batch, a_batch, r_batch, done_batch = replay_buffer.sample(batch_size)
            
            if use_expert_demo:
                # sample minibatch from expert replay buffer
                s0_e, s1_e, a_e, r_e, done_e = expert_buffer.sample(batch_size)

                # Combine agent and expert batches
                s0_batch.extend(s0_e)
                s1_batch.extend(s1_e)
                a_batch.extend(a_e)
                r_batch.extend(r_e)
                done_batch.extend(done_e)

            # reshape to stacked tensor
            s0_batch = torch.stack(s0_batch, dim=0).to(device)
            a_batch = torch.tensor(a_batch).unsqueeze(dim=1).to(device)
            s1_batch = torch.stack(s1_batch, dim=0).to(device)
            r_batch = torch.tensor(r_batch).unsqueeze(dim=1).to(device)
            done_batch = torch.tensor(done_batch).to(device)

            # compute loss, perform a gradient update (backwards step)
            # target q-values
            with torch.no_grad():
                # get max q-value of next step
                next_q_vals = q_target_network(s1_batch).max(dim=1, keepdim=True)[0]

                y_target = r_batch + (gamma * next_q_vals * (1 - done_batch.unsqueeze(1)))

            #predicted q-values
            q_network.train()
            q_vals = torch.gather(q_network(s0_batch), dim=1, index=a_batch)
            
            # compute loss
            loss = loss_fn(q_vals, y_target)

            # optimize 
            optimizer.zero_grad() # Zero out gradients
            loss.backward() # compute gradient
            optimizer.step() # take a gradient step
            
            episode_step += 1 # increment our episode step counter
            episode_loss += loss.item()
        episode_reward += reward

        # update target q network
        if (steps % update_freq == 0):
            q_target_network.load_state_dict(q_network.state_dict())

        # make the new state be the old state
        state0 = state1.detach().clone().to(device)
        
        # Save model weights checkpoint
        if step % checkpoint_interval == 0:
            print("saving checkpoint...")
            torch.save(q_network.state_dict(), "dqn_weights_{}.pth".format(environment))

        
        # handle when episode ends/terminates
        if done or truncated:
            episode = episode + 1
            
            # print training metrics
            if (episode % 50 == 0):
                avg_loss = (episode_loss)
                print("Episode {}, Reward: {}, Loss: {}, epsilon: {}, total timesteps: {}".format(episode, episode_reward, avg_loss, epsilon, step))
            

            episode_loss = 0
            episode_reward = 0
            episode_step = 0

            # reset env
            state0, _= env.reset()
            state0 = torch.tensor(state0).to(device)

            

# Rollout trained model

# init environment
env = gym.make(environment, render_mode = "human")

# init model
q_network = Q_network(env.observation_space.shape[0], env.action_space.n).to(device)

# load weights
q_network.load_state_dict(torch.load("dqn_weights_{}.pth".format(environment)))

# rollout
state0, _ = env.reset()
state0 = torch.tensor(state0)
while True:
    # select action
    a = q_network(state0).detach().numpy().argmax()

    # execute action in environment
    state1, reward, done, truncated, info= env.step(a)
    state0 = torch.tensor(state1)

    if done:
        state0, _ = env.reset()
        state0 = torch.tensor(state0)
        

    
    

