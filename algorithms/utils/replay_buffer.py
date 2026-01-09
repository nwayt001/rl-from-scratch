import numpy as np
from collections import deque
import random

# Replay Buffer
class Replay_Buffer():
    def __init__(self, max_buffer_size: int):
        self.max_buffer_size = max_buffer_size

        # current size of replay buffer
        self.buffer_size = 0

        #init deque buffers
        self.state0_buffer = deque(maxlen = max_buffer_size)
        self.state1_buffer = deque(maxlen = max_buffer_size)
        self.action_buffer = deque(maxlen = max_buffer_size)
        self.reward_buffer = deque(maxlen = max_buffer_size)
        self.done_buffer = deque(maxlen = max_buffer_size)
        

    # add experience to buffer
    def append(self, state0, state1, a, reward, done):
        # add to deque object
        self.state0_buffer.append(state0)
        self.state1_buffer.append(state1)
        self.action_buffer.append(a)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

        # keep track of number of samples in the buffer
        if self.buffer_size < self.max_buffer_size:
            self.buffer_size += 1
        else:
            self.buffer_size = self.max_buffer_size

    # sample a batch of experience
    def sample(self, batch_size = 16):

        idxs = random.sample(range(self.buffer_size), batch_size)

        s0_batch = [self.state0_buffer[i] for i in idxs]
        s1_batch = [self.state1_buffer[i] for i in idxs]
        a_batch = [self.action_buffer[i] for i in idxs]
        r_batch = [self.reward_buffer[i] for i in idxs]
        done_batch = [self.done_buffer[i] for i in idxs]
    
        return s0_batch, s1_batch, a_batch, r_batch, done_batch