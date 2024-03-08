import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from qmix import MixingNet, AgentNet
import copy
import numpy as np
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, max_size=1000000):
        super(ReplayBuffer, self).__init__()
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)
        
    # Add the replay memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sample the replay memory
    def sample(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    # Reset the replay memory
    def reset(self):
        self.memory = []


class QMIX():
    def __init__(self, env, memory_size=1000000, batch_size=32, target_update=100, hidden_dim=64, rnn_dim=64, eps=1, gamma=0.99, learning_rate=5e-4):
        super(QMIX, self).__init__()
        self.env = env
        self.agent_num = 2# self.env.agent_num
        self.state_num = 20# self.env.observation_space.shape[0]
        self.action_num = 6# self.env.action_space.n
        self.target_update = target_update
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim

        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mixing network
        self.mixing_net = MixingNet(agent_num=self.agent_num, state_num=self.state_num, hidden_dim=self.hidden_dim).to(self.device)
        self.mixing_net_opt = optim.RMSprop(self.mixing_net.parameters(), lr=learning_rate)
        
        # Mixing target network 
        self.mixing_target_net = MixingNet(agent_num=self.agent_num, state_num=self.state_num, hidden_dim=self.hidden_dim).to(self.device)
        self.mixing_target_net.load_state_dict(self.mixing_net.state_dict())

        # Agent network and agent target network
        self.agent_net = []
        self.agent_net_opt = []
        self.agent_target_net = []

        for i in range(self.agent_num):
            # Agent network
            self.agent_net.append(AgentNet(obs_num=self.state_num, action_num=self.action_num, rnn_dim=self.rnn_dim).to(self.device))
            self.agent_net_opt.append(optim.RMSprop(self.agent_net[i].parameters(), lr=learning_rate))

            # Agent target network
            self.agent_target_net.append(AgentNet(obs_num=self.state_num, action_num=self.action_num, rnn_dim=self.rnn_dim).to(self.device))
            self.agent_target_net[i].load_state_dict(self.agent_net[i].state_dict())

        # Replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size

    # def training(self):
    #     # Replay buffer
    #     states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
    def training(self, states, actions, rewards, next_states, dones):    
        states = [torch.FloatTensor(s).to(self.device) for s in states]
        actions = [torch.FloatTensor(a).to(self.device) for a in actions]
        rewards = [torch.FloatTensor(r).to(self.device) for r in rewards]
        next_states = [torch.FloatTensor(ns).to(self.device) for ns in next_states]
        dones = [torch.FloatTensor(d).to(self.device) for d in dones]
        q_values = []

        for i in range(self.agent_num):
            hidden_states = self.agent_net[i].init_hidden().unsqueeze(0).expand(self.batch_size, self.agent_num, -1)
            q_value = self.agent_net[i](states[i], actions[i], hidden_states)
            q_values.append(q_value)


if __name__ == "__main__":
    states = torch.FloatTensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])
    actions = torch.FloatTensor([[[0], [1], [2]], [[3], [4], [5]]])
    rewards = torch.FloatTensor([[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]])
    next_states = torch.FloatTensor([[[11, 12, 13], [14, 15, 16], [17, 18, 19]], [[21, 22, 23], [24, 25, 26], [27, 28, 29]]])
    dones = torch.FloatTensor([[[False], [False], [False]], [[False], [False], [False]]])

    qmix = QMIX(env=5, batch_size=2)
    qmix.training(states, actions, rewards, next_states, dones)