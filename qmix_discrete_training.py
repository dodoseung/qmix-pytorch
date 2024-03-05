import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from qmix import MixingNet, AgentNet
import copy
import numpy as np

class ReplayBuffer():
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        self.memory = []
        
    # Add the replay memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sample the replay memory
    def sample(self):
        batch = self.memory
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    # Reset the replay memory
    def reset(self):
        self.memory = []


class QMIX():
    def __init__(self, env, memory_size=1000000, batch_size=32, target_update=100, hidden_dim=64, rnn_dim=64, eps=1, gamma=0.99, learning_rate=5e-4):
        super(QMIX, self).__init__()
        self.env = env
        self.agent_num = self.env.agent_num
        self.state_num = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n
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

    def training(self):
        # Replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = [torch.FloatTensor(s).to(self.device) for s in states]
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = [torch.FloatTensor(ns).to(self.device) for ns in next_states]
        dones = torch.FloatTensor(dones).to(self.device)

