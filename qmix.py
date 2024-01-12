import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np


class MixingNet(nn.Module):
    def __init__(self, agent_num, state_num, hidden_dim):
        super(MixingNet, self).__init__()
        self.agent_num = agent_num
        self.state_num = state_num
        self.hidden_dim = hidden_dim

        self.hyper_w1 = nn.Linear(self.state_num, self.hidden_dim)
        self.hyper_w2 = nn.Linear(self.state_num, self.hidden_dim)
        self.hyper_w3 = nn.Linear(self.state_num, self.hidden_dim)
        self.hyper_w4_1 = nn.Linear(self.state_num, self.hidden_dim)
        self.hyper_w4_2 = nn.Linear(self.state_num, self.hidden_dim)

        return 0
    
    def forward(self, state, q_values):
        return 0
    
class AgentNet(nn.Module):
    def __init__(self, obs_num, action_num, rnn_dim):
        super(AgentNet, self).__init__()
        self.obs_num = obs_num
        self.action_num = action_num
        self.rnn_dim = rnn_dim

        self.fc1 = nn.Linear(self.obs_num, self.rnn_dim)
        self.gru = nn.GRUCell(self.rnn_dim, self.rnn_dim)
        self.fc2 = nn.Linear(self.rnn_dim, self.action_num)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.rnn_dim).zero_()
    
    def forward(self, observation, last_action, hidden_state):
        x = torch.cat([observation, last_action], 1)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.rnn_dim)
        h = self.gru(x, h_in)
        q = self.fc2(h)

        return q, h
