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

        # The weights and bias of the hypernetwork (non-negative) for the first layer
        self.hyper_w1 = nn.Linear(self.state_num, self.agent_num * self.hidden_dim)
        self.hyper_b1 = nn.Linear(self.state_num, self.hidden_dim)
        
        # The weights and bias of the hypernetwork (non-negative) for the first layer
        self.hyper_w2 = nn.Linear(self.state_num, self.hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_num, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 1))
    
    def forward(self, states, q_values):
        batch_size = states.shape(0)
        
        # Reshaping
        states = states.reshape(-1, self.state_num)
        q_values = q_values.reshape(-1, 1, self.agent_num)

        # First layer
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.reshape(-1, self.agent_num, self.hidden_dim)
        b1 = self.hyper_b1(states)
        b1 = b1.reshape(-1, 1, self.hidden_dim)
        x = F.elu(torch.bmm(q_values, w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.reshape(-1, self.hidden_dim, 1)
        b2 = self.hyper_b2(states)
        b2 = b2.reshape(-1, 1, 1)
        y = torch.bmm(x, w2) + b2

        # Reshaping
        q_tot = y.reshape(batch_size, -1, 1)

        return q_tot
    
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
