#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
from layers.gate_layer import GateLayer


# In[ ]:


class MLP1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP1, self).__init__()
        
        self.fc1   = nn.Linear(input_dim, 300, bias=False)
        self.gate1 = GateLayer(300, 300, [1, -1])
        self.fc2   = nn.Linear(300, 100, bias=False)
        self.gate2 = GateLayer(100, 100, [1, -1])

        self.out   = nn.Linear(100, output_dim, bias=False)

    def forward(self, x):
        out = x.view(-1, 54)
        out = F.relu(self.fc1(out))
        out = self.gate1(out)
        out = F.relu(self.fc2(out))
        out = self.gate2(out)
        out = self.out(out)
        
        return out


# In[ ]:


class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2, self).__init__()
        
        self.fc1   = nn.Linear(input_dim, 1024, bias=False)
        self.gate1 = GateLayer(1024, 1024,[1, -1])
        self.fc2   = nn.Linear(1024, 1024, bias=False)
        self.gate2 = GateLayer(1024, 1024,[1, -1])
        self.fc3   = nn.Linear(1024, 1024, bias=False)
        self.gate3 = GateLayer(1024, 1024,[1, -1])
        self.out   = nn.Linear(1024, output_dim, bias=False)

    def forward(self, x):
        out = x.view(-1, 54)
        out = F.relu(self.fc1(out))
        out = self.gate1(out)
        out = F.relu(self.fc2(out))
        out = self.gate2(out)
        out = F.relu(self.fc3(out))
        out = self.gate3(out)
        out = self.out(out)
        return out

