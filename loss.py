#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn



class DOICPLoss(nn.Module):
    def __init__(self, gamma):
        super(DOICPLoss, self).__init__()
        self.gamma = gamma

    def forward(self, A, q):
        sigma = torch.sigmoid(self.gamma * (A - q*torch.ones_like(A)))
        S = torch.sum(sigma, axis=1)
        #L = torch.mean(torch.relu(S), axis=0)
        L = torch.log(torch.mean(torch.relu(S), axis=0))
        #L = torch.log(torch.mean(torch.clamp(S, min=0), axis=0))
        #L = torch.mean((S-1)**2, axis=0)
        #L = torch.mean(torch.abs(S-1), axis=0)
        #L = torch.mean(torch.abs(torch.log(S+1)-torch.log(torch.Tensor([2]))), axis=0)
        #L = torch.mean((torch.log(S+1)-torch.log(torch.Tensor([2])))**2, axis=0)

        return L



