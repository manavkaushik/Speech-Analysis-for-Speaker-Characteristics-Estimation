# Soft Attention

import torch
from torch import nn

class Attention(nn.Module):

    def __init__(self, lambda_weight=0.8, n_channels=16):
        super(Attention, self).__init__()
        self.lambda_weight = torch.tensor(lambda_weight, dtype=torch.float32, requires_grad=False)
        self.linear1 = nn.Linear(in_features=n_channels, out_features=n_channels)
        self.linear2 = nn.Linear(in_features=n_channels, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        v = self.tanh(self.linear1(x))
        v = self.linear2(v)
       
        v = v.squeeze(2)
        
        v = v*self.lambda_weight
        v = self.softmax(v)
        v = v.unsqueeze(-1)
        output = (x*v).sum(axis=1)
        
        return output