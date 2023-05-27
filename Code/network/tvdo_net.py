import torch.nn as nn
import torch


class TVDONet(nn.Module):
    def __init__(self):
        super(TVDONet, self).__init__()

    def forward(self, q_values):
        return torch.sum(q_values, dim=2, keepdim=True)

