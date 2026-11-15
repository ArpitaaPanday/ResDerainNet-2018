"""PyTorch architecture for the ResDerainNet 2018 reproduction."""
import torch
from torch import nn
import torch.nn.functional as F

class ResDerainNet(nn.Module):
    """20-convolution residual rain-estimation network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.convs = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=3, padding=1) for _ in range(18)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(64, eps=1e-8) for _ in range(18)
        ])
        self.conv20 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        return self.conv20(x)
