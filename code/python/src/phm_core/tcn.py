"""有限的因果预测实验，不将预测误差换算为剩余寿命。"""

import torch
from torch import nn


class CausalConv(nn.Module):
    def __init__(self, source: int, target: int, dilation: int):
        super().__init__()
        self.left = 2 * dilation
        self.layer = nn.Conv1d(source, target, 3, dilation=dilation)

    def forward(self, x):
        return self.layer(torch.nn.functional.pad(x, (self.left, 0)))


class TCN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv(channels, 16, 1),
            nn.ReLU(),
            CausalConv(16, 16, 2),
            nn.ReLU(),
            CausalConv(16, channels, 4),
        )

    def forward(self, x):
        return self.layers(x.transpose(1, 2)).transpose(1, 2)
