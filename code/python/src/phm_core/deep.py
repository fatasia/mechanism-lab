"""波形自编码器；训练集归一化参数由部署包装器固定。"""

import math

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    """全窗口压缩至小瓶颈，学习绝对时刻与跨通道关系。"""

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.shape = tuple(shape)
        size = math.prod(shape)
        hidden = 64 if size > 64 else 16
        latent = 8 if size > 64 else 2
        self.layers = nn.Sequential(
            nn.Linear(size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.Tanh(),
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values.flatten(1)).reshape(-1, *self.shape)


class ConvAutoencoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 8, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(8, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, channels, 5, padding=2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        reconstructed = self.decoder(self.encoder(values.transpose(1, 2)))
        return reconstructed[:, :, : values.shape[1]].transpose(1, 2)


class ReconstructionScore(nn.Module):
    """部署图输出 [batch,1]，高分表示更异常。"""

    def __init__(self, model: ConvAutoencoder):
        super().__init__()
        self.model = model

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (self.model(values) - values).abs().mean(dim=(1, 2)).unsqueeze(1)
