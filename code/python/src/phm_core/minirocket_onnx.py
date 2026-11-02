"""将已拟合 MiniRocket 数值参数映射到 ONNX 基础算子，不依赖 aeon 推理。"""

from itertools import combinations

import numpy as np
import torch
from torch import nn


def shifted(values, offset):
    if offset < 0:
        return nn.functional.pad(values[:, :, :offset], (-offset, 0))
    if offset > 0:
        return nn.functional.pad(values[:, :, offset:], (0, offset))
    return values


class DilatedPPV(nn.Module):
    """公开固定 84 核及交替有效区间；保留逐通道运算顺序以校验阈值交点。"""

    def __init__(self, dilation, biases, channels, counts, parity):
        super().__init__()
        self.dilation = int(dilation)
        self.padding = 4 * self.dilation
        self.register_buffer(
            "triples", torch.tensor(list(combinations(range(9), 3)), dtype=torch.long)
        )
        self.register_buffer("biases", torch.tensor(biases, dtype=torch.float32))
        padded = np.zeros((84, 3), dtype=np.int64)
        active = np.zeros((84, 3), dtype=np.float32)
        start = 0
        for i, count in enumerate(counts):
            padded[i, :count] = channels[start : start + count]
            active[i, :count] = 1
            start += count
        self.register_buffer("channels", torch.from_numpy(padded))
        self.register_buffer("active", torch.from_numpy(active))
        self.register_buffer(
            "full_padding", ((torch.arange(84) + parity) % 2 == 0)[None, :, None]
        )

    def forward(self, x):
        negative, triple = -x, 3 * x
        alpha = negative
        for offset in (-4, -3, -2, -1, 1, 2, 3, 4):
            alpha = alpha + shifted(negative, offset * self.dilation)
        gamma = torch.stack(
            [shifted(triple, offset * self.dilation) for offset in range(-4, 5)], dim=1
        )
        response = alpha[:, None] + gamma[:, self.triples[:, 0]]
        response = response + gamma[:, self.triples[:, 1]]
        response = response + gamma[:, self.triples[:, 2]]
        selected = torch.gather(
            response,
            2,
            self.channels[None, :, :, None].expand(x.shape[0], -1, -1, x.shape[2]),
        )
        summed = selected[:, :, 0] * self.active[None, :, 0, None]
        for i in (1, 2):
            summed = summed + selected[:, :, i] * self.active[None, :, i, None]
        full = (summed[:, :, None] > self.biases[None, :, :, None]).float().mean(-1)
        valid = (
            (
                summed[:, :, None, self.padding : -self.padding]
                > self.biases[None, :, :, None]
            )
            .float()
            .mean(-1)
        )
        return torch.where(self.full_padding, full, valid).flatten(1)


class MiniRocketONNX(nn.Module):
    """输入原始 BTC 窗口；缩放、固定卷积/PPV、特征缩放和岭分类全部在图内。"""

    def __init__(self, parameters):
        super().__init__()
        for name in (
            "mean",
            "scale",
            "feature_mean",
            "feature_scale",
            "coef",
            "intercept",
        ):
            self.register_buffer(
                name, torch.tensor(parameters[name], dtype=torch.float32)
            )
        blocks, combo_start, channel_start, feature_start = [], 0, 0, 0
        for parity, (dilation, features) in enumerate(
            zip(parameters["dilations"], parameters["features_per_dilation"])
        ):
            counts = parameters["channel_counts"][combo_start : combo_start + 84]
            channels = parameters["channel_indices"][
                channel_start : channel_start + counts.sum()
            ]
            biases = parameters["biases"][
                feature_start : feature_start + 84 * features
            ].reshape(84, features)
            blocks.append(DilatedPPV(dilation, biases, channels, counts, parity))
            combo_start += 84
            channel_start += counts.sum()
            feature_start += 84 * features
        self.blocks = nn.ModuleList(blocks)

    def forward(self, values):
        x = ((values - self.mean) / self.scale).transpose(1, 2)
        features = torch.cat([block(x) for block in self.blocks], dim=1)
        return (
            (features - self.feature_mean) / self.feature_scale
        ) @ self.coef.T + self.intercept
