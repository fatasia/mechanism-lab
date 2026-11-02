"""可导出的多尺度诊断网络与 ROCKET 思路的固定随机卷积对照。"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn


class Normalized(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def normalized(self, values):
        return ((values - self.mean) / self.scale).transpose(1, 2)


class KernelTransform(Normalized):
    """固定种子的随机多尺度/膨胀卷积，PPV 与最大值池化；不是原版 ROCKET 复现。"""

    def __init__(self, mean, scale, seed=42):
        super().__init__(mean, scale)
        generator = torch.Generator().manual_seed(seed)
        self.filters = nn.ModuleList()
        for kernel in (3, 7, 11):
            for dilation in (1, 2, 4, 8):
                layer = nn.Conv1d(
                    3,
                    16,
                    kernel,
                    dilation=dilation,
                    padding=dilation * (kernel - 1) // 2,
                )
                weight = torch.randn(layer.weight.shape, generator=generator)
                layer.weight.data.copy_(weight - weight.mean((1, 2), keepdim=True))
                layer.bias.data.copy_(torch.rand(16, generator=generator) * 2 - 1)
                layer.requires_grad_(False)
                self.filters.append(layer)

    def forward(self, values):
        values = nn.functional.avg_pool1d(self.normalized(values), 2)
        features = []
        for layer in self.filters:
            response = layer(values)
            features.extend([(response > 0).float().mean(2), response.amax(2)])
        return torch.cat(features, dim=1)


class KernelDiagnostic(nn.Module):
    def __init__(self, transform, center, scale, coef, intercept):
        super().__init__()
        self.transform = transform
        for name, value in {
            "center": center,
            "scale": scale,
            "coef": coef,
            "intercept": intercept,
        }.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def forward(self, values):
        z = (self.transform(values) - self.center) / self.scale
        return z @ self.coef.T + self.intercept


class MultiScaleBlock(nn.Module):
    def __init__(self, inputs, dilation):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    inputs,
                    12,
                    kernel,
                    dilation=dilation,
                    padding=dilation * (kernel - 1) // 2,
                )
                for kernel in (3, 9, 21)
            ]
        )
        self.pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(inputs, 12, 1)
        )
        self.norm = nn.BatchNorm1d(48)
        self.skip = nn.Conv1d(inputs, 48, 1) if inputs != 48 else nn.Identity()

    def forward(self, x):
        z = torch.cat([branch(x) for branch in self.branches] + [self.pool(x)], dim=1)
        return torch.relu(self.norm(z) + self.skip(x))


class MultiScaleDiagnostic(Normalized):
    """多尺度残差时序编码与误差/电流幅值支路，输出四类未校准分数。"""

    def __init__(self, mean, scale):
        super().__init__(mean, scale)
        self.encoder = nn.Sequential(
            MultiScaleBlock(3, 1), MultiScaleBlock(48, 2), MultiScaleBlock(48, 4)
        )
        self.head = nn.Sequential(
            nn.Linear(102, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 4)
        )

    def forward(self, values):
        x = self.normalized(values)
        z = self.encoder(nn.functional.avg_pool1d(x, 2))
        amplitude = torch.cat([x.square().mean(2).sqrt(), x.abs().amax(2)], dim=1)
        return self.head(torch.cat([z.mean(2), z.amax(2), amplitude], dim=1))


def infer_batches(model, values):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                model(torch.from_numpy(batch)).numpy()
                for batch in np.array_split(values, max(1, (len(values) + 63) // 64))
            ]
        )


def fit_multiscale(values, labels, train, validation, seed=42, epochs=60):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    mean, scale = values[train].mean((0, 1)), values[train].std((0, 1))
    if np.any(scale == 0):
        raise ValueError("训练通道不能为常量")
    model = MultiScaleDiagnostic(mean, scale)
    weights = len(train) / (4 * np.bincount(labels[train], minlength=4))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    x, y = torch.from_numpy(values), torch.from_numpy(labels).long()
    best, stale, checkpoint, history = float("inf"), 0, None, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for ids in torch.tensor(train)[torch.randperm(len(train))].split(32):
            optimizer.zero_grad()
            loss = criterion(model(x[ids]), y[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            valid = criterion(model(x[validation]), y[validation]).item()
        history.append(
            {"epoch": epoch + 1, "train": float(np.mean(losses)), "validation": valid}
        )
        if valid < best:
            best, stale, checkpoint = valid, 0, deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(checkpoint)
    return model.eval(), history
