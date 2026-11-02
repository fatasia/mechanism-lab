"""冻结的 ResNet18 多尺度表征和有界正常 patch 记忆库。"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(incoming, outgoing, 1, stride, bias=False),
                nn.BatchNorm2d(outgoing),
            )
            if stride != 1 or incoming != outgoing
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + residual)


class PatchEncoder(nn.Module):
    """标准 ResNet18 到 layer3；权重键与原发布方一致，不依赖 torchvision。"""

    def __init__(self, projection_dimensions=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, 2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256))
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )
        projection = np.random.default_rng(42).normal(
            size=(384, projection_dimensions)
        ).astype("float32") / np.sqrt(projection_dimensions)
        self.register_buffer(
            "projection", torch.from_numpy(projection.astype("float32"))
        )

    def load_pretrained(self, path):
        weights = torch.load(path, map_location="cpu", weights_only=True)
        own = self.state_dict()
        # 仅加载明确的三个残差 stage，随机投影和归一化由本实验固定。
        # 发布的早期 checkpoint 不含 BN 训练批数计数器；冻结推理不使用该计数。
        generated = {"projection", "mean", "std"} | {
            k for k in own if k.endswith("num_batches_tracked")
        }
        selected = {k: weights[k] for k in own if k not in generated}
        selected.update({k: own[k] for k in generated})
        self.load_state_dict(selected, strict=True)

    def forward(self, image):
        x = (image - self.mean) / self.std
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
        x = self.layer1(x)
        a = self.layer2(x)
        b = self.layer3(a)
        a = F.avg_pool2d(F.avg_pool2d(a, 3, 1, 1), 2)
        b = F.avg_pool2d(b, 3, 1, 1)
        return torch.cat([a, b], 1).permute(0, 2, 3, 1) @ self.projection


def kcenter_memory(patches, size=256):
    """固定种子候选池上的贪心 farthest-first，内存与候选池大小有界。"""
    patches = np.asarray(patches, dtype="float32")
    minimum = np.full(len(patches), np.inf)
    selected, index = [], 0
    for _ in range(min(size, len(patches))):
        selected.append(index)
        minimum = np.minimum(minimum, np.square(patches - patches[index]).sum(1))
        index = int(minimum.argmax())
    return patches[selected]


class PatchMemoryScore(nn.Module):
    def __init__(self, encoder, memory):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("memory", torch.as_tensor(memory))

    def forward(self, image):
        features = self.encoder(image)
        distance = (
            features.square().sum(-1, keepdim=True)
            + self.memory.square().sum(-1)
            - 2 * (features @ self.memory.T)
        )
        heatmap = distance.clamp_min(0).amin(-1).sqrt()
        return heatmap.flatten(1).amax(1), heatmap
