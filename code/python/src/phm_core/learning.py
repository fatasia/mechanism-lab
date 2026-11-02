"""健康样本重构训练；模型选择不读取测试集。"""

from copy import deepcopy

import numpy as np
import torch

from phm_core.deep import ConvAutoencoder, DenseAutoencoder


def fit_autoencoder(
    values: np.ndarray,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    architecture: str = "conv",
    seed: int = 42,
    epochs: int = 80,
    scale_floor: float | None = None,
):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    mean = (
        values[train_ids].mean(axis=tuple(range(values.ndim - 1)))
        if values.ndim == 3
        else values[train_ids].mean(axis=0)
    )
    scale = (
        values[train_ids].std(axis=tuple(range(values.ndim - 1)))
        if values.ndim == 3
        else values[train_ids].std(axis=0)
    )
    if scale_floor is not None:
        scale = np.maximum(scale, scale_floor)
    if np.any(scale == 0):
        raise ValueError("训练输入存在常量维度，应在特征配置中移除或指定测量尺度")
    z = torch.from_numpy(((values - mean) / scale).astype("float32"))
    model = (
        ConvAutoencoder(values.shape[-1])
        if architecture == "conv"
        else DenseAutoencoder(values.shape[1:])
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    best, stale, weights, history = float("inf"), 0, None, []
    for epoch in range(epochs):
        model.train()
        losses = []
        training = z[train_ids]
        for batch in training[torch.randperm(len(training))].split(32):
            optimizer.zero_grad()
            loss = torch.nn.functional.huber_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            validation = torch.nn.functional.huber_loss(
                model(z[validation_ids]), z[validation_ids]
            ).item()
        history.append(
            {
                "epoch": epoch + 1,
                "train": float(np.mean(losses)),
                "validation": validation,
            }
        )
        if validation < best:
            best, stale, weights = validation, 0, deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        reconstructed = np.concatenate([model(batch).numpy() for batch in z.split(64)])
    score = np.abs(reconstructed - z.numpy()).reshape(len(values), -1).mean(axis=1)
    return model, mean, scale, score, reconstructed * scale + mean, history
