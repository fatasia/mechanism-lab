"""完整周期块切分；同试验台泛化，不宣称跨机器或未来预测。"""

import numpy as np


def block_split(count: int, seed: int = 42) -> dict[str, np.ndarray]:
    blocks = np.arange(count // 30)
    np.random.default_rng(seed).shuffle(blocks)
    edges = [
        0,
        int(len(blocks) * 0.55),
        int(len(blocks) * 0.7),
        int(len(blocks) * 0.85),
        len(blocks),
    ]
    return {
        name: np.sort(
            np.concatenate(
                [
                    np.arange(b * 30 + 3, b * 30 + 27)
                    for b in blocks[edges[i] : edges[i + 1]]
                ]
            )
        )
        for i, name in enumerate(["train", "validation", "calibration", "test"])
    }
