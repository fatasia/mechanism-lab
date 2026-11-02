"""明确工况的健康参考变换；不从测试波形反推工况或自动吸收异常。"""

import numpy as np


def fit_reference(values, contexts, healthy, reference_ids):
    """尺度用已批准参考内离散度；小样本工况共享尺度下限。"""
    ids = np.asarray(reference_ids, dtype=int)
    contexts = np.asarray(contexts)
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("健康参考须非空且不重复")
    if np.any(ids < 0) or np.any(ids >= len(values)) or not np.all(healthy[ids]):
        raise ValueError("参考只允许明确健康的有效记录")
    centers = {
        c: np.median(values[ids[contexts[ids] == c]], axis=0)
        for c in np.unique(contexts[ids])
    }
    residual = np.stack([values[i] - centers[contexts[i]] for i in ids])
    # 参考内波动共享，而非不同负载间幅值变化；下限仅数值尺度，不称传感器精度。
    pooled = np.maximum(1.4826 * np.median(np.abs(residual), axis=0), 1e-5)
    return {
        c: {
            "center": center.tolist(),
            "scale": pooled.tolist(),
            "reference_ids": ids[contexts[ids] == c].tolist(),
        }
        for c, center in centers.items()
    }


def transform_reference(values, contexts, reference):
    """未知工况直接拒绝，不返回低异常分数或假正常。"""
    if len(values) != len(contexts):
        raise ValueError("工况与记录数量不一致")
    missing = sorted(set(contexts) - reference.keys())
    if missing:
        raise ValueError(f"未投运工况: {missing}")
    center = np.asarray([reference[c]["center"] for c in contexts])
    scale = np.asarray([reference[c]["scale"] for c in contexts])
    return ((values - center) / scale).astype("float32")
