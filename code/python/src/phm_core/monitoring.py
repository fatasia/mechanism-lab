"""固定健康参考的校准、同类群比较、漂移与在线趋势；均不自动吸收新样本。"""

from dataclasses import dataclass

import numpy as np


def calibrated_threshold(healthy_scores, alpha=0.01):
    """Split-conformal 上分位数；有效性仍依赖校准/部署样本可交换性。"""
    values = np.sort(np.asarray(healthy_scores, dtype=float))
    if not 0 < alpha < 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("需要有限健康校准分数及 0<alpha<1")
    rank = int(np.ceil((len(values) + 1) * (1 - alpha)))
    if rank > len(values):
        raise ValueError("校准样本不足以支持所选尾部水平")
    return float(values[rank - 1])


def tail_probability(scores, reference):
    reference = np.sort(np.asarray(reference))
    return (1 + len(reference) - np.searchsorted(reference, scores, side="left")) / (
        len(reference) + 1
    )


def compare_peers(values, peer_values, scale_floor, min_peers=5):
    """同工况、同规格、同单位的其它实体；调用方必须排除目标自身。"""
    peers = np.asarray(peer_values, dtype=float)
    if peers.ndim != 2 or len(peers) < min_peers:
        return {"status": "insufficient_peers", "peer_count": len(peers)}
    if np.asarray(values).shape != (peers.shape[1],):
        raise ValueError("目标和同类群的特征维度不一致")
    if not np.isfinite(peers).all() or not np.isfinite(values).all():
        raise ValueError("同类群比较需要有限数值")
    center = np.median(peers, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(peers - center), axis=0), scale_floor)
    deviation = (np.asarray(values) - center) / scale
    return {
        "status": "scored",
        "peer_count": len(peers),
        "center": center.tolist(),
        "deviation": deviation.tolist(),
        "score": float(np.abs(deviation).max()),
    }


def distribution_drift(reference, current, effect_floor=0.5, family_alpha=0.01):
    """KS + Bonferroni 与标准化均值差；用于复核触发，不是自动重训依据。"""
    from scipy.stats import ks_2samp

    reference, current = np.asarray(reference), np.asarray(current)
    if (
        reference.ndim != 2
        or current.ndim != 2
        or reference.shape[1] != current.shape[1]
        or min(len(reference), len(current)) < 20
    ):
        raise ValueError("漂移比较至少需要两组 20 个同契约特征向量")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("缺失数据先进入质量检查，不进入漂移计算")
    scale = np.maximum(reference.std(0), 1e-8)
    effects = np.abs(current.mean(0) - reference.mean(0)) / scale
    tests = [ks_2samp(a, b) for a, b in zip(reference.T, current.T)]
    return [
        {
            "feature": i,
            "ks": float(test.statistic),
            "p_value": float(test.pvalue),
            "effect": float(effect),
            "review": bool(
                test.pvalue < family_alpha / reference.shape[1]
                and effect >= effect_floor
            ),
        }
        for i, (test, effect) in enumerate(zip(tests, effects))
    ]


@dataclass
class TrendState:
    ewma: float = 0.0
    cusum: float = 0.0
    observations: int = 0


def trend_step(value, state, center, scale, weight=0.1, allowance=0.5):
    if scale <= 0 or not 0 < weight <= 1 or not np.isfinite(value):
        raise ValueError("趋势输入及健康参考尺度无效")
    z = (value - center) / scale
    return TrendState(
        (1 - weight) * state.ewma + weight * z,
        max(0.0, state.cusum + z - allowance),
        state.observations + 1,
    )
