"""健康运动回归：以指令位置、速度、加速度和方向解释跟随误差及电流。"""

import numpy as np
from scipy.signal import savgol_filter

BASIS_NAMES = (
    "position_mm",
    "velocity_mm_s",
    "acceleration_mm_s2",
    "direction",
    "quadratic_velocity_mm2_s2",
)


def motion_basis(command):
    """25Hz离线周期；SG11/poly3仅作用于已知指令，非实测阀位或因果在线滤波。"""
    position = np.asarray(command, dtype=np.float64)
    velocity = savgol_filter(position, 11, 3, deriv=1, delta=1 / 25, axis=1)
    acceleration = savgol_filter(position, 11, 3, deriv=2, delta=1 / 25, axis=1)
    direction = np.where(np.abs(velocity) > 0.05, np.sign(velocity), 0)
    return np.stack(
        [position, velocity, acceleration, direction, velocity * np.abs(velocity)],
        axis=-1,
    )


def fit_motion_reference(basis, signals, reference_windows):
    """只拟合指定健康窗口；系数是经验健康参考，不解释为已识别的惯量或摩擦力。"""
    x = basis[reference_windows].reshape(-1, len(BASIS_NAMES))
    y = signals[reference_windows].reshape(-1, 2)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    centered = (x - mean) / scale
    y_mean = y.mean(0)
    coefficients = np.linalg.solve(
        centered.T @ centered / len(x) + 1e-3 * np.eye(len(BASIS_NAMES)),
        centered.T @ (y - y_mean) / len(x),
    )
    residual = y - (centered @ coefficients + y_mean)
    rms = np.maximum(np.sqrt(np.mean(residual**2, axis=0)), 1e-6)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "y_mean": y_mean.tolist(),
        "coefficients": coefficients.tolist(),
        "reference_rms": rms.tolist(),
        "raw_basis_coefficients": (coefficients / scale[:, None]).tolist(),
        "raw_basis_intercept": (
            y_mean - mean @ (coefficients / scale[:, None])
        ).tolist(),
    }


def motion_residual(basis, signals, reference, standardized=False):
    prediction = (
        (basis - np.asarray(reference["mean"])) / np.asarray(reference["scale"])
    ) @ np.asarray(reference["coefficients"]) + np.asarray(reference["y_mean"])
    residual = signals - prediction
    if standardized:
        residual = residual / np.asarray(reference["reference_rms"])
    return residual.astype("float32")
