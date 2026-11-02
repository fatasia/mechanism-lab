"""采样率分组：不把温度插值成高频数据。"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class SignalProfile:
    channels: tuple[str, ...]
    units: tuple[str, ...]
    rate: int
    start: float = 0
    end: float = 60


PROFILES = {
    "valve-switch-100hz": SignalProfile(("PS1", "PS2"), ("bar", "bar"), 100, 8, 12),
    "thermal-1hz": SignalProfile(
        ("TS1", "TS2", "TS3", "TS4", "VS1"), ("degC",) * 4 + ("mm/s",), 1
    ),
    "process-10hz": SignalProfile(
        ("PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2"),
        ("bar",) * 6 + ("W", "l/min", "l/min"),
        10,
    ),
}


def load_profile(
    archive: Path, profile_name: str, cache: Path
) -> tuple[np.ndarray, np.ndarray, str]:
    """100 Hz 通道抗混叠降到 10 Hz；缓存与原始文件哈希绑定。"""
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    prepared = cache / f"{profile_name}-{digest[:12]}-v3.npz"
    if prepared.exists():
        with np.load(prepared) as data:
            return data["x"], data["labels"], digest
    profile = PROFILES[profile_name]
    with ZipFile(archive) as package:
        channels = []
        for name in profile.channels:
            values = np.loadtxt(package.open(f"{name}.txt")).astype("float32")
            if profile.rate == 10 and values.shape[1] == 6000:
                values = resample_poly(values, up=1, down=10, axis=1, padtype="line")
            values = values[
                :, int(profile.start * profile.rate) : int(profile.end * profile.rate)
            ]
            channels.append(values)
        x = np.stack(channels, axis=2).astype("float32")
        labels = np.loadtxt(package.open("profile.txt"))
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prepared, x=x, labels=labels)
    return x, labels, digest
