"""Strict adapter for Gazebo's flattened 16 x 180 LaserScan."""

from dataclasses import dataclass
import math
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class ScanSpec:
    height: int = 16
    width: int = 360
    max_range: float = 10.0
    vertical_min_deg: float = -7.0
    vertical_max_deg: float = 52.0


def ranges_to_sensor_points(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    spec: ScanSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened sensor-frame XYZ points and explicit valid flags.

    Gazebo emits vertical rays from minimum to maximum pitch. Points remain in
    that order here; the shared panorama projector establishes row-zero-high.
    """
    raw = np.asarray(ranges, dtype=np.float32)
    expected = spec.height * spec.width
    if raw.size != expected:
        raise ValueError(f"expected {expected} ranges, received {raw.size}")
    if not math.isfinite(angle_min) or not math.isfinite(angle_increment) or angle_increment <= 0.0:
        raise ValueError("LaserScan horizontal angle metadata is invalid")

    raw = raw.reshape(spec.height, spec.width)
    valid = np.isfinite(raw) & (raw >= max(float(range_min), 1e-3))
    valid &= raw <= min(float(range_max), spec.max_range)
    safe_range = np.where(valid, raw, spec.max_range).astype(np.float32)

    yaw = angle_min + np.arange(spec.width, dtype=np.float32) * angle_increment
    pitch = np.deg2rad(
        np.linspace(spec.vertical_min_deg, spec.vertical_max_deg, spec.height, dtype=np.float32)
    )
    pitch, yaw = np.meshgrid(pitch, yaw, indexing="ij")
    cos_pitch = np.cos(pitch)
    points = np.stack(
        [
            safe_range * cos_pitch * np.cos(yaw),
            safe_range * cos_pitch * np.sin(yaw),
            safe_range * np.sin(pitch),
        ],
        axis=-1,
    )
    return points.reshape(-1, 3).astype(np.float32), valid.reshape(-1)


def transform_points(points: np.ndarray, translation: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Apply a rigid sensor-to-body transform to row-vector points."""
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("transform quaternion is invalid")
    w, x, y, z = q / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return (np.asarray(points) @ rotation.T + np.asarray(translation)).astype(np.float32)
