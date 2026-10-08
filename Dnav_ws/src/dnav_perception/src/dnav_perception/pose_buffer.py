"""Timestamped odometry buffering and interpolation."""

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PoseSample:
    stamp: float
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_body: np.ndarray


def _slerp(q0: np.ndarray, q1: np.ndarray, ratio: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + ratio * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    return (np.sin((1.0 - ratio) * angle) * q0 + np.sin(ratio * angle) * q1) / np.sin(angle)


class PoseBuffer:
    def __init__(self, maxlen: int = 300):
        self._samples = deque(maxlen=maxlen)
        self._lock = Lock()

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def add(self, sample: PoseSample) -> None:
        if not np.isfinite(sample.stamp):
            raise ValueError("pose timestamp must be finite")
        with self._lock:
            if self._samples and sample.stamp < self._samples[-1].stamp:
                self._samples.clear()
            self._samples.append(sample)

    def interpolate(self, stamp: float, tolerance: float, allow_nearest: bool = False) -> Optional[PoseSample]:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return None
        for sample in samples:
            if abs(sample.stamp - stamp) <= 1e-9:
                return sample
        for left, right in zip(samples[:-1], samples[1:]):
            if left.stamp <= stamp <= right.stamp:
                if stamp - left.stamp > tolerance or right.stamp - stamp > tolerance:
                    return None
                span = right.stamp - left.stamp
                ratio = 0.0 if span <= 1e-12 else (stamp - left.stamp) / span
                return PoseSample(
                    stamp=stamp,
                    position=(1.0 - ratio) * left.position + ratio * right.position,
                    quaternion_wxyz=_slerp(left.quaternion_wxyz, right.quaternion_wxyz, ratio),
                    linear_velocity_body=(1.0 - ratio) * left.linear_velocity_body
                    + ratio * right.linear_velocity_body,
                )
        if allow_nearest:
            nearest = min(samples, key=lambda item: abs(item.stamp - stamp))
            return nearest if abs(nearest.stamp - stamp) <= tolerance else None
        return None
