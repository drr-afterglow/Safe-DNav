"""Thread-safe timestamped input cache for the ROS policy node."""

from dataclasses import dataclass
from threading import Lock
from typing import Optional, Tuple

import numpy as np


class InputUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PerceptionSample:
    stamp: float
    lidar_static: np.ndarray
    lidar_dynamic: np.ndarray
    history_valid: bool


@dataclass(frozen=True)
class OdometrySample:
    stamp: float
    position_world: np.ndarray
    quaternion_world_body_wxyz: np.ndarray
    linear_velocity_body: np.ndarray
    angular_velocity_body: np.ndarray


@dataclass(frozen=True)
class GoalSample:
    stamp: float
    position_world: np.ndarray


class PolicyInputCache:
    def __init__(self):
        self._lock = Lock()
        self._perception: Optional[PerceptionSample] = None
        self._odometry: Optional[OdometrySample] = None
        self._goal: Optional[GoalSample] = None
        self._generation = 0
        self._last_clock: Optional[float] = None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def clear(self) -> None:
        with self._lock:
            self._perception = None
            self._odometry = None
            self._goal = None
            self._generation += 1
            self._last_clock = None

    def update_perception(self, sample: PerceptionSample) -> None:
        with self._lock:
            if self._perception is not None and sample.stamp < self._perception.stamp:
                self._generation += 1
            self._perception = sample

    def update_odometry(self, sample: OdometrySample) -> None:
        with self._lock:
            if self._odometry is not None and sample.stamp < self._odometry.stamp:
                self._generation += 1
            self._odometry = sample

    def update_goal(self, sample: GoalSample) -> None:
        with self._lock:
            self._goal = sample
            self._generation += 1

    def snapshot(
        self,
        now: float,
        *,
        perception_timeout: float,
        odometry_timeout: float,
    ) -> Tuple[PerceptionSample, OdometrySample, GoalSample, int]:
        if not np.isfinite(now):
            raise InputUnavailable("ROS clock is invalid")
        with self._lock:
            if self._last_clock is not None and now < self._last_clock:
                self._generation += 1
            self._last_clock = now
            perception = self._perception
            odometry = self._odometry
            goal = self._goal
            generation = self._generation
        if perception is None:
            raise InputUnavailable("waiting for perception")
        if odometry is None:
            raise InputUnavailable("waiting for odometry")
        if goal is None:
            raise InputUnavailable("waiting for goal")
        perception_age = now - perception.stamp
        odometry_age = now - odometry.stamp
        if perception_age < -1e-3 or perception_age > perception_timeout:
            raise InputUnavailable(f"perception stale ({perception_age:.3f} s)")
        if odometry_age < -1e-3 or odometry_age > odometry_timeout:
            raise InputUnavailable(f"odometry stale ({odometry_age:.3f} s)")
        return perception, odometry, goal, generation
