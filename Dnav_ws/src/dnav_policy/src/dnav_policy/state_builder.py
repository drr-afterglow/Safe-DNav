"""Exact construction of the trained policy's 13-dimensional state."""

import math
from typing import Optional

import numpy as np

from .geometry import normalize_quaternion_wxyz, world_to_body


class PolicyStateBuilder:
    STATE_VELOCITY_SCALE = 3.6
    STATE_ANGULAR_VELOCITY_SCALE = 2.0 * math.pi
    STATE_DISTANCE_SCALE = 0.5 * math.sqrt(48.0**2 + 48.0**2 + 2.0**2)

    def __init__(
        self,
        velocity_scale: float = STATE_VELOCITY_SCALE,
        angular_velocity_scale: float = STATE_ANGULAR_VELOCITY_SCALE,
        distance_scale: float = STATE_DISTANCE_SCALE,
    ):
        if velocity_scale <= 0.0 or angular_velocity_scale <= 0.0 or distance_scale <= 0.0:
            raise ValueError("state normalization scales must be positive")
        self.velocity_scale = float(velocity_scale)
        self.angular_velocity_scale = float(angular_velocity_scale)
        self.distance_scale = float(distance_scale)
        self.previous_velocity_body: Optional[np.ndarray] = None

    @staticmethod
    def _vector(name: str, value) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if result.shape != (3,):
            raise ValueError(f"{name} must contain three values")
        if not np.isfinite(result).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return result

    def reset(self) -> None:
        self.previous_velocity_body = None

    def build(
        self,
        position_world,
        quaternion_world_body_wxyz,
        linear_velocity_body,
        angular_velocity_body,
        goal_position_world,
    ) -> np.ndarray:
        position = self._vector("position_world", position_world)
        quaternion = normalize_quaternion_wxyz(quaternion_world_body_wxyz)
        velocity = self._vector("linear_velocity_body", linear_velocity_body)
        angular_velocity = self._vector("angular_velocity_body", angular_velocity_body)
        goal = self._vector("goal_position_world", goal_position_world)

        relative_position_world = goal - position
        distance = float(np.linalg.norm(relative_position_world))
        if distance > 1e-6:
            target_unit_body = world_to_body(
                quaternion, relative_position_world / distance
            )
        else:
            target_unit_body = np.zeros(3, dtype=np.float64)
        previous_velocity = (
            np.zeros(3, dtype=np.float64)
            if self.previous_velocity_body is None
            else self.previous_velocity_body
        )
        state = np.concatenate(
            (
                np.clip(velocity / self.velocity_scale, -1.0, 1.0),
                np.clip(
                    angular_velocity / self.angular_velocity_scale, -1.0, 1.0
                ),
                target_unit_body,
                np.array([math.tanh(distance / self.distance_scale)]),
                np.clip(previous_velocity / self.velocity_scale, -1.0, 1.0),
            )
        ).astype(np.float32)
        if state.shape != (13,) or not np.isfinite(state).all():
            raise RuntimeError("constructed policy state is invalid")
        self.previous_velocity_body = velocity.copy()
        return state
