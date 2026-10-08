"""Quaternion geometry for policy state and action conversion."""

import numpy as np


def normalize_quaternion_wxyz(quaternion) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(value).all():
        raise ValueError("quaternion contains NaN or Inf")
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError("quaternion norm is too small")
    return value / norm


def ros_xyzw_to_wxyz(quaternion) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return normalize_quaternion_wxyz(value[[3, 0, 1, 2]])


def quaternion_conjugate(quaternion_wxyz) -> np.ndarray:
    value = normalize_quaternion_wxyz(quaternion_wxyz).copy()
    value[1:] *= -1.0
    return value


def quaternion_rotate(quaternion_wxyz, vector) -> np.ndarray:
    quaternion = normalize_quaternion_wxyz(quaternion_wxyz)
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("vector must be a finite three-vector")
    q_vector = quaternion[1:]
    cross = 2.0 * np.cross(q_vector, value)
    return value + quaternion[0] * cross + np.cross(q_vector, cross)


def world_to_body(quaternion_world_body_wxyz, vector_world) -> np.ndarray:
    return quaternion_rotate(quaternion_conjugate(quaternion_world_body_wxyz), vector_world)


def body_to_world(quaternion_world_body_wxyz, vector_body) -> np.ndarray:
    return quaternion_rotate(quaternion_world_body_wxyz, vector_body)
