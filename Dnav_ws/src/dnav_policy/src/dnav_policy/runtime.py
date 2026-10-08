"""ROS-independent policy inference runtime."""

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np
import torch

from .checkpoint import FINAL_POLICY_SHA256, load_policy_checkpoint
from .geometry import normalize_quaternion_wxyz
from .model import PolicyActor


@dataclass(frozen=True)
class PolicyOutput:
    velocity_body: np.ndarray
    velocity_world: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    latency_ms: float


def _torch_quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion_vector = quaternion[..., 1:]
    cross = 2.0 * torch.cross(quaternion_vector, vector, dim=-1)
    return vector + quaternion[..., :1] * cross + torch.cross(quaternion_vector, cross, dim=-1)


class PolicyRuntime:
    def __init__(
        self,
        checkpoint_path,
        *,
        device: str = "cuda",
        fallback_to_cpu: bool = True,
        expected_sha256: Optional[str] = FINAL_POLICY_SHA256,
        action_limit: float = 3.0,
    ):
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            if not fallback_to_cpu:
                raise RuntimeError("CUDA was requested but is unavailable")
            requested = torch.device("cpu")
        self.device = requested
        self.model = PolicyActor(action_limit=action_limit)
        self.checkpoint_sha256 = load_policy_checkpoint(
            self.model, checkpoint_path, expected_sha256=expected_sha256
        )
        self.model.to(self.device).eval()
        self.last_latency_ms = 0.0
        self._warmup()

    def _warmup(self) -> None:
        static = torch.ones(1, 1, 6, 36, device=self.device)
        dynamic = torch.zeros(1, 5, 6, 36, device=self.device)
        dynamic[:, 0] = 1.0
        state = torch.zeros(1, 13, device=self.device)
        with torch.inference_mode():
            self.model(static, dynamic, state)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _input(name: str, value, shape) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, received {result.shape}")
        if not np.isfinite(result).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return np.ascontiguousarray(result)

    @torch.inference_mode()
    def infer(
        self,
        lidar_static,
        lidar_dynamic,
        state,
        quaternion_world_body_wxyz,
    ) -> PolicyOutput:
        static_np = self._input("lidar_static", lidar_static, (1, 6, 36))
        dynamic_np = self._input("lidar_dynamic", lidar_dynamic, (5, 6, 36))
        state_np = self._input("state", state, (13,))
        quaternion_np = normalize_quaternion_wxyz(quaternion_world_body_wxyz).astype(
            np.float32
        )

        static = torch.from_numpy(static_np).unsqueeze(0).to(self.device)
        dynamic = torch.from_numpy(dynamic_np).unsqueeze(0).to(self.device)
        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        quaternion = torch.from_numpy(quaternion_np).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        alpha, beta = self.model.distribution_parameters(static, dynamic, state_tensor)
        normalized_action = alpha / (alpha + beta)
        velocity_body = (
            2.0 * normalized_action * self.model.action_limit - self.model.action_limit
        )
        velocity_world = _torch_quaternion_rotate(quaternion, velocity_body)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not bool(
            torch.isfinite(alpha).all()
            and torch.isfinite(beta).all()
            and torch.isfinite(velocity_body).all()
            and torch.isfinite(velocity_world).all()
        ):
            raise RuntimeError("policy output contains NaN or Inf")
        if not bool((velocity_body.abs() <= self.model.action_limit + 1e-6).all()):
            raise RuntimeError("policy body velocity exceeds the configured action limit")
        self.last_latency_ms = latency_ms
        return PolicyOutput(
            velocity_body=velocity_body[0].cpu().numpy().copy(),
            velocity_world=velocity_world[0].cpu().numpy().copy(),
            alpha=alpha[0].cpu().numpy().copy(),
            beta=beta[0].cpu().numpy().copy(),
            latency_ms=latency_ms,
        )
