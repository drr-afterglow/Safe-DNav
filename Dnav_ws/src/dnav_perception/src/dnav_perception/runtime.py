"""Stateful TorchScript perception runtime."""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision  # noqa: F401 - registers torchvision::deform_conv2d

from .geometry import (
    PanoramaSpec,
    absolute_to_relative_spherical_velocity,
    align_previous_body_frame,
    make_network_input,
    project_points_to_panorama,
)
from .postprocess import build_rl_observation


class PerceptionRuntime:
    def __init__(
        self,
        model_path: str,
        spec: PanoramaSpec,
        device: str = "cuda",
        reference_delta_t: float = 0.096,
        min_delta_t: float = 0.05,
        max_delta_t: float = 0.15,
        mask_threshold: float = 0.5,
        dyn_max_vel: float = 5.0,
        output_size: Tuple[int, int] = (6, 36),
    ):
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        self.device = requested
        self.spec = spec
        self.reference_delta_t = float(reference_delta_t)
        self.min_delta_t = float(min_delta_t)
        self.max_delta_t = float(max_delta_t)
        self.mask_threshold = float(mask_threshold)
        self.dyn_max_vel = float(dyn_max_vel)
        self.output_size = tuple(output_size)
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        self.reset()
        self._warmup()

    def _warmup(self) -> None:
        value = torch.ones((1, 4, self.spec.height, self.spec.width), device=self.device)
        value[:, 2:] = 0.0
        dt = torch.tensor([self.reference_delta_t], device=self.device)
        with torch.inference_mode():
            self.model(value, dt)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset(self) -> None:
        self.previous_points: Optional[torch.Tensor] = None
        self.previous_valid: Optional[torch.Tensor] = None
        self.previous_position: Optional[torch.Tensor] = None
        self.previous_quaternion: Optional[torch.Tensor] = None
        self.previous_stamp: Optional[float] = None
        self.last_depth_image = None
        self.last_probability_image = None

    @torch.inference_mode()
    def process(
        self,
        points_body,
        return_valid,
        stamp: float,
        position_world,
        quaternion_world_wxyz,
        linear_velocity_body,
    ):
        points = torch.as_tensor(points_body, dtype=torch.float32, device=self.device).reshape(1, -1, 3)
        valid = torch.as_tensor(return_valid, dtype=torch.bool, device=self.device).reshape(1, -1)
        position = torch.as_tensor(position_world, dtype=torch.float32, device=self.device).reshape(1, 3)
        quaternion = torch.as_tensor(quaternion_world_wxyz, dtype=torch.float32, device=self.device).reshape(1, 4)
        ego_velocity = torch.as_tensor(linear_velocity_body, dtype=torch.float32, device=self.device).reshape(1, 3)
        if not all(bool(torch.isfinite(value).all()) for value in (points, position, quaternion, ego_velocity)):
            raise ValueError("perception input contains NaN or Inf")
        quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)

        history_valid = self.previous_points is not None
        delta_t = self.reference_delta_t
        if history_valid:
            delta_t = float(stamp - self.previous_stamp)
            if not self.min_delta_t <= delta_t <= self.max_delta_t:
                self.reset()
                history_valid = False
                delta_t = self.reference_delta_t

        if history_valid:
            previous_aligned = align_previous_body_frame(
                self.previous_points,
                self.previous_position,
                self.previous_quaternion,
                position,
                quaternion,
            )
            previous_valid = self.previous_valid
        else:
            previous_aligned = points
            previous_valid = valid

        network_input = make_network_input(points, previous_aligned, self.spec, valid, previous_valid)
        prediction = self.model(
            network_input, torch.tensor([delta_t], dtype=torch.float32, device=self.device)
        )
        expected = {
            "mask_logits": (1, 1, self.spec.height, self.spec.width),
            "velocity": (1, 3, self.spec.height, self.spec.width),
            "log_sigma": (1, 3, self.spec.height, self.spec.width),
        }
        for key, shape in expected.items():
            if key not in prediction or tuple(prediction[key].shape) != shape:
                raise RuntimeError(f"invalid model output {key}: expected {shape}")
            if not bool(torch.isfinite(prediction[key]).all()):
                raise RuntimeError(f"model output {key} contains NaN or Inf")

        if not history_valid:
            prediction["mask_logits"] = torch.full_like(prediction["mask_logits"], -20.0)
            prediction["velocity"] = torch.zeros_like(prediction["velocity"])
        relative_velocity = absolute_to_relative_spherical_velocity(
            prediction["velocity"], ego_velocity, self.spec.vfov_deg
        )
        current_range = project_points_to_panorama(points, self.spec, valid_mask=valid) / self.spec.max_range
        self.last_depth_image = current_range[0, 0].detach().cpu().numpy()
        self.last_probability_image = torch.sigmoid(prediction["mask_logits"])[0, 0].detach().cpu().numpy()
        static, dynamic = build_rl_observation(
            current_range,
            prediction["mask_logits"],
            relative_velocity,
            prediction["log_sigma"],
            dyn_max_vel=self.dyn_max_vel,
            mask_threshold=self.mask_threshold,
            output_size=self.output_size,
        )
        if not bool(torch.isfinite(static).all() and torch.isfinite(dynamic).all()):
            raise RuntimeError("post-processed observation contains NaN or Inf")

        self.previous_points = points.detach()
        self.previous_valid = valid.detach()
        self.previous_position = position.detach()
        self.previous_quaternion = quaternion.detach()
        self.previous_stamp = float(stamp)
        return static.cpu().numpy(), dynamic.cpu().numpy(), history_valid, float(delta_t)
