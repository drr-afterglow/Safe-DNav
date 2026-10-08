"""Frozen deployment geometry from perception_training."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class PanoramaSpec:
    height: int = 16
    width: int = 180
    max_range: float = 10.0
    vfov_deg: Tuple[float, float] = (-7.0, 52.0)

    @property
    def vfov_rad(self) -> Tuple[float, float]:
        return math.radians(self.vfov_deg[0]), math.radians(self.vfov_deg[1])


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_vec, q_w = q[..., 1:], q[..., :1]
    cross = 2.0 * torch.cross(q_vec, vector, dim=-1)
    return vector + q_w * cross + torch.cross(q_vec, cross, dim=-1)


def _body_to_world(points: torch.Tensor, position: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    position = position.unsqueeze(1) if position.ndim == 2 else position
    quaternion = quaternion.unsqueeze(1) if quaternion.ndim == 2 else quaternion
    return quat_rotate(quaternion.expand(*points.shape[:-1], 4), points) + position


def _world_to_body(points: torch.Tensor, position: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    position = position.unsqueeze(1) if position.ndim == 2 else position
    quaternion = quaternion.unsqueeze(1) if quaternion.ndim == 2 else quaternion
    relative = points - position
    return quat_rotate(quat_conjugate(quaternion).expand(*relative.shape[:-1], 4), relative)


def align_previous_body_frame(
    previous_points: torch.Tensor,
    previous_position: torch.Tensor,
    previous_quaternion: torch.Tensor,
    current_position: torch.Tensor,
    current_quaternion: torch.Tensor,
) -> torch.Tensor:
    world = _body_to_world(previous_points, previous_position, previous_quaternion)
    return _world_to_body(world, current_position, current_quaternion)


def absolute_to_relative_spherical_velocity(
    absolute_velocity: torch.Tensor,
    ego_velocity_body: torch.Tensor,
    vfov_deg: Tuple[float, float],
) -> torch.Tensor:
    if absolute_velocity.ndim != 4 or absolute_velocity.shape[1] != 3:
        raise ValueError("absolute_velocity must have shape (B, 3, H, W)")
    if ego_velocity_body.ndim == 1:
        ego_velocity_body = ego_velocity_body.unsqueeze(0)
    _, _, height, width = absolute_velocity.shape
    dtype, device = absolute_velocity.dtype, absolute_velocity.device
    theta = -math.pi + (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 * math.pi / width)
    pitch_step = math.radians(vfov_deg[1] - vfov_deg[0]) / height
    phi = math.radians(vfov_deg[1]) - (torch.arange(height, device=device, dtype=dtype) + 0.5) * pitch_step
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    e_r = torch.stack((cos_phi * cos_theta, cos_phi * sin_theta, sin_phi), dim=0)
    e_theta = torch.stack((-sin_theta, cos_theta, torch.zeros_like(theta)), dim=0)
    e_phi = torch.stack((-sin_phi * cos_theta, -sin_phi * sin_theta, cos_phi), dim=0)
    basis = torch.stack((e_r, e_theta, e_phi), dim=0)
    ego = ego_velocity_body.to(device=device, dtype=dtype)
    return absolute_velocity - torch.einsum("bc,kchw->bkhw", ego, basis)


def project_points_to_panorama(
    points: torch.Tensor,
    spec: PanoramaSpec,
    values: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch, _, _ = points.shape
    depth = torch.linalg.norm(points, dim=-1)
    x, y, z = points.unbind(dim=-1)
    yaw = torch.atan2(y, x)
    pitch = torch.asin((z / depth.clamp_min(1e-6)).clamp(-1.0, 1.0))
    vmin, vmax = spec.vfov_rad
    column = torch.floor((yaw + math.pi) / (2.0 * math.pi / spec.width)).long().clamp(0, spec.width - 1)
    row = torch.floor((vmax - pitch) / ((vmax - vmin) / spec.height)).long().clamp(0, spec.height - 1)
    index = row * spec.width + column
    valid = (depth > 1e-3) & (depth <= spec.max_range)
    if valid_mask is not None:
        valid &= valid_mask

    if values is None:
        image = torch.full(
            (batch, spec.height * spec.width), spec.max_range, device=points.device, dtype=points.dtype
        )
        depths = torch.where(valid, depth, torch.full_like(depth, float("inf")))
        image.scatter_reduce_(1, index, depths, reduce="amin", include_self=True)
        return image.clamp(max=spec.max_range).view(batch, 1, spec.height, spec.width)

    if values.ndim == 2:
        values = values.unsqueeze(-1)
    channels = values.shape[-1]
    min_depth = torch.full(
        (batch, spec.height * spec.width), float("inf"), device=points.device, dtype=points.dtype
    )
    min_depth.scatter_reduce_(
        1, index, torch.where(valid, depth, torch.full_like(depth, float("inf"))), reduce="amin", include_self=True
    )
    nearest = valid & torch.isclose(depth, min_depth.gather(1, index), rtol=1e-5, atol=1e-6)
    safe_index = torch.where(nearest, index, torch.full_like(index, spec.height * spec.width))
    output = torch.zeros(
        (batch, spec.height * spec.width + 1, channels), device=points.device, dtype=points.dtype
    )
    output.scatter_(1, safe_index.unsqueeze(-1).expand(-1, -1, channels), values * nearest.unsqueeze(-1))
    return output[:, :-1].transpose(1, 2).reshape(batch, channels, spec.height, spec.width)


def project_range_and_valid(
    points: torch.Tensor, spec: PanoramaSpec, valid_mask: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if valid_mask is None:
        depth = torch.linalg.norm(points, dim=-1)
        valid_mask = torch.isfinite(depth) & (depth > 1e-3) & (depth <= spec.max_range)
    distance = project_points_to_panorama(points, spec, valid_mask=valid_mask) / spec.max_range
    ones = torch.ones((*points.shape[:2], 1), device=points.device, dtype=points.dtype)
    valid = project_points_to_panorama(points, spec, values=ones, valid_mask=valid_mask).clamp(0.0, 1.0)
    return distance, valid


def make_network_input(
    current_points: torch.Tensor,
    previous_points_aligned: torch.Tensor,
    spec: PanoramaSpec,
    current_valid: Optional[torch.Tensor] = None,
    previous_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    current_range, current_valid_image = project_range_and_valid(current_points, spec, current_valid)
    previous_range, previous_valid_image = project_range_and_valid(previous_points_aligned, spec, previous_valid)
    return torch.cat((current_range, previous_range, current_valid_image, previous_valid_image), dim=1)
