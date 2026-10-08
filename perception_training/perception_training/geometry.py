#点云几何处理
#1.imu点云去畸变
#2.上一帧点云对齐到当前帧坐标系
#3.点云投影到全景图像
#4.拼接当前帧和上一帧的全景图像作为网络输入
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch


@dataclass
class PanoramaSpec:
    height: int = 6
    width: int = 36
    max_range: float = 10.0
    vfov_deg: Tuple[float, float] = (-7.0, 52.0)

    @property
    def vfov_rad(self) -> Tuple[float, float]:
        return (math.radians(self.vfov_deg[0]), math.radians(self.vfov_deg[1]))


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for Isaac/OmniDrones ``(w, x, y, z)`` quaternions."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def integrate_body_angular_velocity(
    quat_w: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Integrate a body-frame angular rate and return a normalized quaternion."""
    rotation_vector = angular_velocity_body * float(dt)
    angle = torch.linalg.norm(rotation_vector, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1e-8,
        torch.sin(half_angle) / angle.clamp_min(1e-8),
        0.5 - angle.square() / 48.0,
    )
    delta_quat = torch.cat([torch.cos(half_angle), rotation_vector * scale], dim=-1)
    integrated = quat_multiply(quat_w, delta_quat)
    return integrated / torch.linalg.norm(integrated, dim=-1, keepdim=True).clamp_min(1e-8)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Quaternion convention follows Isaac/OmniDrones: (w, x, y, z).
    q_vec = q[..., 1:]
    q_w = q[..., :1]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conjugate(q), v)


def transform_world_to_body(points_w: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    if pos_w.ndim == 2:
        pos_w = pos_w.unsqueeze(1)
    if quat_w.ndim == 2:
        quat_w = quat_w.unsqueeze(1)
    rel = points_w - pos_w
    quat = quat_w.expand(*rel.shape[:-1], 4)
    return quat_rotate_inverse(quat, rel)


def transform_body_to_world(points_b: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    if pos_w.ndim == 2:
        pos_w = pos_w.unsqueeze(1)
    if quat_w.ndim == 2:
        quat_w = quat_w.unsqueeze(1)
    quat = quat_w.expand(*points_b.shape[:-1], 4)
    return quat_rotate(quat, points_b) + pos_w


def align_previous_body_frame(
    prev_points_body: torch.Tensor,
    prev_pos_w: torch.Tensor,
    prev_quat_w: torch.Tensor,
    curr_pos_w: torch.Tensor,
    curr_quat_w: torch.Tensor,
) -> torch.Tensor:
    prev_world = transform_body_to_world(prev_points_body, prev_pos_w, prev_quat_w)
    return transform_world_to_body(prev_world, curr_pos_w, curr_quat_w)


def absolute_to_relative_spherical_velocity(
    absolute_velocity: torch.Tensor,
    ego_velocity_body: torch.Tensor,
    vfov_deg: Tuple[float, float],
) -> torch.Tensor:
    """Subtract ego translation from a dense absolute spherical velocity field.

    The body frame is FLU (x forward, y left, z up). Spherical components are
    ordered as radial, increasing azimuth, and increasing pitch. The UAV's
    Cartesian body-frame velocity is projected into the local spherical basis
    independently at every panorama pixel before subtraction.
    """
    if absolute_velocity.ndim != 4 or absolute_velocity.shape[1] != 3:
        raise ValueError("absolute_velocity must have shape (B, 3, H, W)")
    if ego_velocity_body.ndim == 1:
        ego_velocity_body = ego_velocity_body.unsqueeze(0)
    if ego_velocity_body.ndim != 2 or ego_velocity_body.shape[-1] != 3:
        raise ValueError("ego_velocity_body must have shape (B, 3) or (3,)")
    if ego_velocity_body.shape[0] not in (1, absolute_velocity.shape[0]):
        raise ValueError("ego velocity batch size must be one or match the velocity field")

    _, _, height, width = absolute_velocity.shape
    dtype = absolute_velocity.dtype
    device = absolute_velocity.device
    theta = -math.pi + (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 * math.pi / width)
    pitch_max = math.radians(vfov_deg[1])
    pitch_step = math.radians(vfov_deg[1] - vfov_deg[0]) / height
    phi = pitch_max - (torch.arange(height, device=device, dtype=dtype) + 0.5) * pitch_step
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    e_r = torch.stack([cos_phi * cos_theta, cos_phi * sin_theta, sin_phi], dim=0)
    e_theta = torch.stack([-sin_theta, cos_theta, torch.zeros_like(theta)], dim=0)
    e_phi = torch.stack([-sin_phi * cos_theta, -sin_phi * sin_theta, cos_phi], dim=0)
    basis = torch.stack([e_r, e_theta, e_phi], dim=0)

    ego = ego_velocity_body.to(device=device, dtype=dtype)
    ego_spherical = torch.einsum("bc,kchw->bkhw", ego, basis)
    return absolute_velocity - ego_spherical


def deskew_points_body(
    raw_points_body: torch.Tensor,
    point_times: Optional[torch.Tensor],
    linear_vel_body: torch.Tensor,
    angular_vel_body: torch.Tensor,
    frame_time: Union[float, torch.Tensor],
) -> torch.Tensor:
    """First-order IMU deskew to the frame-end body coordinate."""
    if point_times is None:
        return raw_points_body
    if linear_vel_body.ndim == 2:
        linear_vel_body = linear_vel_body.unsqueeze(1)
    if angular_vel_body.ndim == 2:
        angular_vel_body = angular_vel_body.unsqueeze(1)
    time_dtype = point_times.dtype if point_times.is_floating_point() else torch.float64
    frame_time_tensor = torch.as_tensor(frame_time, device=raw_points_body.device, dtype=time_dtype)
    if frame_time_tensor.ndim == 0:
        frame_time_tensor = frame_time_tensor.reshape(1, 1)
    elif frame_time_tensor.ndim == 1:
        frame_time_tensor = frame_time_tensor.unsqueeze(-1)
    dt = (frame_time_tensor - point_times.to(dtype=time_dtype)).to(dtype=raw_points_body.dtype).unsqueeze(-1)
    translated = raw_points_body - linear_vel_body * dt
    # Small-angle inverse rotation approximation, adequate for high-rate lidar frames.
    return translated - torch.cross(angular_vel_body * dt, translated, dim=-1)


def project_points_to_panorama(
    points_body: torch.Tensor,
    spec: PanoramaSpec,
    values: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    reduce: str = "amin",
    empty_value: Optional[float] = None,
) -> torch.Tensor:
    """Project body-frame points to a panoramic image.

    If values is None, depth is projected with nearest-depth reduction.
    Otherwise values must be shaped (B, N, C) or (B, N). With
    reduce="nearest", feature values are selected from the nearest valid point
    in each cell; otherwise features are averaged per cell.
    """
    bsz, num_points, _ = points_body.shape
    device = points_body.device
    dtype = points_body.dtype
    x, y, z = points_body.unbind(dim=-1)
    depth = torch.linalg.norm(points_body, dim=-1)

    yaw = torch.atan2(y, x)
    pitch = torch.asin((z / depth.clamp_min(1e-6)).clamp(-1.0, 1.0))
    vmin, vmax = spec.vfov_rad
    h_res = (2.0 * math.pi) / spec.width
    v_res = (vmax - vmin) / spec.height
    u = torch.floor((yaw + math.pi) / h_res).long().clamp(0, spec.width - 1)
    # Equirectangular image convention from the final design: row zero is the
    # upper pitch limit and row H-1 is the lower pitch limit.
    v = torch.floor((vmax - pitch) / v_res).long().clamp(0, spec.height - 1)
    flat_idx = v * spec.width + u

    valid = (depth > 1e-3) & (depth <= spec.max_range)
    if valid_mask is not None:
        valid = valid & valid_mask

    if values is None:
        fill = spec.max_range if empty_value is None else empty_value
        img = torch.full((bsz, spec.height * spec.width), fill, device=device, dtype=dtype)
        vals = torch.where(valid, depth, torch.full_like(depth, float("inf")))
        img.scatter_reduce_(1, flat_idx, vals, reduce=reduce, include_self=True)
        img = img.clamp(max=spec.max_range)
        return img.view(bsz, 1, spec.height, spec.width)

    if values.ndim == 2:
        values = values.unsqueeze(-1)
    channels = values.shape[-1]
    safe_idx = torch.where(valid, flat_idx, torch.full_like(flat_idx, spec.height * spec.width))
    if reduce == "nearest":
        min_depth = torch.full((bsz, spec.height * spec.width), float("inf"), device=device, dtype=dtype)
        depth_vals = torch.where(valid, depth, torch.full_like(depth, float("inf")))
        min_depth.scatter_reduce_(1, flat_idx, depth_vals, reduce="amin", include_self=True)
        nearest_depth = min_depth.gather(1, flat_idx)
        nearest = valid & torch.isclose(depth, nearest_depth, rtol=1e-5, atol=1e-6)
        safe_idx = torch.where(nearest, flat_idx, torch.full_like(flat_idx, spec.height * spec.width))
        out = torch.zeros((bsz, spec.height * spec.width + 1, channels), device=device, dtype=dtype)
        out.scatter_(1, safe_idx.unsqueeze(-1).expand(-1, -1, channels), values * nearest.unsqueeze(-1))
        return out[:, : spec.height * spec.width].transpose(1, 2).view(bsz, channels, spec.height, spec.width)

    out = torch.zeros((bsz, spec.height * spec.width + 1, channels), device=device, dtype=dtype)
    cnt = torch.zeros((bsz, spec.height * spec.width + 1, 1), device=device, dtype=dtype)
    out.scatter_add_(1, safe_idx.unsqueeze(-1).expand(-1, -1, channels), values * valid.unsqueeze(-1))
    cnt.scatter_add_(1, safe_idx.unsqueeze(-1), valid.unsqueeze(-1).to(dtype))
    out = out[:, : spec.height * spec.width] / cnt[:, : spec.height * spec.width].clamp_min(1.0)
    return out.transpose(1, 2).view(bsz, channels, spec.height, spec.width)


def project_range_and_valid(
    points_body: torch.Tensor,
    spec: PanoramaSpec,
    return_valid: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project range and an explicit return-valid map with identical geometry."""
    if return_valid is None:
        depth = torch.linalg.norm(points_body, dim=-1)
        return_valid = torch.isfinite(depth) & (depth > 1e-3) & (depth <= spec.max_range)
    else:
        return_valid = return_valid.to(device=points_body.device, dtype=torch.bool)
        if return_valid.shape != points_body.shape[:2]:
            raise ValueError("return_valid must have shape (B, N) matching points_body")
    distance = project_points_to_panorama(points_body, spec, valid_mask=return_valid)
    ones = torch.ones((*points_body.shape[:2], 1), device=points_body.device, dtype=points_body.dtype)
    valid = project_points_to_panorama(
        points_body,
        spec,
        values=ones,
        valid_mask=return_valid,
        reduce="nearest",
    ).clamp(0.0, 1.0)
    if normalize:
        distance = distance / spec.max_range
    return distance, valid


def make_network_input(
    curr_points_body: torch.Tensor,
    prev_points_body_aligned: torch.Tensor,
    spec: PanoramaSpec,
    normalize: bool = True,
    curr_return_valid: Optional[torch.Tensor] = None,
    prev_return_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    curr, curr_valid = project_range_and_valid(curr_points_body, spec, curr_return_valid, normalize)
    prev, prev_valid = project_range_and_valid(prev_points_body_aligned, spec, prev_return_valid, normalize)
    return torch.cat([curr, prev, curr_valid, prev_valid], dim=1)
