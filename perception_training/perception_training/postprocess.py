#后处理RL预测
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


def spherical_to_radial_tangent(velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    v_radial = velocity[:, 0:1]
    v_tangent = torch.linalg.norm(velocity[:, 1:3], dim=1, keepdim=True)
    return v_radial, v_tangent


def _pool_to_size(x: torch.Tensor, output_size: Tuple[int, int], mode: str) -> torch.Tensor:
    if x.shape[-2:] == output_size:
        return x
    if mode == "min":
        return -F.adaptive_max_pool2d(-x, output_size)
    if mode == "max":
        return F.adaptive_max_pool2d(x, output_size)
    raise ValueError(f"Unsupported pooling mode: {mode}")


def _max_danger_pool(
    dyn_depth: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
    output_size: Tuple[int, int],
    max_range_norm: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dyn_depth.shape[-2:] == output_size:
        return dyn_depth, velocity, sigma, mask

    mask_pooled = _pool_to_size(mask, output_size, mode="max")
    # Frozen on the independent motion-risk validation set: preserve the
    # nearest predicted dynamic return in each RL cell. This is robust to
    # velocity-estimation noise and consistently retains substantially more
    # oracle CBF/TTC risk than the former speed-times-sigma score.
    danger = -dyn_depth
    danger = torch.where(mask > 0, danger, torch.full_like(danger, -torch.inf))
    _, max_idx = F.adaptive_max_pool2d(danger, output_size, return_indices=True)

    flat_depth = dyn_depth.flatten(2)
    depth_idx = max_idx.flatten(2)
    dyn_depth_pooled = flat_depth.gather(2, depth_idx).view(
        dyn_depth.shape[0], 1, output_size[0], output_size[1]
    )
    flat_velocity = velocity.flatten(2)
    flat_idx = max_idx.flatten(2).expand(-1, velocity.shape[1], -1)
    velocity_pooled = flat_velocity.gather(2, flat_idx).view(
        velocity.shape[0], velocity.shape[1], output_size[0], output_size[1]
    )
    flat_sigma = sigma.flatten(2)
    sigma_pooled = flat_sigma.gather(2, depth_idx).view(
        sigma.shape[0], 1, output_size[0], output_size[1]
    )
    has_dynamic = mask_pooled > 0
    dyn_depth_pooled = torch.where(
        has_dynamic,
        dyn_depth_pooled,
        torch.full_like(dyn_depth_pooled, max_range_norm),
    )
    velocity_pooled = velocity_pooled * has_dynamic.to(velocity.dtype)
    sigma_pooled = sigma_pooled * has_dynamic.to(sigma.dtype)
    return dyn_depth_pooled, velocity_pooled, sigma_pooled, mask_pooled


def build_rl_observation(
    curr_range_norm: torch.Tensor,
    mask_logits: torch.Tensor,
    velocity: torch.Tensor,
    log_sigma: torch.Tensor,
    dyn_max_vel: float,
    mask_threshold: float = 0.5,
    dynamic_channels: Literal[3, 5] = 3,
    hard_mask: bool = True,
    output_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert network heads to RL static/dynamic observation tensors."""
    prob = torch.sigmoid(mask_logits)
    mask = (prob > mask_threshold).to(curr_range_norm.dtype) if hard_mask else prob
    max_range_norm = 1.0
    static_depth = curr_range_norm * (1.0 - mask) + max_range_norm * mask
    dyn_depth = curr_range_norm * mask + max_range_norm * (1.0 - mask)

    vel_masked = velocity * mask
    v_radial, v_tangent = spherical_to_radial_tangent(vel_masked)
    sigma = torch.exp(log_sigma).mean(dim=1, keepdim=True) * mask

    if output_size is not None:
        static_depth = _pool_to_size(static_depth, output_size, mode="min")
        dyn_depth, vel_masked, sigma, mask = _max_danger_pool(
            dyn_depth,
            vel_masked,
            sigma,
            mask,
            output_size,
            max_range_norm=max_range_norm,
        )
        v_radial, v_tangent = spherical_to_radial_tangent(vel_masked)

    if dynamic_channels == 5:
        dyn = torch.cat(
            [
                dyn_depth,
                (vel_masked[:, 0:1] / dyn_max_vel).clamp(-1.0, 1.0),
                (vel_masked[:, 1:2] / dyn_max_vel).clamp(-1.0, 1.0),
                (vel_masked[:, 2:3] / dyn_max_vel).clamp(-1.0, 1.0),
                sigma.clamp(0.0, 10.0) / 10.0,
            ],
            dim=1,
        )
    else:
        dyn = torch.cat(
            [
                dyn_depth,
                (v_radial / dyn_max_vel).clamp(-1.0, 1.0),
                (v_tangent / dyn_max_vel).clamp(0.0, 1.0),
            ],
            dim=1,
        )
    return static_depth.clamp(0.0, 1.0), dyn.clamp(-1.0, 1.0)
