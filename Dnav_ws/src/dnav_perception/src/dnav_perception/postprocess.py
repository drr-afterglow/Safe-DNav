"""Frozen five-channel PPO/CBF observation post-processing."""

from typing import Tuple

import torch
import torch.nn.functional as functional


def _minimum_pool(value: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    return -functional.adaptive_max_pool2d(-value, output_size)


def _nearest_dynamic_pool(depth, velocity, sigma, mask, output_size):
    pooled_mask = functional.adaptive_max_pool2d(mask, output_size)
    danger = torch.where(mask > 0, -depth, torch.full_like(depth, -torch.inf))
    _, indices = functional.adaptive_max_pool2d(danger, output_size, return_indices=True)
    depth_indices = indices.flatten(2)
    pooled_depth = depth.flatten(2).gather(2, depth_indices).reshape(depth.shape[0], 1, *output_size)
    velocity_indices = depth_indices.expand(-1, velocity.shape[1], -1)
    pooled_velocity = velocity.flatten(2).gather(2, velocity_indices).reshape(
        velocity.shape[0], velocity.shape[1], *output_size
    )
    pooled_sigma = sigma.flatten(2).gather(2, depth_indices).reshape(sigma.shape[0], 1, *output_size)
    present = pooled_mask > 0
    pooled_depth = torch.where(present, pooled_depth, torch.ones_like(pooled_depth))
    return pooled_depth, pooled_velocity * present, pooled_sigma * present


def build_rl_observation(
    current_range_normalized: torch.Tensor,
    mask_logits: torch.Tensor,
    relative_velocity: torch.Tensor,
    log_sigma: torch.Tensor,
    dyn_max_vel: float = 5.0,
    mask_threshold: float = 0.5,
    output_size: Tuple[int, int] = (6, 36),
):
    mask = (torch.sigmoid(mask_logits) > mask_threshold).to(current_range_normalized.dtype)
    static_depth = current_range_normalized * (1.0 - mask) + mask
    dynamic_depth = current_range_normalized * mask + (1.0 - mask)
    velocity = relative_velocity * mask
    sigma = torch.exp(log_sigma).mean(dim=1, keepdim=True) * mask
    static_depth = _minimum_pool(static_depth, output_size)
    dynamic_depth, velocity, sigma = _nearest_dynamic_pool(
        dynamic_depth, velocity, sigma, mask, output_size
    )
    dynamic = torch.cat(
        (
            dynamic_depth,
            (velocity[:, 0:1] / dyn_max_vel).clamp(-1.0, 1.0),
            (velocity[:, 1:2] / dyn_max_vel).clamp(-1.0, 1.0),
            (velocity[:, 2:3] / dyn_max_vel).clamp(-1.0, 1.0),
            sigma.clamp(0.0, 10.0) / 10.0,
        ),
        dim=1,
    )
    return static_depth.clamp(0.0, 1.0), dynamic.clamp(-1.0, 1.0)
