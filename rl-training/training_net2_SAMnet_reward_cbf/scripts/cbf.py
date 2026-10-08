"""Pure-Torch robust CBF-derived safety cost utilities."""

import torch


def compute_cbf_violation_cost(
    distance_m: torch.Tensor,
    relative_radial_velocity_mps: torch.Tensor,
    combined_uncertainty_mps: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    safe_distance_m: float,
    barrier_gain: float,
    confidence_scale: float,
    ttc_horizon_s: float,
    topk: int,
    topk_mean_weight: float,
) -> torch.Tensor:
    """Compute a robust CBF-derived cost from the frozen physical contract.

    ``relative_radial_velocity_mps`` is positive when separating and negative
    when approaching. ``combined_uncertainty_mps`` is the standard deviation
    after combining perception and ego-velocity uncertainty.
    """
    if distance_m.ndim != 3:
        raise ValueError("CBF fields must have shape (B, H, W)")
    if not (
        distance_m.shape
        == relative_radial_velocity_mps.shape
        == combined_uncertainty_mps.shape
        == valid_mask.shape
    ):
        raise ValueError("distance, radial velocity, uncertainty and valid mask must have identical shapes")
    if safe_distance_m <= 0.0:
        raise ValueError("safe_distance_m must be positive")
    if barrier_gain <= 0.0:
        raise ValueError("barrier_gain must be positive")
    if confidence_scale < 0.0:
        raise ValueError("confidence_scale must be non-negative")
    if ttc_horizon_s <= 0.0:
        raise ValueError("ttc_horizon_s must be positive")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if not 0.0 <= topk_mean_weight <= 1.0:
        raise ValueError("topk_mean_weight must be in [0, 1]")

    valid = valid_mask.to(dtype=torch.bool)
    sigma = combined_uncertainty_mps.clamp_min(0.0)

    # Lower confidence bound: a more negative value means a more dangerous
    # approach. The two terms both have physical units of m/s.
    robust_radial_velocity = relative_radial_velocity_mps - confidence_scale * sigma
    barrier = distance_m - safe_distance_m
    cbf_value = robust_radial_velocity + barrier_gain * barrier
    violation = torch.relu(-cbf_value)

    robust_approach_speed = torch.relu(-robust_radial_velocity)
    ttc = distance_m.clamp_min(0.0) / (robust_approach_speed + 1e-3)
    ttc_weight = torch.exp(-ttc / ttc_horizon_s)
    distance_weight = torch.exp(-distance_m.clamp_min(0.0) / safe_distance_m)
    pixel_cost = violation * (0.55 + 0.45 * ttc_weight) * (0.55 + 0.45 * distance_weight)
    pixel_cost = torch.where(valid, pixel_cost, torch.zeros_like(pixel_cost))

    flat_cost = pixel_cost.flatten(1)
    flat_valid = valid.flatten(1)
    k = min(int(topk), flat_cost.shape[1])
    topk_values = torch.topk(flat_cost, k=k, dim=1).values
    valid_count = flat_valid.sum(dim=1)
    topk_valid_count = valid_count.clamp(min=1, max=k).to(flat_cost.dtype)
    topk_mean = topk_values.sum(dim=1) / topk_valid_count
    top1 = topk_values[:, 0]
    cost = topk_mean_weight * topk_mean + (1.0 - topk_mean_weight) * top1
    return torch.where(valid_count > 0, cost, torch.zeros_like(cost)).unsqueeze(1)

def compute_dense_proximity_cost(
    distance_m: torch.Tensor,
    *,
    margin_m: float,
    topk: int,
) -> torch.Tensor:
    """Return a dense normalized near-obstacle cost using the closest rays.

    Distances at or beyond margin_m have zero cost. The cost rises linearly
    toward one as a surface approaches zero distance. Averaging only the
    largest topk ray costs prevents distant rays from diluting the signal.
    """
    if distance_m.ndim < 2:
        raise ValueError("distance_m must include batch and ray dimensions")
    if margin_m <= 0.0:
        raise ValueError("margin_m must be positive")
    if topk <= 0:
        raise ValueError("topk must be positive")

    flat_distance = distance_m.flatten(1).clamp_min(0.0)
    pixel_cost = torch.relu((margin_m - flat_distance) / margin_m).clamp_max(1.0)
    k = min(int(topk), pixel_cost.shape[1])
    return torch.topk(pixel_cost, k=k, dim=1).values.mean(dim=1, keepdim=True)

