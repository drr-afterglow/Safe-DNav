import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid = valid.to(value.dtype)
    return (value * valid).sum() / valid.sum().clamp_min(1.0)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, dice_weight: float = 1.0):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("focal alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.dice_weight = float(dice_weight)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        focal_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = target.float()
        valid = valid.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_raw = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        if focal_weight is None:
            focal = masked_mean(focal_raw, valid)
        else:
            focal_weight = focal_weight.to(focal_raw.dtype)
            weighted_valid = valid * focal_weight
            focal = (focal_raw * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
        intersection = (prob * target * valid).sum()
        denominator = (prob * valid).sum() + (target * valid).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        return focal + self.dice_weight * dice


class PerceptionLoss(nn.Module):
    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_kin: float = 1.0,
        lambda_static: float = 0.1,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
        far_positive_weight: float = 1.0,
        far_distance_start_m: float = 5.0,
        max_range_m: float = 10.0,
        velocity_component_weights=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_kin = lambda_kin
        self.lambda_static = lambda_static
        if far_positive_weight < 1.0:
            raise ValueError("far_positive_weight must be >= 1")
        if not 0.0 <= far_distance_start_m <= max_range_m:
            raise ValueError("far_distance_start_m must be within sensor range")
        self.seg_loss = FocalDiceLoss(focal_alpha, focal_gamma, dice_weight)
        self.far_positive_weight = float(far_positive_weight)
        self.far_distance_start_m = float(far_distance_start_m)
        self.max_range_m = float(max_range_m)
        component_weights = torch.as_tensor(velocity_component_weights, dtype=torch.float32)
        if component_weights.numel() != 3 or not torch.isfinite(component_weights).all():
            raise ValueError("velocity_component_weights must contain three finite values")
        if (component_weights <= 0).any():
            raise ValueError("velocity_component_weights must be positive")
        component_weights = component_weights / component_weights.mean()
        self.register_buffer(
            "velocity_component_weights",
            component_weights.reshape(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = batch["mask"].float()
        velocity_gt = batch["velocity"].float()
        valid = batch.get("valid", torch.ones_like(mask)).float()
        dyn = (mask * valid).expand_as(velocity_gt)
        static = ((1.0 - mask) * valid).expand_as(velocity_gt)

        focal_weight = None
        if self.far_positive_weight > 1.0:
            current_depth_m = batch["input"][:, 0:1].float() * self.max_range_m
            far_positive = (mask > 0.5) & (current_depth_m >= self.far_distance_start_m)
            focal_weight = torch.ones_like(mask)
            focal_weight = torch.where(
                far_positive,
                torch.full_like(focal_weight, self.far_positive_weight),
                focal_weight,
            )
        seg = self.seg_loss(pred["mask_logits"], mask, valid, focal_weight=focal_weight)
        sigma_std = torch.exp(pred["log_sigma"]).clamp_min(1e-4)
        laplace_scale = sigma_std / math.sqrt(2.0)
        kin_raw = torch.abs(velocity_gt - pred["velocity"]) / laplace_scale + torch.log(laplace_scale)
        weighted_kin_raw = kin_raw * self.velocity_component_weights.to(kin_raw.dtype)
        kin = (weighted_kin_raw * dyn).sum() / dyn.sum().clamp_min(1.0)
        static_zero = (torch.abs(pred["velocity"]) * static).sum() / static.sum().clamp_min(1.0)
        total = self.lambda_seg * seg + self.lambda_kin * kin + self.lambda_static * static_zero
        return {
            "loss": total,
            "loss_seg": seg.detach(),
            "loss_kin": kin.detach(),
            "loss_static": static_zero.detach(),
        }


@torch.no_grad()
def perception_metrics(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    mask_gt = batch["mask"].float()
    valid = batch.get("valid", torch.ones_like(mask_gt)).bool()
    mask_pred = torch.sigmoid(pred["mask_logits"]) > threshold
    target = mask_gt > 0.5
    tp = (mask_pred & target & valid).sum().float()
    fp = (mask_pred & ~target & valid).sum().float()
    fn = (~mask_pred & target & valid).sum().float()
    tn = (~mask_pred & ~target & valid).sum().float()
    dyn = (target & valid).expand_as(batch["velocity"])
    vel_mae = (
        torch.abs(pred["velocity"][dyn] - batch["velocity"][dyn]).mean()
        if dyn.any()
        else torch.zeros((), device=mask_gt.device)
    )
    return {
        "metric/iou": tp / (tp + fp + fn).clamp_min(1.0),
        "metric/precision": tp / (tp + fp).clamp_min(1.0),
        "metric/recall": tp / (tp + fn).clamp_min(1.0),
        "metric/fpr": fp / (fp + tn).clamp_min(1.0),
        "metric/mask_acc": (tp + tn) / (tp + fp + fn + tn).clamp_min(1.0),
        "metric/vel_mae": vel_mae,
    }
