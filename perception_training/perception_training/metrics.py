"""Dataset-level metrics for SA-UMNet perception evaluation."""

import math
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


_COMPONENTS = ("vr", "vtheta", "vphi")


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _confusion_metrics(counts: Dict[str, float]) -> Dict[str, float]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "iou": _safe_div(tp, tp + fp + fn),
        "fpr": _safe_div(fp, fp + tn),
        "accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
        "positive_support": tp + fn,
        "valid_support": tp + fp + fn + tn,
    }


def _new_confusion() -> Dict[str, float]:
    return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}


def _pearson(stats: Dict[str, float]) -> float:
    n = stats["count"]
    if n <= 1.0:
        return 0.0
    covariance = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / n
    variance_x = stats["sum_x2"] - stats["sum_x"] ** 2 / n
    variance_y = stats["sum_y2"] - stats["sum_y"] ** 2 / n
    denominator = math.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
    return covariance / denominator if denominator > 1e-12 else 0.0


class PerceptionMetricAccumulator:
    """Accumulate exact epoch-level metrics without retaining predictions.

    Static-speed P95 is approximated with a bounded histogram. All other
    metrics are computed from exact sufficient statistics accumulated over
    every valid pixel in the epoch.
    """

    def __init__(
        self,
        max_range_m: float = 10.0,
        mask_threshold: float = 0.5,
        distance_bins_m: Sequence[float] = (0.0, 2.0, 5.0, 10.0),
        pool_output_size: Tuple[int, int] = (6, 36),
        seam_width: int = 3,
        static_speed_hist_max_mps: float = 10.0,
        static_speed_hist_bins: int = 1024,
    ):
        bins = tuple(float(value) for value in distance_bins_m)
        if len(bins) < 2 or any(right <= left for left, right in zip(bins, bins[1:])):
            raise ValueError("distance_bins_m must contain at least two increasing edges")
        if max_range_m <= 0.0:
            raise ValueError("max_range_m must be positive")
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError("mask_threshold must be in (0, 1)")
        if seam_width < 1:
            raise ValueError("seam_width must be positive")
        if static_speed_hist_max_mps <= 0.0 or static_speed_hist_bins < 2:
            raise ValueError("static speed histogram settings must be positive")

        self.max_range_m = float(max_range_m)
        self.mask_threshold = float(mask_threshold)
        self.distance_bins_m = bins
        self.pool_output_size = tuple(int(value) for value in pool_output_size)
        self.seam_width = int(seam_width)
        self.static_speed_hist_max_mps = float(static_speed_hist_max_mps)
        self.static_speed_hist_bins = int(static_speed_hist_bins)
        self.reset()

    def reset(self) -> None:
        self.mask_counts = _new_confusion()
        self.distance_counts = {
            (left, right): _new_confusion()
            for left, right in zip(self.distance_bins_m, self.distance_bins_m[1:])
        }
        self.seam_counts = _new_confusion()
        self.velocity = {
            name: {"count": 0.0, "abs": 0.0, "sq": 0.0}
            for name in (*_COMPONENTS, "all")
        }
        self.calibration = {
            name: {
                "count": 0.0,
                "nll": 0.0,
                "sigma": 0.0,
                "covered_1": 0.0,
                "covered_2": 0.0,
                "sum_x": 0.0,
                "sum_y": 0.0,
                "sum_x2": 0.0,
                "sum_y2": 0.0,
                "sum_xy": 0.0,
            }
            for name in (*_COMPONENTS, "all")
        }
        self.static_speed_count = 0.0
        self.static_speed_sum = 0.0
        self.static_speed_hist = torch.zeros(self.static_speed_hist_bins, dtype=torch.float64)
        self.static_speed_overflow = 0.0
        self.pool_gt_cells = 0.0
        self.pool_detected_cells = 0.0
        self.pool_selected_target_cells = 0.0
        self.seam_pair_count = 0.0
        self.seam_prob_gap_sum = 0.0
        self.seam_prob_gap_error_sum = 0.0
        self.sample_count = 0.0

    @staticmethod
    def _update_confusion(
        counts: Dict[str, float], prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> None:
        counts["tp"] += float((prediction & target & valid).sum().item())
        counts["fp"] += float((prediction & ~target & valid).sum().item())
        counts["fn"] += float((~prediction & target & valid).sum().item())
        counts["tn"] += float((~prediction & ~target & valid).sum().item())

    @staticmethod
    def _update_error_stats(stats: Dict[str, float], error: torch.Tensor) -> None:
        if error.numel() == 0:
            return
        values = error.float()
        stats["count"] += float(values.numel())
        stats["abs"] += float(values.sum().item())
        stats["sq"] += float(values.square().sum().item())

    @staticmethod
    def _update_calibration_stats(
        stats: Dict[str, float], absolute_error: torch.Tensor, sigma: torch.Tensor
    ) -> None:
        if absolute_error.numel() == 0:
            return
        error = absolute_error.float()
        sigma = sigma.float().clamp(1e-4, 1e4)
        scale = sigma / math.sqrt(2.0)
        nll = error / scale + torch.log(2.0 * scale)
        stats["count"] += float(error.numel())
        stats["nll"] += float(nll.sum().item())
        stats["sigma"] += float(sigma.sum().item())
        stats["covered_1"] += float((error <= sigma).sum().item())
        stats["covered_2"] += float((error <= 2.0 * sigma).sum().item())
        stats["sum_x"] += float(error.sum().item())
        stats["sum_y"] += float(sigma.sum().item())
        stats["sum_x2"] += float(error.square().sum().item())
        stats["sum_y2"] += float(sigma.square().sum().item())
        stats["sum_xy"] += float((error * sigma).sum().item())

    @torch.no_grad()
    def update(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        mask_gt = batch["mask"].float()
        valid = batch.get("valid", torch.ones_like(mask_gt)).bool()
        target = mask_gt > 0.5
        probability = torch.sigmoid(pred["mask_logits"].float())
        prediction = probability > self.mask_threshold
        velocity_pred = pred["velocity"].float()
        velocity_gt = batch["velocity"].float()
        sigma = torch.exp(pred["log_sigma"].float()).clamp(1e-4, 1e4)
        dynamic = target & valid
        static = ~target & valid
        self.sample_count += float(mask_gt.shape[0])

        self._update_confusion(self.mask_counts, prediction, target, valid)

        depth_m = batch["input"][:, 0:1].float() * self.max_range_m
        for (left, right), counts in self.distance_counts.items():
            in_range = valid & (depth_m >= left) & (depth_m < right)
            self._update_confusion(counts, prediction, target, in_range)

        dynamic_velocity = dynamic.expand_as(velocity_gt)
        absolute_error = torch.abs(velocity_pred - velocity_gt)
        for channel, name in enumerate(_COMPONENTS):
            selected_error = absolute_error[:, channel : channel + 1][dynamic]
            selected_sigma = sigma[:, channel : channel + 1][dynamic]
            self._update_error_stats(self.velocity[name], selected_error)
            self._update_calibration_stats(self.calibration[name], selected_error, selected_sigma)
        all_error = absolute_error[dynamic_velocity]
        all_sigma = sigma[dynamic_velocity]
        self._update_error_stats(self.velocity["all"], all_error)
        self._update_calibration_stats(self.calibration["all"], all_error, all_sigma)

        static_speed = torch.linalg.vector_norm(velocity_pred, dim=1, keepdim=True)[static]
        if static_speed.numel() > 0:
            self.static_speed_count += float(static_speed.numel())
            self.static_speed_sum += float(static_speed.sum().item())
            clipped = static_speed.float().clamp(0.0, self.static_speed_hist_max_mps)
            histogram = torch.histc(
                clipped,
                bins=self.static_speed_hist_bins,
                min=0.0,
                max=self.static_speed_hist_max_mps,
            )
            self.static_speed_hist += histogram.cpu().double()
            self.static_speed_overflow += float((static_speed > self.static_speed_hist_max_mps).sum().item())

        height, width = target.shape[-2:]
        seam_width = min(self.seam_width, max(width // 2, 1))
        seam_columns = torch.zeros(width, dtype=torch.bool, device=target.device)
        seam_columns[:seam_width] = True
        seam_columns[-seam_width:] = True
        seam_valid = valid & seam_columns.view(1, 1, 1, width)
        self._update_confusion(self.seam_counts, prediction, target, seam_valid)

        pair_valid = valid[..., 0] & valid[..., -1]
        if pair_valid.any():
            pred_gap = torch.abs(probability[..., 0] - probability[..., -1])[pair_valid]
            target_gap = torch.abs(mask_gt[..., 0] - mask_gt[..., -1])[pair_valid]
            self.seam_pair_count += float(pred_gap.numel())
            self.seam_prob_gap_sum += float(pred_gap.sum().item())
            self.seam_prob_gap_error_sum += float(torch.abs(pred_gap - target_gap).sum().item())

        output_h = min(self.pool_output_size[0], height)
        output_w = min(self.pool_output_size[1], width)
        output_size = (output_h, output_w)
        gt_cells = F.adaptive_max_pool2d(dynamic.float(), output_size) > 0.5
        # Runtime pooling only accepts current valid returns. Apply the same
        # contract here so invalid predictions cannot inflate cell recall and
        # then fail the selected-valid check under a different denominator.
        pool_prediction = prediction & valid
        predicted_cells = F.adaptive_max_pool2d(pool_prediction.float(), output_size) > 0.5
        danger = torch.linalg.vector_norm(velocity_pred, dim=1, keepdim=True) * sigma.mean(dim=1, keepdim=True)
        danger = torch.where(pool_prediction, danger, torch.full_like(danger, -torch.inf))
        _, selected_indices = F.adaptive_max_pool2d(danger, output_size, return_indices=True)
        selected_target = target.flatten(2).gather(2, selected_indices.flatten(2)).view_as(gt_cells)
        selected_valid = valid.flatten(2).gather(2, selected_indices.flatten(2)).view_as(gt_cells)
        self.pool_gt_cells += float(gt_cells.sum().item())
        self.pool_detected_cells += float((gt_cells & predicted_cells).sum().item())
        self.pool_selected_target_cells += float(
            (gt_cells & predicted_cells & selected_target & selected_valid).sum().item()
        )

    def _static_speed_p95(self) -> float:
        if self.static_speed_count <= 0.0:
            return 0.0
        threshold = 0.95 * self.static_speed_count
        cumulative = torch.cumsum(self.static_speed_hist, dim=0)
        if cumulative[-1].item() < threshold:
            return self.static_speed_hist_max_mps
        index = int(torch.searchsorted(cumulative, torch.tensor(threshold, dtype=cumulative.dtype)).item())
        return self.static_speed_hist_max_mps * (index + 0.5) / self.static_speed_hist_bins

    def current_mask_iou(self) -> float:
        return _confusion_metrics(self.mask_counts)["iou"]

    def compute(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        mask = _confusion_metrics(self.mask_counts)
        for name, value in mask.items():
            result[f"metric/mask_{name}"] = float(value)
        # Backward-compatible names used by earlier logs and tooling.
        result["metric/iou"] = mask["iou"]
        result["metric/precision"] = mask["precision"]
        result["metric/recall"] = mask["recall"]
        result["metric/fpr"] = mask["fpr"]
        result["metric/mask_acc"] = mask["accuracy"]

        for (left, right), counts in self.distance_counts.items():
            label = f"{left:g}_{right:g}m".replace(".", "p")
            values = _confusion_metrics(counts)
            result[f"metric/distance_{label}_recall"] = values["recall"]
            result[f"metric/distance_{label}_iou"] = values["iou"]
            result[f"metric/distance_{label}_positive_support"] = values["positive_support"]

        for name, stats in self.velocity.items():
            result[f"metric/velocity_{name}_mae_mps"] = _safe_div(stats["abs"], stats["count"])
            result[f"metric/velocity_{name}_rmse_mps"] = math.sqrt(_safe_div(stats["sq"], stats["count"]))
            result[f"metric/velocity_{name}_support"] = stats["count"]
        result["metric/vel_mae"] = result["metric/velocity_all_mae_mps"]

        result["metric/static_speed_mean_mps"] = _safe_div(
            self.static_speed_sum, self.static_speed_count
        )
        result["metric/static_speed_p95_mps"] = self._static_speed_p95()
        result["metric/static_speed_support"] = self.static_speed_count
        result["metric/static_speed_hist_clip_fraction"] = _safe_div(
            self.static_speed_overflow, self.static_speed_count
        )

        for name, stats in self.calibration.items():
            count = stats["count"]
            result[f"metric/uncertainty_{name}_laplace_nll"] = _safe_div(stats["nll"], count)
            result[f"metric/uncertainty_{name}_mean_sigma_mps"] = _safe_div(stats["sigma"], count)
            result[f"metric/uncertainty_{name}_coverage_1sigma"] = _safe_div(stats["covered_1"], count)
            result[f"metric/uncertainty_{name}_coverage_2sigma"] = _safe_div(stats["covered_2"], count)
            result[f"metric/uncertainty_{name}_error_sigma_corr"] = _pearson(stats)

        result["metric/danger_pool_cell_recall"] = _safe_div(
            self.pool_detected_cells, self.pool_gt_cells
        )
        result["metric/danger_pool_selected_target_retention"] = _safe_div(
            self.pool_selected_target_cells, self.pool_gt_cells
        )
        result["metric/danger_pool_gt_cell_support"] = self.pool_gt_cells

        seam = _confusion_metrics(self.seam_counts)
        result["metric/seam_iou"] = seam["iou"]
        result["metric/seam_recall"] = seam["recall"]
        result["metric/seam_positive_support"] = seam["positive_support"]
        result["metric/seam_probability_gap"] = _safe_div(
            self.seam_prob_gap_sum, self.seam_pair_count
        )
        result["metric/seam_probability_gap_error"] = _safe_div(
            self.seam_prob_gap_error_sum, self.seam_pair_count
        )
        result["metric/evaluated_samples"] = self.sample_count
        return result


def composite_selection_score(
    metrics: Dict[str, float],
    dyn_max_vel_mps: float = 5.0,
    iou_weight: float = 1.0,
    recall_weight: float = 1.0,
    velocity_weight: float = 1.0,
    static_speed_weight: float = 1.0,
) -> float:
    """Checkpoint score balancing segmentation and physical velocity errors."""
    velocity_scale = max(float(dyn_max_vel_mps), 1e-6)
    return (
        float(iou_weight) * float(metrics.get("metric/mask_iou", 0.0))
        + float(recall_weight) * float(metrics.get("metric/mask_recall", 0.0))
        - float(velocity_weight)
        * float(metrics.get("metric/velocity_all_mae_mps", 0.0))
        / velocity_scale
        - float(static_speed_weight)
        * float(metrics.get("metric/static_speed_mean_mps", 0.0))
        / velocity_scale
    )
