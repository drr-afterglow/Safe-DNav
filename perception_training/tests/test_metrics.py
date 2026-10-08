import math
import unittest

import torch

from perception_training.metrics import PerceptionMetricAccumulator, composite_selection_score


class PerceptionMetricAccumulatorTest(unittest.TestCase):
    def test_perfect_predictions_cover_all_metric_families(self):
        target = torch.tensor([[[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]]])
        valid = torch.ones_like(target)
        depth = torch.tensor([[[[0.1, 0.3, 0.6, 0.9], [0.15, 0.4, 0.7, 0.95]]]])
        network_input = torch.cat([depth, depth, valid, valid], dim=1)
        velocity_gt = torch.zeros(1, 3, 2, 4)
        dynamic = target.bool().expand_as(velocity_gt)
        values = torch.tensor([1.0, -2.0, 0.5]).view(1, 3, 1, 1).expand_as(velocity_gt)
        velocity_gt[dynamic] = values[dynamic]
        logits = torch.where(target.bool(), torch.full_like(target, 20.0), torch.full_like(target, -20.0))
        pred = {
            "mask_logits": logits,
            "velocity": velocity_gt.clone(),
            "log_sigma": torch.zeros_like(velocity_gt),
        }
        batch = {
            "input": network_input,
            "mask": target,
            "valid": valid,
            "velocity": velocity_gt,
        }

        accumulator = PerceptionMetricAccumulator(
            max_range_m=10.0,
            distance_bins_m=(0.0, 2.0, 5.0, 10.0),
            pool_output_size=(1, 2),
            seam_width=1,
            static_speed_hist_bins=100,
        )
        accumulator.update(pred, batch)
        metrics = accumulator.compute()

        for key in ("mask_precision", "mask_recall", "mask_f1", "mask_iou"):
            self.assertAlmostEqual(metrics[f"metric/{key}"], 1.0)
        for component in ("vr", "vtheta", "vphi", "all"):
            self.assertAlmostEqual(metrics[f"metric/velocity_{component}_mae_mps"], 0.0)
            self.assertAlmostEqual(metrics[f"metric/velocity_{component}_rmse_mps"], 0.0)
        self.assertAlmostEqual(metrics["metric/uncertainty_all_coverage_1sigma"], 1.0)
        self.assertAlmostEqual(metrics["metric/uncertainty_all_coverage_2sigma"], 1.0)
        self.assertTrue(math.isfinite(metrics["metric/uncertainty_all_laplace_nll"]))
        self.assertAlmostEqual(metrics["metric/danger_pool_cell_recall"], 1.0)
        self.assertAlmostEqual(metrics["metric/danger_pool_selected_target_retention"], 1.0)
        self.assertAlmostEqual(metrics["metric/static_speed_mean_mps"], 0.0)
        self.assertLess(metrics["metric/static_speed_p95_mps"], 0.11)
        self.assertGreater(metrics["metric/distance_0_2m_positive_support"], 0.0)
        self.assertGreater(metrics["metric/distance_2_5m_positive_support"], 0.0)
        self.assertGreater(metrics["metric/distance_5_10m_positive_support"], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_epoch_aggregation_uses_global_counts(self):
        accumulator = PerceptionMetricAccumulator(
            max_range_m=10.0,
            distance_bins_m=(0.0, 10.0),
            pool_output_size=(1, 4),
        )
        for target, logits in (
            (torch.tensor([[[[1.0]]]]), torch.tensor([[[[20.0]]]])),
            (torch.ones(1, 1, 1, 3), torch.full((1, 1, 1, 3), -20.0)),
        ):
            valid = torch.ones_like(target)
            width = target.shape[-1]
            batch = {
                "input": torch.cat(
                    [
                        torch.full_like(target, 0.5),
                        torch.full_like(target, 0.5),
                        valid,
                        valid,
                    ],
                    dim=1,
                ),
                "mask": target,
                "valid": valid,
                "velocity": torch.zeros(1, 3, 1, width),
            }
            pred = {
                "mask_logits": logits,
                "velocity": torch.zeros(1, 3, 1, width),
                "log_sigma": torch.zeros(1, 3, 1, width),
            }
            accumulator.update(pred, batch)
        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["metric/mask_recall"], 0.25)
        self.assertAlmostEqual(metrics["metric/mask_precision"], 1.0)
        self.assertEqual(metrics["metric/mask_positive_support"], 4.0)
        self.assertEqual(metrics["metric/evaluated_samples"], 2.0)

    def test_empty_dynamic_batch_is_finite(self):
        target = torch.zeros(2, 1, 2, 4)
        valid = torch.ones_like(target)
        batch = {
            "input": torch.cat([torch.ones(2, 2, 2, 4), valid, valid], dim=1),
            "mask": target,
            "valid": valid,
            "velocity": torch.zeros(2, 3, 2, 4),
        }
        pred = {
            "mask_logits": torch.full_like(target, -20.0),
            "velocity": torch.zeros(2, 3, 2, 4),
            "log_sigma": torch.zeros(2, 3, 2, 4),
        }
        accumulator = PerceptionMetricAccumulator(pool_output_size=(1, 2))
        accumulator.update(pred, batch)
        metrics = accumulator.compute()
        self.assertEqual(metrics["metric/mask_recall"], 0.0)
        self.assertEqual(metrics["metric/velocity_all_support"], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_composite_checkpoint_score_penalizes_velocity_errors(self):
        good = {
            "metric/mask_iou": 0.7,
            "metric/mask_recall": 0.9,
            "metric/velocity_all_mae_mps": 0.2,
            "metric/static_speed_mean_mps": 0.1,
        }
        bad = dict(good)
        bad["metric/velocity_all_mae_mps"] = 1.2
        self.assertGreater(
            composite_selection_score(good, dyn_max_vel_mps=5.0),
            composite_selection_score(bad, dyn_max_vel_mps=5.0),
        )


if __name__ == "__main__":
    unittest.main()
