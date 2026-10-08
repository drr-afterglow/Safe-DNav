import sys
import unittest
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from cbf import compute_cbf_violation_cost, compute_dense_proximity_cost


class TestCBFContract(unittest.TestCase):
    def cost(self, distance, radial, sigma, valid=None):
        distance = torch.as_tensor(distance, dtype=torch.float32).reshape(1, 1, -1)
        radial = torch.as_tensor(radial, dtype=torch.float32).reshape_as(distance)
        sigma = torch.as_tensor(sigma, dtype=torch.float32).reshape_as(distance)
        if valid is None:
            valid = torch.ones_like(distance, dtype=torch.bool)
        else:
            valid = torch.as_tensor(valid, dtype=torch.bool).reshape_as(distance)
        return compute_cbf_violation_cost(
            distance,
            radial,
            sigma,
            valid,
            safe_distance_m=0.5,
            barrier_gain=1.2,
            confidence_scale=2.0,
            ttc_horizon_s=1.2,
            topk=16,
            topk_mean_weight=0.7,
        ).item()

    def test_no_dynamic_pixels_has_zero_cost(self):
        self.assertEqual(self.cost([0.1], [-3.0], [1.0], [False]), 0.0)

    def test_far_separating_obstacle_has_zero_cost(self):
        self.assertEqual(self.cost([3.0], [1.0], [0.0]), 0.0)

    def test_closer_and_faster_approach_increases_cost(self):
        moderate = self.cost([1.0], [-1.0], [0.0])
        dangerous = self.cost([0.4], [-2.0], [0.0])
        self.assertGreater(dangerous, moderate)
        self.assertGreater(moderate, 0.0)

    def test_uncertainty_increases_cost(self):
        certain = self.cost([1.0], [-0.4], [0.0])
        uncertain = self.cost([1.0], [-0.4], [0.5])
        self.assertGreater(uncertain, certain)

    def test_topk_mean_is_not_diluted_by_invalid_pixels(self):
        single = self.cost([0.4], [-1.0], [0.0])
        padded = self.cost(
            [0.4] + [0.1] * 31,
            [-1.0] + [-5.0] * 31,
            [0.0] * 32,
            [True] + [False] * 31,
        )
        self.assertAlmostEqual(single, padded, places=6)


    def test_dense_proximity_cost_activates_before_collision(self):
        far = torch.full((1, 1, 2, 4), 2.0)
        near = far.clone()
        near[..., 0] = 1.0
        far_cost = compute_dense_proximity_cost(far, margin_m=1.5, topk=4)
        near_cost = compute_dense_proximity_cost(near, margin_m=1.5, topk=4)
        self.assertEqual(far_cost.item(), 0.0)
        self.assertGreater(near_cost.item(), 0.0)

    def test_dense_proximity_cost_increases_when_closer(self):
        moderate = compute_dense_proximity_cost(
            torch.full((1, 1, 2, 4), 1.0), margin_m=1.5, topk=8
        )
        dangerous = compute_dense_proximity_cost(
            torch.full((1, 1, 2, 4), 0.3), margin_m=1.5, topk=8
        )
        self.assertGreater(dangerous.item(), moderate.item())

    def test_dense_proximity_topk_is_not_diluted_by_far_rays(self):
        compact = torch.tensor([[0.3, 0.4, 0.5, 0.6]])
        padded = torch.cat([compact, torch.full((1, 60), 10.0)], dim=1)
        compact_cost = compute_dense_proximity_cost(compact, margin_m=1.5, topk=4)
        padded_cost = compute_dense_proximity_cost(padded, margin_m=1.5, topk=4)
        self.assertAlmostEqual(compact_cost.item(), padded_cost.item(), places=6)


if __name__ == '__main__':
    unittest.main()
