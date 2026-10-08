import copy
from pathlib import Path
import unittest

import torch

from dnav_policy.checkpoint import (
    FINAL_POLICY_SHA256,
    SOURCE_SHAPES,
    load_policy_checkpoint,
    sha256_file,
    validate_source_state_dict,
)
from dnav_policy.model import PolicyActor


CHECKPOINT = Path(__file__).resolve().parents[3] / "weights" / "policy_best.pt"


class PolicyModelCheckpointTest(unittest.TestCase):
    def test_final_checkpoint_loads_strictly_and_runs(self):
        self.assertEqual(sha256_file(CHECKPOINT), FINAL_POLICY_SHA256)
        model = PolicyActor()
        load_policy_checkpoint(model, CHECKPOINT)
        model.eval()
        static = torch.ones(2, 1, 6, 36)
        dynamic = torch.zeros(2, 5, 6, 36)
        dynamic[:, 0] = 1.0
        state = torch.zeros(2, 13)
        with torch.inference_mode():
            alpha, beta = model.distribution_parameters(static, dynamic, state)
            velocity = model(static, dynamic, state)
        self.assertEqual(tuple(alpha.shape), (2, 3))
        self.assertEqual(tuple(beta.shape), (2, 3))
        self.assertTrue(bool(torch.isfinite(velocity).all()))
        self.assertTrue(bool((velocity.abs() <= 3.0).all()))

    def test_missing_source_key_is_rejected(self):
        state = {
            key: torch.zeros(shape) if shape else torch.tensor(0.0)
            for key, shape in SOURCE_SHAPES.items()
        }
        broken = copy.copy(state)
        broken.pop(next(iter(broken)))
        with self.assertRaisesRegex(ValueError, "checkpoint key mismatch"):
            validate_source_state_dict(broken)

    def test_input_shape_is_checked(self):
        model = PolicyActor()
        with self.assertRaisesRegex(ValueError, "lidar_static"):
            model(torch.zeros(1, 1, 5, 36), torch.zeros(1, 5, 6, 36), torch.zeros(1, 13))


if __name__ == "__main__":
    unittest.main()
