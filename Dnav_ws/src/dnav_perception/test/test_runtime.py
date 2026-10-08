import math
import unittest

import numpy as np
import torch

from dnav_perception.geometry import PanoramaSpec, make_network_input
from dnav_perception.postprocess import build_rl_observation
from dnav_perception.runtime import PerceptionRuntime
from dnav_perception.scan_adapter import ScanSpec, ranges_to_sensor_points
from perception_training.geometry import make_network_input as reference_make_network_input
from perception_training.postprocess import build_rl_observation as reference_build_rl_observation


class FrozenLogicRegressionTest(unittest.TestCase):
    def test_geometry_matches_training(self):
        torch.manual_seed(7)
        points = torch.randn(1, 2880, 3)
        points = points / torch.linalg.norm(points, dim=-1, keepdim=True) * 5.0
        previous = points + torch.tensor([[[0.1, -0.2, 0.05]]])
        valid = torch.rand(1, 2880) > 0.1
        spec = PanoramaSpec()
        actual = make_network_input(points, previous, spec, valid, valid)
        expected = reference_make_network_input(points, previous, spec, curr_return_valid=valid, prev_return_valid=valid)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_postprocess_matches_training(self):
        torch.manual_seed(8)
        current = torch.rand(1, 1, 16, 180)
        logits = torch.randn(1, 1, 16, 180)
        velocity = torch.randn(1, 3, 16, 180)
        sigma = torch.randn(1, 3, 16, 180).clamp(-5.0, 3.0)
        actual = build_rl_observation(current, logits, velocity, sigma)
        expected = reference_build_rl_observation(
            current,
            logits,
            velocity,
            sigma,
            dyn_max_vel=5.0,
            mask_threshold=0.5,
            dynamic_channels=5,
            output_size=(6, 36),
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)


class RuntimeTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the deployment regression")
    def test_two_frames_and_reset(self):
        scan_spec = ScanSpec()
        ranges = np.full(5760, 5.0, dtype=np.float32)
        points, valid = ranges_to_sensor_points(
            ranges,
            -math.pi + math.pi / 360,
            2.0 * math.pi / 360,
            0.05,
            10.0,
            scan_spec,
        )
        runtime = PerceptionRuntime(
            "/home/amo/Safe-DNav/Dnav_ws/src/dnav_perception/models/perception_best.ts.pt",
            PanoramaSpec(),
            device="cuda",
        )
        pose = np.zeros(3, dtype=np.float32)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        velocity = np.zeros(3, dtype=np.float32)
        first = runtime.process(points, valid, 1.0, pose, quaternion, velocity)
        second = runtime.process(points, valid, 1.096, pose, quaternion, velocity)
        reset = runtime.process(points, valid, 1.5, pose, quaternion, velocity)
        self.assertEqual(first[0].shape, (1, 1, 6, 36))
        self.assertEqual(first[1].shape, (1, 5, 6, 36))
        self.assertFalse(first[2])
        self.assertTrue(second[2])
        self.assertFalse(reset[2])
        self.assertTrue(np.isfinite(second[0]).all())
        self.assertTrue(np.isfinite(second[1]).all())
        np.testing.assert_allclose(first[1][:, 0], 1.0)
        np.testing.assert_allclose(first[1][:, 1:], 0.0)


if __name__ == "__main__":
    unittest.main()
