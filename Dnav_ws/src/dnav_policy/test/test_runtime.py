from pathlib import Path
import math
import unittest

import numpy as np

from dnav_policy.runtime import PolicyRuntime


CHECKPOINT = Path(__file__).resolve().parents[3] / "weights" / "policy_best.pt"


class PolicyRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = PolicyRuntime(CHECKPOINT, device="cpu")

    @staticmethod
    def inputs():
        static = np.ones((1, 6, 36), dtype=np.float32)
        dynamic = np.zeros((5, 6, 36), dtype=np.float32)
        dynamic[0] = 1.0
        state = np.zeros(13, dtype=np.float32)
        return static, dynamic, state

    def test_inference_is_finite_and_bounded(self):
        output = self.runtime.infer(*self.inputs(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(output.velocity_body.shape, (3,))
        self.assertEqual(output.velocity_world.shape, (3,))
        self.assertTrue(np.isfinite(output.velocity_world).all())
        self.assertTrue((np.abs(output.velocity_body) <= 3.0).all())
        self.assertGreaterEqual(output.latency_ms, 0.0)

    def test_world_velocity_uses_full_quaternion_rotation(self):
        half = math.sqrt(0.5)
        identity = self.runtime.infer(*self.inputs(), [1.0, 0.0, 0.0, 0.0])
        yaw = self.runtime.infer(*self.inputs(), [half, 0.0, 0.0, half])
        np.testing.assert_allclose(yaw.velocity_body, identity.velocity_body, atol=0.0)
        np.testing.assert_allclose(
            yaw.velocity_world,
            [-identity.velocity_body[1], identity.velocity_body[0], identity.velocity_body[2]],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_bad_shape_and_nonfinite_input_are_rejected(self):
        static, dynamic, state = self.inputs()
        with self.assertRaisesRegex(ValueError, "lidar_static"):
            self.runtime.infer(static[:, :, :-1], dynamic, state, [1.0, 0.0, 0.0, 0.0])
        state[0] = np.nan
        with self.assertRaisesRegex(ValueError, "state"):
            self.runtime.infer(static, dynamic, state, [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
