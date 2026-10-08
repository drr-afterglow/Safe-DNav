import math
import unittest

import numpy as np

from dnav_policy.geometry import (
    body_to_world,
    ros_xyzw_to_wxyz,
    world_to_body,
)
from dnav_policy.state_builder import PolicyStateBuilder


class PolicyGeometryTest(unittest.TestCase):
    def test_ros_quaternion_order_and_yaw_rotation(self):
        half = math.sqrt(0.5)
        quaternion = ros_xyzw_to_wxyz([0.0, 0.0, half, half])
        np.testing.assert_allclose(quaternion, [half, 0.0, 0.0, half], atol=1e-7)
        np.testing.assert_allclose(
            body_to_world(quaternion, [1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            world_to_body(quaternion, [1.0, 0.0, 0.0]),
            [0.0, -1.0, 0.0],
            atol=1e-7,
        )

    def test_invalid_quaternion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "norm"):
            body_to_world([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0])


class PolicyStateBuilderTest(unittest.TestCase):
    def test_field_order_normalization_and_previous_velocity(self):
        builder = PolicyStateBuilder()
        state = builder.build(
            position_world=[0.0, 0.0, 0.0],
            quaternion_world_body_wxyz=[1.0, 0.0, 0.0, 0.0],
            linear_velocity_body=[3.6, -7.2, 1.8],
            angular_velocity_body=[math.pi, -2.0 * math.pi, 4.0 * math.pi],
            goal_position_world=[10.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(state[0:3], [1.0, -1.0, 0.5])
        np.testing.assert_allclose(state[3:6], [0.5, -1.0, 1.0])
        np.testing.assert_allclose(state[6:9], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(
            float(state[9]), math.tanh(10.0 / builder.STATE_DISTANCE_SCALE), places=7
        )
        np.testing.assert_allclose(state[10:13], 0.0)

        second = builder.build(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(second[10:13], [1.0, -1.0, 0.5])
        builder.reset()
        reset = builder.build(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(reset[6:13], 0.0)

    def test_target_direction_is_in_body_frame(self):
        half = math.sqrt(0.5)
        builder = PolicyStateBuilder()
        state = builder.build(
            [0.0, 0.0, 0.0],
            [half, 0.0, 0.0, half],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(state[6:9], [0.0, -1.0, 0.0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
