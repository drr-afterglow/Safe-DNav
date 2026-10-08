import math
import unittest

import numpy as np

from dnav_perception.pose_buffer import PoseBuffer, PoseSample
from dnav_perception.scan_adapter import ScanSpec, ranges_to_sensor_points, transform_points


class ScanAdapterTest(unittest.TestCase):
    def test_shape_validity_and_cell_centers(self):
        spec = ScanSpec()
        ranges = np.full(spec.height * spec.width, 5.0, dtype=np.float32)
        ranges[0] = np.inf
        angle_min = -math.pi + math.pi / spec.width
        increment = 2.0 * math.pi / spec.width
        points, valid = ranges_to_sensor_points(ranges, angle_min, increment, 0.05, 10.0, spec)
        self.assertEqual(points.shape, (5760, 3))
        self.assertEqual(valid.shape, (5760,))
        self.assertFalse(valid[0])
        self.assertTrue(valid[1:].all())
        self.assertAlmostEqual(float(np.linalg.norm(points[1])), 5.0, places=5)

    def test_transform_translation(self):
        transformed = transform_points(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([0.1, -0.2, 0.3]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(transformed, [[1.1, -0.2, 0.3]], atol=1e-6)


class PoseBufferTest(unittest.TestCase):
    def test_linear_and_quaternion_interpolation(self):
        buffer = PoseBuffer()
        buffer.add(PoseSample(0.0, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)))
        buffer.add(
            PoseSample(
                0.1,
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.array([2.0, 0.0, 0.0]),
            )
        )
        sample = buffer.interpolate(0.05, tolerance=0.06)
        self.assertIsNotNone(sample)
        np.testing.assert_allclose(sample.position, [0.5, 1.0, 1.5])
        np.testing.assert_allclose(sample.linear_velocity_body, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(sample.quaternion_wxyz)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
