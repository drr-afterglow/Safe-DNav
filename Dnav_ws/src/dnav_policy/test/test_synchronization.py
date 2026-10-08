import unittest

import numpy as np

from dnav_policy.synchronization import (
    GoalSample,
    InputUnavailable,
    OdometrySample,
    PerceptionSample,
    PolicyInputCache,
)


class PolicyInputCacheTest(unittest.TestCase):
    def populated(self):
        cache = PolicyInputCache()
        cache.update_perception(
            PerceptionSample(1.0, np.ones((1, 6, 36)), np.zeros((5, 6, 36)), True)
        )
        cache.update_odometry(
            OdometrySample(
                1.0,
                np.zeros(3),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(3),
            )
        )
        cache.update_goal(GoalSample(1.0, np.ones(3)))
        return cache

    def test_complete_fresh_snapshot(self):
        cache = self.populated()
        perception, odometry, goal, generation = cache.snapshot(
            1.05, perception_timeout=0.25, odometry_timeout=0.1
        )
        self.assertTrue(perception.history_valid)
        self.assertEqual(odometry.stamp, 1.0)
        np.testing.assert_array_equal(goal.position_world, np.ones(3))
        self.assertEqual(generation, 1)

    def test_missing_and_stale_inputs_are_rejected(self):
        with self.assertRaisesRegex(InputUnavailable, "perception"):
            PolicyInputCache().snapshot(
                1.0, perception_timeout=0.25, odometry_timeout=0.1
            )
        cache = self.populated()
        with self.assertRaisesRegex(InputUnavailable, "perception stale"):
            cache.snapshot(1.3, perception_timeout=0.25, odometry_timeout=0.5)

    def test_new_goal_and_clock_rollback_advance_generation(self):
        cache = self.populated()
        first = cache.snapshot(1.05, perception_timeout=1.0, odometry_timeout=1.0)[3]
        cache.update_goal(GoalSample(1.1, np.full(3, 2.0)))
        second = cache.snapshot(1.1, perception_timeout=1.0, odometry_timeout=1.0)[3]
        with self.assertRaisesRegex(InputUnavailable, "perception stale"):
            cache.snapshot(0.9, perception_timeout=1.0, odometry_timeout=1.0)
        third = cache.generation
        self.assertGreater(second, first)
        self.assertGreater(third, second)


if __name__ == "__main__":
    unittest.main()
