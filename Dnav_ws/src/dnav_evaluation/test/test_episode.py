import math
import unittest

from dnav_evaluation.episode import EpisodeConfig, EpisodeMonitor, EpisodeOutcome


class EpisodeMonitorTest(unittest.TestCase):
    def test_success_path_metrics_and_spl(self):
        monitor = EpisodeMonitor(EpisodeConfig(goal=(2.0, 0.0, 1.0)))
        monitor.start(10.0, (0.0, 0.0, 1.0))
        self.assertEqual(monitor.update(11.0, (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)), EpisodeOutcome.RUNNING)
        self.assertEqual(monitor.update(12.0, (1.6, 0.0, 1.0), (0.6, 0.0, 0.0)), EpisodeOutcome.SUCCESS)
        result = monitor.result()
        self.assertAlmostEqual(result.path_length, 1.6)
        self.assertAlmostEqual(result.initial_distance, 2.0)
        self.assertAlmostEqual(result.final_distance, 0.4)
        self.assertAlmostEqual(result.spl, 1.0)
        self.assertAlmostEqual(result.max_acceleration, 0.4)

    def test_detour_reduces_spl(self):
        monitor = EpisodeMonitor(EpisodeConfig(goal=(2.0, 0.0, 1.0)))
        monitor.start(0.0, (0.0, 0.0, 1.0))
        monitor.update(1.0, (0.0, 1.0, 1.0), (0.0, 1.0, 0.0))
        monitor.update(2.0, (2.0, 1.0, 1.0), (1.0, 0.0, 0.0))
        monitor.update(3.0, (2.0, 0.4, 1.0), (0.0, -0.6, 0.0))
        self.assertAlmostEqual(monitor.result().spl, 2.0/3.6)

    def test_collision_classification_is_terminal(self):
        monitor = EpisodeMonitor(EpisodeConfig(goal=(2.0, 0.0, 1.0)))
        monitor.start(0.0, (0.0, 0.0, 1.0))
        self.assertEqual(monitor.collision(True, "person1"), EpisodeOutcome.COLLISION_DYNAMIC)
        self.assertEqual(monitor.collision(False, "wall"), EpisodeOutcome.COLLISION_DYNAMIC)

    def test_timeout(self):
        monitor = EpisodeMonitor(EpisodeConfig(goal=(2.0, 0.0, 1.0), timeout=2.0))
        monitor.start(4.0, (0.0, 0.0, 1.0))
        self.assertEqual(monitor.update(6.0, (0.1, 0.0, 1.0), (0.0, 0.0, 0.0)), EpisodeOutcome.TIMEOUT)

    def test_boundary_and_time_rollback(self):
        boundary = EpisodeMonitor(EpisodeConfig(goal=(2.0, 0.0, 1.0)))
        boundary.start(1.0, (0.0, 0.0, 1.0))
        self.assertEqual(boundary.update(2.0, (0.0, 0.0, 0.19), (0.0, 0.0, 0.0)), EpisodeOutcome.BOUNDARY_VIOLATION)
        rollback = EpisodeMonitor(EpisodeConfig(goal=(2.0, 0.0, 1.0)))
        rollback.start(2.0, (0.0, 0.0, 1.0))
        self.assertEqual(rollback.update(1.0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)), EpisodeOutcome.SYSTEM_FAULT)

    def test_clearance_speed_and_acceleration_extrema(self):
        monitor = EpisodeMonitor(EpisodeConfig(goal=(5.0, 0.0, 1.0)))
        monitor.start(0.0, (0.0, 0.0, 1.0))
        monitor.update(1.0, (0.2, 0.0, 1.0), (0.2, 0.0, -0.1), 2.0, 1.5, 1.0)
        monitor.update(2.0, (0.6, 0.0, 1.0), (0.4, 0.3, 0.2), 1.0, 0.8, 0.4)
        result = monitor.result()
        self.assertEqual(result.min_raw_clearance, 1.0)
        self.assertEqual(result.min_static_clearance, 0.8)
        self.assertEqual(result.min_dynamic_clearance, 0.4)
        self.assertAlmostEqual(result.max_horizontal_speed, 0.5)
        self.assertAlmostEqual(result.max_vertical_speed, 0.2)
        self.assertAlmostEqual(result.max_acceleration, math.sqrt(0.2**2+0.3**2+0.3**2))

    def test_invalid_configuration_and_samples(self):
        with self.assertRaises(ValueError):
            EpisodeConfig(goal=(0.0, 0.0, 0.0), goal_radius=0.0)
        monitor = EpisodeMonitor(EpisodeConfig(goal=(1.0, 0.0, 1.0)))
        monitor.start(0.0, (0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            monitor.update(1.0, (0.0, 0.0, 1.0), (float("nan"), 0.0, 0.0))
        with self.assertRaises(ValueError):
            monitor.update(1.0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), raw_clearance=-1.0)


if __name__ == "__main__":
    unittest.main()
