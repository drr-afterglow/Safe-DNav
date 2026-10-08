import unittest

from dnav_evaluation.response import compute_response_metrics


class ResponseMetricsTest(unittest.TestCase):
    def test_response_metrics(self):
        samples = [
            {"time": 0.0, "phase": "accelerate", "phase_time": 0.0,
             "x": 0.0, "y": 0.0, "actual_vx": 0.0, "actual_vy": 0.0},
            {"time": 0.1, "phase": "accelerate", "phase_time": 0.1,
             "x": 0.0, "y": 0.0, "actual_vx": 0.05, "actual_vy": 0.0},
            {"time": 0.5, "phase": "accelerate", "phase_time": 0.5,
             "x": 0.1, "y": 0.0, "actual_vx": 0.45, "actual_vy": 0.0},
            {"time": 2.6, "phase": "accelerate", "phase_time": 2.6,
             "x": 1.1, "y": 0.0, "actual_vx": 0.5, "actual_vy": 0.0},
            {"time": 3.0, "phase": "brake", "phase_time": 0.0,
             "x": 1.3, "y": 0.0, "actual_vx": 0.5, "actual_vy": 0.0},
            {"time": 3.5, "phase": "brake", "phase_time": 0.5,
             "x": 1.4, "y": 0.0, "actual_vx": 0.04, "actual_vy": 0.0},
            {"time": 7.0, "phase": "turn", "phase_time": 0.0,
             "x": 2.0, "y": 0.0, "actual_vx": 0.5, "actual_vy": 0.0},
            {"time": 7.8, "phase": "turn", "phase_time": 0.8,
             "x": 2.2, "y": 0.2, "actual_vx": 0.04, "actual_vy": 0.46},
            {"time": 10.0, "phase": "final_brake", "phase_time": 0.0,
             "x": 2.2, "y": 1.2, "actual_vx": 0.0, "actual_vy": 0.5},
            {"time": 10.4, "phase": "final_brake", "phase_time": 0.4,
             "x": 2.2, "y": 1.3, "actual_vx": 0.0, "actual_vy": 0.04},
        ]
        result = compute_response_metrics(samples)
        self.assertAlmostEqual(result["rise_time_10_to_90"], 0.4)
        self.assertAlmostEqual(result["steady_forward_speed"], 0.5)
        self.assertAlmostEqual(result["first_brake"]["stopping_time"], 0.5)
        self.assertAlmostEqual(result["first_brake"]["stopping_distance"], 0.1)
        self.assertAlmostEqual(result["turn_settling_time"], 0.8)
        self.assertAlmostEqual(result["final_brake"]["stopping_time"], 0.4)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            compute_response_metrics([])


if __name__ == "__main__":
    unittest.main()
