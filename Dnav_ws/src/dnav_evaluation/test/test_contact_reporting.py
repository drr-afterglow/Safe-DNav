import csv
import json
import tempfile
import unittest
from pathlib import Path

from dnav_evaluation.contact import ContactKind, classify_vehicle_contact
from dnav_evaluation.reporting import write_episode_result, write_trajectory


class ContactAndReportingTest(unittest.TestCase):
    def test_contact_classification(self):
        self.assertEqual(
            classify_vehicle_contact(
                "quadcopter::base_link::collision",
                "person3::link::bottom",
                dynamic_tokens=("person",),
            ),
            ContactKind.DYNAMIC,
        )
        self.assertEqual(
            classify_vehicle_contact("quadcopter::base_link::collision", "person3::link::bottom"),
            ContactKind.STATIC,
        )
        self.assertEqual(
            classify_vehicle_contact("wall::link::collision", "quadcopter::base_link::collision"),
            ContactKind.STATIC,
        )
        self.assertIsNone(classify_vehicle_contact("wall::a", "ground::b"))
        self.assertIsNone(classify_vehicle_contact("quadcopter::a", "quadcopter::b"))

    def test_json_and_csv_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = write_episode_result(
                directory, "smoke", {"outcome": "success", "elapsed": 3.0}, run_id="r01"
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            with (Path(directory)/"scenario_results.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(payload["scenario"], "smoke")
        self.assertEqual(payload["run_id"], "r01")
        self.assertEqual(destination.name, "smoke__r01.json")
        self.assertEqual(payload["outcome"], "success")
        self.assertEqual(rows[0]["scenario"], "smoke")
        self.assertEqual(rows[0]["outcome"], "success")

    def test_trajectory_csv(self):
        rows = [
            {"time": 0.0, "x": 1.0, "controller_state": "ACTIVE"},
            {"time": 0.1, "x": 1.1, "controller_state": "ACTIVE"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = write_trajectory(directory, "smoke", rows, run_id="r01")
            with destination.open("r", encoding="utf-8", newline="") as stream:
                loaded = list(csv.DictReader(stream))
        self.assertEqual(destination.name, "smoke__r01__trajectory.csv")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[1]["x"], "1.1")


if __name__ == "__main__":
    unittest.main()
