import math
import tempfile
import unittest
from pathlib import Path

from dnav_evaluation.scenario import (
    load_scenarios,
    validate_spawn_configuration,
    validate_world_file,
)


class ScenarioTest(unittest.TestCase):
    def test_load_checked_scenario(self):
        text = """scenarios:
  - name: smoke
    world: worlds/test/empty.world
    start: [0, 0, 0.1]
    yaw: 1.0
    goal: [2, 0, 1]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scenario.yaml"
            path.write_text(text, encoding="utf-8")
            scenarios = load_scenarios(str(path))
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0].name, "smoke")
        self.assertEqual(scenarios[0].episode.timeout, 35.2)
        self.assertEqual(scenarios[0].max_horizontal_speed, 0.5)

    def test_duplicate_names_are_rejected(self):
        text = """scenarios:
  - {name: repeated, world: a.world, start: [0, 0, 1], goal: [1, 0, 1]}
  - {name: repeated, world: b.world, start: [0, 0, 1], goal: [1, 0, 1]}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scenario.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scenarios(str(path))

    def test_world_file_must_exist_and_match_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root/"worlds/test/empty.world"
            expected.parent.mkdir(parents=True)
            expected.write_text("<sdf/>", encoding="utf-8")
            self.assertEqual(
                validate_world_file(str(expected), "worlds/test/empty.world", str(root)),
                expected.resolve(),
            )
            other = root/"worlds/test/other.world"
            other.write_text("<sdf/>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match scenario"):
                validate_world_file(str(other), "worlds/test/empty.world", str(root))
            with self.assertRaisesRegex(ValueError, "does not exist"):
                validate_world_file(str(root/"missing.world"), "worlds/test/empty.world", str(root))

    def test_spawn_configuration_must_match_scenario(self):
        text = """scenarios:
  - {name: smoke, world: empty.world, start: [-5, -5, 0.1], yaw: 1.0, goal: [1, 1, 1]}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scenario.yaml"
            path.write_text(text, encoding="utf-8")
            scenario = load_scenarios(str(path))[0]
        validate_spawn_configuration(scenario.start, scenario.yaw+2.0*math.pi, scenario)
        with self.assertRaisesRegex(ValueError, "spawn position"):
            validate_spawn_configuration((0.0, 0.0, 0.1), scenario.yaw, scenario)
        with self.assertRaisesRegex(ValueError, "spawn yaw"):
            validate_spawn_configuration(scenario.start, scenario.yaw+0.1, scenario)


if __name__ == "__main__":
    unittest.main()
