"""Validated YAML scenario definitions."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Tuple

import yaml

from .episode import EpisodeConfig, Vector3, _vector3


@dataclass(frozen=True)
class Scenario:
    name: str
    world: str
    start: Vector3
    yaw: float
    episode: EpisodeConfig
    max_horizontal_speed: float
    dynamic_model_tokens: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.world:
            raise ValueError("scenario name and world are required")
        if not math.isfinite(self.yaw):
            raise ValueError("scenario yaw must be finite")
        if not math.isfinite(self.max_horizontal_speed) or self.max_horizontal_speed <= 0.0:
            raise ValueError("maximum horizontal speed must be positive")


def load_scenarios(filename: str) -> List[Scenario]:
    path = Path(filename)
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("scenarios"), list):
        raise ValueError("scenario file must contain a scenarios list")
    scenarios: List[Scenario] = []
    names = set()
    for data in document["scenarios"]:
        if not isinstance(data, dict):
            raise ValueError("each scenario must be a mapping")
        name = str(data.get("name", ""))
        if name in names:
            raise ValueError(f"duplicate scenario name: {name}")
        names.add(name)
        goal = _vector3(data["goal"], "goal")
        scenarios.append(Scenario(
            name=name,
            world=str(data.get("world", "")),
            start=_vector3(data["start"], "start"),
            yaw=float(data.get("yaw", 0.0)),
            episode=EpisodeConfig(
                goal=goal,
                goal_radius=float(data.get("goal_radius", 0.5)),
                timeout=float(data.get("timeout", 35.2)),
                min_altitude=float(data.get("min_altitude", 0.2)),
                max_altitude=float(data.get("max_altitude", 4.5)),
            ),
            max_horizontal_speed=float(data.get("max_horizontal_speed", 0.5)),
            dynamic_model_tokens=tuple(str(item) for item in data.get("dynamic_model_tokens", [])),
        ))
    if not scenarios:
        raise ValueError("scenario file must not be empty")
    return scenarios


def validate_world_file(configured_world: str, scenario_world: str, gazebo_package: str) -> Path:
    """Require the launched Gazebo world to exactly match the selected scenario."""
    configured = Path(configured_world).expanduser()
    expected = Path(gazebo_package)/scenario_world
    if not configured.is_file():
        raise ValueError(f"configured Gazebo world does not exist: {configured}")
    if not expected.is_file():
        raise ValueError(f"scenario Gazebo world does not exist: {expected}")
    configured = configured.resolve()
    expected = expected.resolve()
    if configured != expected:
        raise ValueError(
            f"launched Gazebo world does not match scenario: {configured} != {expected}"
        )
    return configured


def validate_spawn_configuration(configured_start: Vector3, configured_yaw: float,
                                 scenario: Scenario, tolerance: float = 1e-6) -> None:
    """Reject launches whose Gazebo spawn pose differs from the selected scenario."""
    start = _vector3(configured_start, "configured spawn position")
    yaw = float(configured_yaw)
    if not math.isfinite(yaw) or not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("spawn yaw and tolerance must be finite")
    if any(abs(actual-expected) > tolerance
           for actual, expected in zip(start, scenario.start)):
        raise ValueError(
            f"configured Gazebo spawn position does not match scenario: "
            f"{start} != {scenario.start}"
        )
    yaw_error = math.atan2(math.sin(yaw-scenario.yaw), math.cos(yaw-scenario.yaw))
    if abs(yaw_error) > tolerance:
        raise ValueError(
            f"configured Gazebo spawn yaw does not match scenario: "
            f"{yaw} != {scenario.yaw}"
        )
