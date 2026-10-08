"""ROS-independent episode state machine and navigation metrics."""

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Dict, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]


class EpisodeOutcome(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    COLLISION_STATIC = "collision_static"
    COLLISION_DYNAMIC = "collision_dynamic"
    BOUNDARY_VIOLATION = "boundary_violation"
    TIMEOUT = "timeout"
    SYSTEM_FAULT = "system_fault"


@dataclass(frozen=True)
class EpisodeConfig:
    goal: Vector3
    goal_radius: float = 0.5
    timeout: float = 35.2
    min_altitude: float = 0.2
    max_altitude: float = 4.5

    def __post_init__(self) -> None:
        values = (*self.goal, self.goal_radius, self.timeout,
                  self.min_altitude, self.max_altitude)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("episode configuration must contain finite values")
        if self.goal_radius <= 0.0 or self.timeout <= 0.0:
            raise ValueError("goal radius and timeout must be positive")
        if self.min_altitude >= self.max_altitude:
            raise ValueError("minimum altitude must be below maximum altitude")


@dataclass(frozen=True)
class EpisodeResult:
    outcome: EpisodeOutcome
    elapsed: float
    initial_distance: float
    final_distance: float
    path_length: float
    spl: float
    min_raw_clearance: Optional[float]
    min_static_clearance: Optional[float]
    min_dynamic_clearance: Optional[float]
    max_horizontal_speed: float
    max_vertical_speed: float
    max_acceleration: float
    samples: int
    detail: str
    start_position: Vector3
    final_position: Vector3

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["outcome"] = self.outcome.value
        return result


def _vector3(value: Sequence[float], name: str) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three elements")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def _distance(left: Vector3, right: Vector3) -> float:
    return math.sqrt(sum((a-b)**2 for a, b in zip(left, right)))


def _minimum(current: Optional[float], value: Optional[float]) -> Optional[float]:
    if value is None:
        return current
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("clearance must be finite and non-negative")
    return value if current is None else min(current, value)


class EpisodeMonitor:
    """Accumulates one episode and applies terminal conditions deterministically."""

    def __init__(self, config: EpisodeConfig):
        self.config = config
        self.started = False
        self.outcome = EpisodeOutcome.RUNNING
        self.detail = ""
        self.start_time = 0.0
        self.last_time = 0.0
        self.start_position: Vector3 = (0.0, 0.0, 0.0)
        self.last_position: Vector3 = self.start_position
        self.last_velocity: Optional[Vector3] = None
        self.path_length = 0.0
        self.samples = 0
        self.min_raw_clearance: Optional[float] = None
        self.min_static_clearance: Optional[float] = None
        self.min_dynamic_clearance: Optional[float] = None
        self.max_horizontal_speed = 0.0
        self.max_vertical_speed = 0.0
        self.max_acceleration = 0.0

    def start(self, timestamp: float, position: Sequence[float]) -> None:
        if self.started:
            raise RuntimeError("episode has already started")
        if not math.isfinite(timestamp):
            raise ValueError("start timestamp must be finite")
        self.start_position = _vector3(position, "position")
        self.last_position = self.start_position
        self.start_time = self.last_time = float(timestamp)
        self.started = True

    def update(self, timestamp: float, position: Sequence[float], velocity: Sequence[float],
               raw_clearance: Optional[float] = None,
               static_clearance: Optional[float] = None,
               dynamic_clearance: Optional[float] = None,
               system_fault: Optional[str] = None) -> EpisodeOutcome:
        if not self.started:
            raise RuntimeError("episode has not started")
        if self.outcome != EpisodeOutcome.RUNNING:
            return self.outcome
        timestamp = float(timestamp)
        current_position = _vector3(position, "position")
        current_velocity = _vector3(velocity, "velocity")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        dt = timestamp-self.last_time
        if dt < 0.0:
            return self._finish(EpisodeOutcome.SYSTEM_FAULT, "simulation time moved backwards")

        self.path_length += _distance(current_position, self.last_position)
        self.min_raw_clearance = _minimum(self.min_raw_clearance, raw_clearance)
        self.min_static_clearance = _minimum(self.min_static_clearance, static_clearance)
        self.min_dynamic_clearance = _minimum(self.min_dynamic_clearance, dynamic_clearance)
        self.max_horizontal_speed = max(
            self.max_horizontal_speed, math.hypot(current_velocity[0], current_velocity[1])
        )
        self.max_vertical_speed = max(self.max_vertical_speed, abs(current_velocity[2]))
        if self.last_velocity is not None and dt > 0.0:
            self.max_acceleration = max(
                self.max_acceleration,
                _distance(current_velocity, self.last_velocity)/dt,
            )
        self.last_time = timestamp
        self.last_position = current_position
        self.last_velocity = current_velocity
        self.samples += 1

        if system_fault:
            return self._finish(EpisodeOutcome.SYSTEM_FAULT, system_fault)
        if current_position[2] < self.config.min_altitude or current_position[2] > self.config.max_altitude:
            return self._finish(EpisodeOutcome.BOUNDARY_VIOLATION, "altitude outside evaluation bounds")
        if _distance(current_position, self.config.goal) < self.config.goal_radius:
            return self._finish(EpisodeOutcome.SUCCESS, "goal reached")
        if timestamp-self.start_time >= self.config.timeout:
            return self._finish(EpisodeOutcome.TIMEOUT, "episode timeout")
        return self.outcome

    def collision(self, dynamic: bool, detail: str) -> EpisodeOutcome:
        if not self.started:
            raise RuntimeError("episode has not started")
        outcome = EpisodeOutcome.COLLISION_DYNAMIC if dynamic else EpisodeOutcome.COLLISION_STATIC
        return self._finish(outcome, detail)

    def _finish(self, outcome: EpisodeOutcome, detail: str) -> EpisodeOutcome:
        if self.outcome == EpisodeOutcome.RUNNING:
            self.outcome = outcome
            self.detail = detail
        return self.outcome

    def result(self) -> EpisodeResult:
        if not self.started:
            raise RuntimeError("episode has not started")
        initial = _distance(self.start_position, self.config.goal)
        final = _distance(self.last_position, self.config.goal)
        spl = 0.0
        if self.outcome == EpisodeOutcome.SUCCESS:
            spl = initial/max(initial, self.path_length, 1e-9)
        return EpisodeResult(
            outcome=self.outcome,
            elapsed=max(0.0, self.last_time-self.start_time),
            initial_distance=initial,
            final_distance=final,
            path_length=self.path_length,
            spl=spl,
            min_raw_clearance=self.min_raw_clearance,
            min_static_clearance=self.min_static_clearance,
            min_dynamic_clearance=self.min_dynamic_clearance,
            max_horizontal_speed=self.max_horizontal_speed,
            max_vertical_speed=self.max_vertical_speed,
            max_acceleration=self.max_acceleration,
            samples=self.samples,
            detail=self.detail,
            start_position=self.start_position,
            final_position=self.last_position,
        )
