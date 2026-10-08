"""Metrics for commanded-velocity response experiments."""

import math
from statistics import mean
from typing import Dict, Iterable, List, Optional


def _first(rows: Iterable[dict], predicate) -> Optional[dict]:
    return next((row for row in rows if predicate(row)), None)


def _phase(samples: List[dict], name: str) -> List[dict]:
    return [row for row in samples if row["phase"] == name]


def _braking_metrics(rows: List[dict], stop_speed: float) -> Dict[str, Optional[float]]:
    if not rows:
        return {"stopping_time": None, "stopping_distance": None}
    start = rows[0]
    stopped = _first(
        rows,
        lambda row: row["phase_time"] >= 0.05
        and math.hypot(row["actual_vx"], row["actual_vy"]) <= stop_speed,
    )
    if stopped is None:
        return {"stopping_time": None, "stopping_distance": None}
    return {
        "stopping_time": stopped["phase_time"],
        "stopping_distance": math.hypot(stopped["x"]-start["x"], stopped["y"]-start["y"]),
    }


def compute_response_metrics(samples: List[dict], target_speed: float = 0.5,
                             stop_speed: float = 0.05) -> Dict[str, object]:
    """Compute rise, braking and 90-degree direction-change response metrics."""
    if target_speed <= 0.0 or stop_speed <= 0.0 or stop_speed >= target_speed:
        raise ValueError("response thresholds are invalid")
    if not samples:
        raise ValueError("response samples must not be empty")

    accelerate = _phase(samples, "accelerate")
    low = _first(accelerate, lambda row: row["actual_vx"] >= 0.1*target_speed)
    high = _first(accelerate, lambda row: row["actual_vx"] >= 0.9*target_speed)
    rise_time = None
    if low is not None and high is not None:
        rise_time = high["phase_time"]-low["phase_time"]
    steady = [row["actual_vx"] for row in accelerate if row["phase_time"] >= 2.5]

    turn = _phase(samples, "turn")
    settled_turn = _first(
        turn,
        lambda row: row["actual_vy"] >= 0.9*target_speed
        and abs(row["actual_vx"]) <= 0.1*target_speed,
    )

    accelerations = []
    for previous, current in zip(samples, samples[1:]):
        dt = current["time"]-previous["time"]
        if dt > 1e-6:
            accelerations.append(math.hypot(
                current["actual_vx"]-previous["actual_vx"],
                current["actual_vy"]-previous["actual_vy"],
            )/dt)

    return {
        "target_speed": target_speed,
        "stop_speed_threshold": stop_speed,
        "rise_time_10_to_90": rise_time,
        "steady_forward_speed": mean(steady) if steady else None,
        "first_brake": _braking_metrics(_phase(samples, "brake"), stop_speed),
        "turn_settling_time": None if settled_turn is None else settled_turn["phase_time"],
        "final_brake": _braking_metrics(_phase(samples, "final_brake"), stop_speed),
        "peak_horizontal_acceleration": max(accelerations) if accelerations else None,
        "sample_count": len(samples),
    }
