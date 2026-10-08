"""Gazebo collision-name classification used by the episode runner."""

from enum import Enum
from typing import Optional


class ContactKind(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


def classify_vehicle_contact(collision1: str, collision2: str,
                             vehicle_token: str = "quadcopter",
                             dynamic_tokens=()) -> Optional[ContactKind]:
    """Return obstacle kind when exactly one side belongs to the vehicle."""
    first_vehicle = vehicle_token.lower() in collision1.lower()
    second_vehicle = vehicle_token.lower() in collision2.lower()
    if first_vehicle == second_vehicle:
        return None
    obstacle = collision2 if first_vehicle else collision1
    lowered = obstacle.lower()
    return ContactKind.DYNAMIC if any(
        str(token).lower() in lowered for token in dynamic_tokens
    ) else ContactKind.STATIC
