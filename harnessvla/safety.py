"""Independent command validation; physical command delivery remains unavailable."""

import math
import time
from dataclasses import dataclass

from .contracts import Observation


class SafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class JointCommand:
    positions_rad: tuple[float, ...]
    max_speed_rad_s: float
    gripper_position_m: float


@dataclass(frozen=True)
class SafetyLimits:
    joint_min_rad: tuple[float, ...]
    joint_max_rad: tuple[float, ...]
    max_speed_rad_s: float
    gripper_min_m: float
    gripper_max_m: float
    max_observation_age_s: float


def validate_joint_command(
    command: JointCommand, observation: Observation, limits: SafetyLimits, now_ns: int | None = None
) -> None:
    """Reject a command before any adapter sees it. Does not authorize motion."""
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if observation.image_quality != "ok" or observation.robot_status != "idle":
        raise SafetyError("observation or robot status invalid")
    if observation.monotonic_ns > now_ns or (now_ns - observation.monotonic_ns) / 1e9 > limits.max_observation_age_s:
        raise SafetyError("observation stale or from the future")
    count = len(limits.joint_min_rad)
    if count == 0 or not (len(limits.joint_max_rad) == len(command.positions_rad) == len(observation.joint_positions_rad) == count):
        raise SafetyError("joint dimension mismatch")
    values = (*command.positions_rad, command.max_speed_rad_s, command.gripper_position_m)
    if not all(math.isfinite(value) for value in values):
        raise SafetyError("non-finite command")
    if command.max_speed_rad_s <= 0 or command.max_speed_rad_s > limits.max_speed_rad_s:
        raise SafetyError("speed limit exceeded")
    if not limits.gripper_min_m <= command.gripper_position_m <= limits.gripper_max_m:
        raise SafetyError("gripper limit exceeded")
    for index, position in enumerate(command.positions_rad):
        if not limits.joint_min_rad[index] <= position <= limits.joint_max_rad[index]:
            raise SafetyError(f"joint {index} limit exceeded")


class PiperAdapter:
    """Deliberately disabled until driver, firmware, calibration and site safety are verified."""

    def read_state(self):
        raise SafetyError("Piper driver and hardware configuration have not been verified")

    def send(self, command: JointCommand) -> None:
        raise SafetyError("physical motion is disabled; no Piper command path is implemented")
