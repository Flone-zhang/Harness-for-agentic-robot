"""Clearly synthetic sensor and skill fixtures. Never connects to hardware."""

import time
import uuid
from datetime import datetime, timezone

from .config import SKILLS
from .contracts import ExecutionResult, Observation, PredicateResult, Subtask


def observe(world: dict, quality: str = "ok") -> Observation:
    observation_id = str(uuid.uuid4())
    return Observation(
        observation_id=observation_id,
        monotonic_ns=time.monotonic_ns(),
        wall_time=datetime.now(timezone.utc).isoformat(),
        agent_image_ref=f"mock://agent/{observation_id}",
        wrist_image_ref=f"mock://wrist/{observation_id}",
        joint_positions_rad=[0.0] * 6,
        end_effector_pose_m_rad=[0.0] * 6,
        gripper_position_m=0.0,
        frame="mock_base",
        image_quality=quality,
        robot_status="idle",
        signals={
            "red_in_blue_box": world.get("red_in_blue_box"),
            "green_in_bowl": world.get("green_in_bowl"),
            "red_button_triggered": world.get("red_button_triggered"),
            "box_pose": world.get("box_pose"),
        },
    )


def predicate(name: str, observation: Observation) -> PredicateResult:
    if observation.image_quality != "ok":
        return PredicateResult("unknown", None, "invalid_observation")
    value = observation.signals.get(name)
    if value is None:
        return PredicateResult("unknown", None, "missing_mock_signal")
    if not isinstance(value, bool):
        return PredicateResult("unknown", None, "invalid_mock_signal")
    return PredicateResult(
        "true" if value else "false",
        f"{observation.observation_id}#signals/{name}",
        "mock_world_sensor",
        1.0,
    )


def execute(subtask: Subtask, world: dict, observation: Observation, fault: str | None) -> ExecutionResult:
    if subtask.skill not in SKILLS:
        return ExecutionResult("unsupported", 0, observation.observation_id, 0.0, "unknown_skill")
    if fault == "empty_grasp":
        return ExecutionResult("failure", 1, observation.observation_id, 0.25, fault)
    if fault == "drop":
        world[subtask.success_predicate] = False
        world["current_object_pose"] = None
        return ExecutionResult("failure", 1, observation.observation_id, 0.25, fault)
    if fault == "false_completion":
        return ExecutionResult("success", 1, observation.observation_id, 0.99, fault)
    if fault == "missing_signal":
        world[subtask.success_predicate] = None
        return ExecutionResult("success", 1, observation.observation_id, 0.99, fault)
    world[subtask.success_predicate] = True
    return ExecutionResult("success", 1, observation.observation_id, 0.99)
