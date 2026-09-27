"""Small, explicit data contracts shared by the orchestration boundaries."""

from dataclasses import asdict, dataclass
from typing import Any, Literal

PredicateValue = Literal["true", "false", "unknown"]
DecisionAction = Literal["wait", "end", "replay", "recover", "replan", "fail"]


@dataclass(frozen=True)
class Observation:
    observation_id: str
    monotonic_ns: int
    wall_time: str
    agent_image_ref: str
    wrist_image_ref: str
    joint_positions_rad: list[float]
    end_effector_pose_m_rad: list[float]
    gripper_position_m: float
    frame: str
    image_quality: str
    robot_status: str
    signals: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Subtask:
    subtask_id: str
    skill: str
    instruction: str
    prerequisites: list[str]
    expected_effect: str
    success_predicate: str
    action_budget: int
    timeout_s: float

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    status: Literal["success", "failure", "budget_exhausted", "unsupported"]
    action_steps: int
    observation_ref: str
    progress: float
    fault: str | None = None

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredicateResult:
    value: PredicateValue
    evidence_ref: str | None
    source: str
    confidence: float | None = None

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Fact:
    key: str
    value: Any
    fact_type: Literal["scene", "constraint", "diagnostic"]
    source: str
    observed_monotonic_ns: int
    expires_after_s: float | None
    valid: bool

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionInput:
    safety_fault: str | None
    image_quality: str
    predicate: PredicateValue
    execution_status: str
    fault: str | None
    attempt: int
    max_attempts: int
    replans: int
    max_replans: int
    observation_waits: int
    max_observation_waits: int
    progress: float

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    action: DecisionAction
    reason: str
    retry_count: int

    def json(self) -> dict[str, Any]:
        return asdict(self)
