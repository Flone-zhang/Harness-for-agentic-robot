"""Validated, versioned offline configuration and fixed T1 plan."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .contracts import Subtask

SKILLS = {
    "mock_place_red_in_blue_box": "red_in_blue_box",
    "mock_place_green_in_bowl": "green_in_bowl",
    "mock_press_red_button": "red_button_triggered",
}


@dataclass(frozen=True)
class Config:
    mode: str = "mock"
    task_timeout_s: float = 300.0
    subtask_timeout_s: float = 60.0
    action_budget: int = 8
    max_attempts: int = 2
    max_replans: int = 1
    max_observation_waits: int = 2
    max_observation_age_s: float = 1.0
    checkpoint_id: str = "none-mock-only"

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        if path is None:
            raw = {}
        else:
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"cannot read config {path}: {exc}") from exc
            if path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as exc:
                    raise ValueError("YAML config requires PyYAML; install project dependencies") from exc
                try:
                    raw = yaml.safe_load(content)
                except yaml.YAMLError as exc:
                    raise ValueError(f"invalid YAML config {path}: {exc}") from exc
            else:
                try:
                    raw = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON config {path}: {exc}") from exc
        return cls.load_dict(raw)

    @classmethod
    def load_dict(cls, raw: dict) -> "Config":
        if not isinstance(raw, dict):
            raise ValueError("config must be a mapping of field names to values")
        if not all(isinstance(key, str) for key in raw):
            raise ValueError("config field names must be strings")
        if set(raw) - set(cls.__dataclass_fields__):
            raise ValueError(f"unknown config keys: {sorted(set(raw) - set(cls.__dataclass_fields__))}")
        config = cls(**raw)
        if config.mode != "mock":
            raise ValueError("only mock mode is implemented; physical motion is disabled")
        for name in ("task_timeout_s", "subtask_timeout_s", "max_observation_age_s"):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("action_budget", "max_attempts", "max_observation_waits"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if type(config.max_replans) is not int or config.max_replans < 0:
            raise ValueError("max_replans must be an integer >= 0")
        if not isinstance(config.checkpoint_id, str) or not config.checkpoint_id:
            raise ValueError("checkpoint_id must be a nonempty string")
        return config

    def json(self) -> dict:
        return dict(self.__dict__)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.json(), sort_keys=True).encode()).hexdigest()


def fixed_t1_plan(config: Config) -> list[Subtask]:
    goals = [
        ("red_to_blue", "mock_place_red_in_blue_box", "Put the red block in the blue box"),
        ("green_to_bowl", "mock_place_green_in_bowl", "Put the green block in the bowl"),
        ("press_button", "mock_press_red_button", "Press the red button"),
    ]
    plan = []
    for index, (subtask_id, skill, instruction) in enumerate(goals):
        predicate = SKILLS[skill]
        plan.append(Subtask(
            subtask_id=subtask_id,
            skill=skill,
            instruction=instruction,
            prerequisites=[SKILLS[goals[index - 1][1]]] if index else [],
            expected_effect=predicate,
            success_predicate=predicate,
            action_budget=config.action_budget,
            timeout_s=config.subtask_timeout_s,
        ))
    return plan


def validate_plan(plan: list[Subtask]) -> None:
    available = set()
    seen_ids = set()
    for item in plan:
        if item.subtask_id in seen_ids:
            raise ValueError(f"duplicate subtask id: {item.subtask_id}")
        if SKILLS.get(item.skill) != item.success_predicate:
            raise ValueError(f"unsupported skill/predicate: {item.skill}/{item.success_predicate}")
        if item.expected_effect != item.success_predicate:
            raise ValueError("expected effect must match verified predicate")
        if not set(item.prerequisites) <= available:
            raise ValueError(f"unsatisfied prerequisites: {item.subtask_id}")
        if item.action_budget < 1 or item.timeout_s <= 0:
            raise ValueError("invalid subtask budget")
        available.add(item.success_predicate)
        seen_ids.add(item.subtask_id)
