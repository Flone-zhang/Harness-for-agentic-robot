"""PI0 task names with a user-requested correction and a constrained T1 whitelist."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import Subtask

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "pi0_action_skills.json"
T1_GOALS = frozenset({"red_in_blue_box", "green_in_bowl", "red_button_triggered"})


class SkillRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ActionSkill:
    id: str
    source_key: str
    policy_task: str
    preconditions: tuple[str, ...]
    adds: tuple[str, ...]
    removes: tuple[str, ...]
    success_predicate: str
    t1_allowed: bool

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillRegistry:
    version: int
    source: str
    execution_enabled: bool
    skills: tuple[ActionSkill, ...]

    def by_id(self) -> dict[str, ActionSkill]:
        return {skill.id: skill for skill in self.skills}

    def json(self) -> dict[str, Any]:
        return {"version": self.version, "source": self.source,
                "execution_enabled": self.execution_enabled,
                "skills": [skill.json() for skill in self.skills]}


def load_action_skills(path: Path = DEFAULT_REGISTRY_PATH) -> SkillRegistry:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SkillRegistryError(f"cannot load action skill registry: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("skills"), list):
        raise SkillRegistryError("unsupported action skill registry format")
    if type(raw.get("execution_enabled")) is not bool:
        raise SkillRegistryError("execution_enabled must be boolean")
    if not isinstance(raw.get("source"), str) or not raw["source"]:
        raise SkillRegistryError("registry source is required")
    skills = []
    seen_ids, seen_keys = set(), set()
    for entry in raw["skills"]:
        if not isinstance(entry, dict):
            raise SkillRegistryError("skill must be a JSON object")
        required = {"id", "source_key", "policy_task", "preconditions", "adds",
                    "removes", "success_predicate", "t1_allowed"}
        if set(entry) != required:
            raise SkillRegistryError(f"invalid skill fields for {entry.get('id')}")
        if not all(isinstance(entry[key], str) and entry[key] for key in
                   ("id", "source_key", "policy_task", "success_predicate")):
            raise SkillRegistryError("skill identifiers and policy task must be nonempty strings")
        if not all(isinstance(entry[key], list) and all(isinstance(x, str) and x for x in entry[key])
                   for key in ("preconditions", "adds", "removes")):
            raise SkillRegistryError("skill predicates must be string lists")
        if type(entry["t1_allowed"]) is not bool:
            raise SkillRegistryError("t1_allowed must be boolean")
        if entry["id"] in seen_ids or entry["source_key"] in seen_keys:
            raise SkillRegistryError("duplicate skill ID or source key")
        if entry["success_predicate"] not in entry["adds"]:
            raise SkillRegistryError("success predicate must be an expected added effect")
        if set(entry["adds"]) & set(entry["removes"]):
            raise SkillRegistryError("skill cannot add and remove the same predicate")
        skills.append(ActionSkill(
            id=entry["id"], source_key=entry["source_key"],
            policy_task=entry["policy_task"],
            preconditions=tuple(entry["preconditions"]),
            adds=tuple(entry["adds"]), removes=tuple(entry["removes"]),
            success_predicate=entry["success_predicate"],
            t1_allowed=entry["t1_allowed"],
        ))
        seen_ids.add(entry["id"])
        seen_keys.add(entry["source_key"])
    if not skills:
        raise SkillRegistryError("no action skills registered")
    return SkillRegistry(1, raw["source"], raw["execution_enabled"], tuple(skills))


def validate_t1_sequence(
    skill_ids: list[str], verified_predicates: set[str], registry: SkillRegistry,
    action_budget: int, timeout_s: float,
) -> list[Subtask]:
    """Validate a *proposed* suffix. Simulated effects are not completion evidence."""
    if not isinstance(skill_ids, list) or not all(isinstance(x, str) for x in skill_ids):
        raise SkillRegistryError("skill_ids must be an array of strings")
    if len(skill_ids) > len(registry.skills):
        raise SkillRegistryError("plan exceeds registered skill count")
    if type(action_budget) is not int or action_budget < 1 or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise SkillRegistryError("invalid planner budgets")
    known_predicates = {name for skill in registry.skills for name in
                        (*skill.preconditions, *skill.adds, *skill.removes)}
    if not verified_predicates <= known_predicates:
        raise SkillRegistryError(f"unknown verified predicates: {sorted(verified_predicates - known_predicates)}")
    by_id = registry.by_id()
    simulated = set(verified_predicates)
    seen = set()
    plan = []
    for index, skill_id in enumerate(skill_ids, start=1):
        skill = by_id.get(skill_id)
        if skill is None:
            raise SkillRegistryError(f"unknown skill: {skill_id}")
        if not skill.t1_allowed:
            raise SkillRegistryError(f"skill not allowed in T1 plan: {skill_id}")
        if skill_id in seen or skill.success_predicate in simulated:
            raise SkillRegistryError(f"duplicate or already-complete skill: {skill_id}")
        missing = set(skill.preconditions) - simulated
        if missing:
            raise SkillRegistryError(f"preconditions not met for {skill_id}: {sorted(missing)}")
        plan.append(Subtask(
            subtask_id=f"pi0_{index}_{skill.id}", skill=skill.id,
            instruction=skill.policy_task, prerequisites=list(skill.preconditions),
            expected_effect=skill.success_predicate,
            success_predicate=skill.success_predicate,
            action_budget=action_budget, timeout_s=float(timeout_s),
        ))
        simulated.difference_update(skill.removes)
        simulated.update(skill.adds)
        seen.add(skill_id)
    missing_goals = T1_GOALS - simulated
    if missing_goals:
        raise SkillRegistryError(f"plan does not cover T1 goals: {sorted(missing_goals)}")
    return plan
