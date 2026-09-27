"""Evidence-aware Qwen planning over registered PI0 task names; never executes actions."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .contracts import Subtask
from .policy_skills import (DEFAULT_REGISTRY_PATH, SkillRegistry, SkillRegistryError,
                            T1_GOALS, load_action_skills, validate_t1_sequence)

DEFAULT_QWEN_CONFIG = Path(__file__).resolve().parent.parent / "config" / "piper_harness.yaml"


class PlanningError(ValueError):
    pass


@dataclass(frozen=True)
class QwenConfig:
    api_key: str
    model: str
    base_url: str
    timeout_s: float
    enable_thinking: bool
    action_budget: int
    subtask_timeout_s: float

    @classmethod
    def load(cls, path: Path = DEFAULT_QWEN_CONFIG) -> "QwenConfig":
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PlanningError(f"cannot load Qwen config at {path}: {exc}") from exc
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise PlanningError("Qwen YAML config requires PyYAML") from exc
            try:
                document = yaml.safe_load(content)
            except yaml.YAMLError as exc:
                raise PlanningError(f"invalid Qwen YAML config at {path}: {exc}") from exc
            if not isinstance(document, dict) or not isinstance(document.get("qwen"), dict):
                raise PlanningError("Qwen YAML config must contain a qwen mapping")
            # The shared runtime YAML also contains visual-loop controls.  The
            # text-only planner consumes only its own, explicitly declared
            # fields so the two modes can safely share one secret/config file.
            raw = {key: document["qwen"].get(key) for key in cls.__dataclass_fields__}
        else:
            try:
                raw = json.loads(content)
            except json.JSONDecodeError as exc:
                raise PlanningError(f"invalid Qwen JSON config at {path}: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise PlanningError("Qwen config fields do not match the example file")
        config = cls(**raw)
        if not isinstance(config.api_key, str) or not isinstance(config.model, str) or not config.model:
            raise PlanningError("Qwen API key/model must be strings")
        if not isinstance(config.base_url, str):
            raise PlanningError("Qwen base_url must be a string")
        parsed = urlsplit(config.base_url)
        if (parsed.scheme != "https" or not parsed.hostname or
                not parsed.hostname.endswith(".aliyuncs.com") or
                parsed.path.rstrip("/") != "/compatible-mode/v1" or
                parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise PlanningError("Qwen base_url must be an HTTPS Alibaba Cloud compatible-mode/v1 endpoint")
        if type(config.timeout_s) not in (int, float) or not math.isfinite(config.timeout_s) or config.timeout_s <= 0:
            raise PlanningError("timeout_s must be positive and finite")
        if type(config.enable_thinking) is not bool:
            raise PlanningError("enable_thinking must be true or false")
        if type(config.action_budget) is not int or config.action_budget < 1:
            raise PlanningError("action_budget must be a positive integer")
        if (not isinstance(config.subtask_timeout_s, (int, float)) or
                not math.isfinite(config.subtask_timeout_s) or config.subtask_timeout_s <= 0):
            raise PlanningError("subtask_timeout_s must be positive and finite")
        return config


@dataclass(frozen=True)
class PlanResult:
    task_id: str
    model: str
    verified_predicates: tuple[str, ...]
    subtasks: tuple[Subtask, ...]
    response_id: str | None
    usage: dict[str, Any]
    latency_s: float
    motion_authorized: bool = False

    def json(self) -> dict[str, Any]:
        return asdict(self)


def _read_streamed_completion(response: Any) -> dict[str, Any]:
    """Join SSE answer deltas; reasoning deltas are deliberately discarded."""
    max_wire_bytes = 4 * 1024 * 1024
    max_answer_chars = 64 * 1024
    wire_bytes = 0
    answer_parts: list[str] = []
    answer_chars = 0
    response_id = None
    usage: dict[str, Any] = {}
    finished = False
    done = False
    for raw_line in response:
        if not isinstance(raw_line, bytes):
            raise PlanningError("Qwen returned an invalid stream line")
        wire_bytes += len(raw_line)
        if wire_bytes > max_wire_bytes:
            raise PlanningError("Qwen stream exceeds size limit")
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PlanningError("Qwen stream is not UTF-8") from exc
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        if not data:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PlanningError("Qwen stream contains invalid JSON") from exc
        if not isinstance(event, dict) or "error" in event:
            raise PlanningError("Qwen stream reported an invalid response")
        if isinstance(event.get("id"), str):
            response_id = event["id"]
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            raise PlanningError("Qwen stream choices must be an array")
        if not choices:
            continue  # usage-only event
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
            raise PlanningError("Qwen stream choice has no delta")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            if finish_reason != "stop":
                raise PlanningError("Qwen stream ended without a complete stop; no plan accepted")
            finished = True
        content = choice["delta"].get("content")
        if content is not None:
            if not isinstance(content, str):
                raise PlanningError("Qwen answer delta must be text")
            answer_chars += len(content)
            if answer_chars > max_answer_chars:
                raise PlanningError("Qwen answer exceeds size limit")
            answer_parts.append(content)
    if not done or not finished:
        raise PlanningError("Qwen stream ended before a complete answer")
    answer = "".join(answer_parts)
    if not answer:
        raise PlanningError("Qwen stream contained no final answer")
    return {"id": response_id, "choices": [{"message": {"content": answer}}], "usage": usage}


def _http_transport(config: QwenConfig, payload: dict[str, Any]) -> dict[str, Any]:
    if not config.api_key.strip():
        raise PlanningError(f"Qwen API key is empty; fill qwen.api_key in {DEFAULT_QWEN_CONFIG}")
    endpoint = config.base_url.rstrip("/") + "/chat/completions"
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json",
                 "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_s) as response:
            return _read_streamed_completion(response)
    except HTTPError as exc:
        raise PlanningError(f"Qwen request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise PlanningError("Qwen request failed; check network, region and API key") from exc


def _extract_skill_ids(response: dict[str, Any]) -> list[str]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PlanningError("Qwen response has no assistant message") from exc
    if not isinstance(content, str):
        raise PlanningError("Qwen assistant content must be a JSON string")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlanningError("Qwen plan is not strict JSON; no skill will run") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"skill_ids"}:
        raise PlanningError("Qwen plan must contain only skill_ids")
    return parsed["skill_ids"]


class QwenPlanner:
    def __init__(self, config: QwenConfig, registry: SkillRegistry,
                 transport: Callable[[QwenConfig, dict[str, Any]], dict[str, Any]] | None = None):
        self.config = config
        self.registry = registry
        self.transport = transport or _http_transport

    def plan(self, task_id: str, verified_predicates: set[str], scene_summary: str = "") -> PlanResult:
        if task_id not in {"T1", "T2", "T3"}:
            raise PlanningError("task_id must be T1, T2 or T3")
        if not isinstance(scene_summary, str) or len(scene_summary) > 4000:
            raise PlanningError("scene_summary must be text up to 4000 characters")
        if not isinstance(verified_predicates, set) or not all(isinstance(x, str) for x in verified_predicates):
            raise PlanningError("verified_predicates must be a set of strings")
        # Validate caller-supplied evidence names before sending anything to the API.
        known = {name for skill in self.registry.skills for name in
                 (*skill.preconditions, *skill.adds, *skill.removes)}
        if not verified_predicates <= known:
            raise PlanningError(f"unknown verified predicates: {sorted(verified_predicates - known)}")
        if not self.config.api_key.strip():
            raise PlanningError(f"Qwen API key is empty; fill qwen.api_key in {DEFAULT_QWEN_CONFIG}")
        allowed = [skill for skill in self.registry.skills if skill.t1_allowed]
        system = (
            "You are a constrained robot task planner. Return ONLY a JSON object of the form "
            '{"skill_ids":["registered_id",...]}. Use only IDs in the provided registry. '
            "Plan the unfinished suffix of T1 in order. Never infer physical success from an action; "
            "verified_predicates are the only completed facts. No explanations or Markdown."
        )
        user = {
            "task_id": task_id,
            "goal_predicates": sorted(T1_GOALS),
            "verified_predicates": sorted(verified_predicates),
            "scene_summary": scene_summary,
            "available_skills": [
                {"id": skill.id, "policy_task": skill.policy_task,
                 "preconditions": list(skill.preconditions),
                 "expected_effects": list(skill.adds)} for skill in allowed
            ],
        }
        payload = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            "enable_thinking": self.config.enable_thinking,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # The model page shows thinking + streaming without response_format.
        # JSON mode is only requested when thinking is off; every result is
        # strictly parsed and checked against the skill registry either way.
        if not self.config.enable_thinking:
            payload["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        response = self.transport(self.config, payload)
        latency = time.perf_counter() - started
        if not isinstance(response, dict):
            raise PlanningError("Qwen transport returned an invalid response")
        ids = _extract_skill_ids(response)
        try:
            subtasks = validate_t1_sequence(
                ids, verified_predicates, self.registry,
                self.config.action_budget, self.config.subtask_timeout_s,
            )
        except SkillRegistryError as exc:
            raise PlanningError(f"Qwen plan rejected: {exc}") from exc
        usage = response.get("usage", {})
        return PlanResult(
            task_id=task_id, model=self.config.model,
            verified_predicates=tuple(sorted(verified_predicates)),
            subtasks=tuple(subtasks),
            response_id=response.get("id") if isinstance(response.get("id"), str) else None,
            usage=usage if isinstance(usage, dict) else {},
            latency_s=latency,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen planner over registered PI0 action skills; no execution")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("skills", help="list the exact fine-tuning task names")
    plan = commands.add_parser("plan", help="call Qwen after filling qwen.api_key in config/piper_harness.yaml")
    plan.add_argument("--task-id", choices=("T1", "T2", "T3"), default="T1")
    plan.add_argument("--verified", action="append", default=[], help="independently verified true predicate; repeatable")
    plan.add_argument("--scene-summary", default="", help="optional text scene summary sent to Qwen")
    plan.add_argument("--config", type=Path, default=DEFAULT_QWEN_CONFIG)
    validate = commands.add_parser("validate", help="validate saved Qwen JSON content without API key")
    validate.add_argument("--file", type=Path, required=True,
                          help='file containing {"skill_ids":[...]}')
    validate.add_argument("--verified", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_action_skills()
        if args.command == "skills":
            output = registry.json()
        elif args.command == "validate":
            try:
                saved = json.loads(args.file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PlanningError(f"cannot read saved plan: {exc}") from exc
            if not isinstance(saved, dict) or set(saved) != {"skill_ids"}:
                raise PlanningError("saved plan must contain only skill_ids")
            output = {"subtasks": [item.json() for item in validate_t1_sequence(
                saved["skill_ids"], set(args.verified), registry, 20, 60.0,
            )], "motion_authorized": False}
        else:
            config = QwenConfig.load(args.config)
            output = QwenPlanner(config, registry).plan(
                args.task_id, set(args.verified), args.scene_summary,
            ).json()
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (PlanningError, SkillRegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
