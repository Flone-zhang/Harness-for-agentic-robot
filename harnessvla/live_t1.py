"""Scene-aware Qwen planning and bounded PI0 execution for a Piper task.

The language model can select only registered skills.  It never emits joint
commands.  PI0 is the sole action source, while this module owns pausing,
budgets, visual evidence, retries, audit events, and safe shutdown.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .pi0 import inspect_checkpoint
from .policy_skills import T1_GOALS, ActionSkill, SkillRegistry, load_action_skills
from .qwen_planner import PlanningError, QwenConfig, _http_transport
from .store import Store, code_hash


class LiveT1Error(RuntimeError):
    """A controlled run stop that may use safe-pose-then-disable shutdown."""


class LiveSafetyFault(LiveT1Error):
    """A perception/control fault requiring immediate disable without homing."""


@dataclass(frozen=True)
class EvidencePair:
    top_path: str
    wrist_path: str
    top_sha256: str
    wrist_sha256: str
    top_data_url: str
    wrist_data_url: str

    def audit_json(self) -> dict[str, str]:
        result = asdict(self)
        result.pop("top_data_url")
        result.pop("wrist_data_url")
        return result


@dataclass(frozen=True)
class PlannerDecision:
    decision: Literal["execute", "finish", "fail"]
    skill_id: str | None
    reason: str
    response_id: str | None
    usage: dict[str, Any]
    latency_s: float

    def audit_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationDecision:
    predicate: str
    value: Literal["true", "false", "unknown"]
    confidence: float
    reason: str
    response_id: str | None
    usage: dict[str, Any]
    latency_s: float

    def audit_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionReport:
    actions_sent: int
    action_records: tuple[dict[str, Any], ...]
    elapsed_s: float
    timed_out: bool = False


@dataclass(frozen=True)
class LiveSettings:
    project_root: Path
    source_path: Path
    task_id: str
    task_command: str
    db_path: Path
    checkpoint: Path
    checkpoint_hash: str
    device: str
    server_address: str
    can_port: str
    fps: int
    camera_width: int
    camera_height: int
    camera_fps: int
    top_camera_serial: str
    wrist_camera_serial: str
    max_joint_step: float
    max_gripper_step: float
    console_status: bool
    verification_interval_actions: int
    wait_pose_min_actions: int
    wait_pose_stable_intervals: int
    wait_pose_joint_range: float
    wait_pose_gripper_range: float
    verification_retries: int
    jpeg_quality: int
    execution_timeout_s: float
    reset_hz: float
    reset_duration_s: float
    reset_max_joint_step: float
    home_gripper: float


class LiveExecutor(Protocol):
    def start(self) -> None: ...
    def capture_frames(self) -> dict[str, Any]: ...
    def execute(self, skill: ActionSkill, action_budget: int, timeout_s: float) -> ExecutionReport: ...
    def normal_stop(self) -> None: ...
    def emergency_stop(self) -> None: ...


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _project_path(project_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _positive_int(value: Any, name: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        suffix = f" between 1 and {maximum}" if maximum is not None else " a positive integer"
        raise ValueError(f"{name} must be{suffix}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def load_live_settings(config: dict[str, Any], source_path: Path,
                       require_motion: bool = True) -> LiveSettings:
    project_root = Path(__file__).resolve().parent.parent
    run = config.get("run")
    pi0 = config.get("pi0")
    qwen = config.get("qwen")
    if not all(isinstance(section, dict) for section in (run, pi0, qwen)):
        raise ValueError("run, pi0 and qwen must be YAML mappings")
    task_command = run.get("task_command")
    if (not isinstance(task_command, str) or not task_command.strip()
            or len(task_command) > 2000):
        raise ValueError("run.task_command must be nonempty text up to 2000 characters")
    console_status = run.get("console_status", True)
    if type(console_status) is not bool:
        raise ValueError("run.console_status must be true or false")
    if require_motion and pi0.get("motion_enabled") is not True:
        raise ValueError("qwen_pi0 requires pi0.motion_enabled: true")
    expected_hash = pi0.get("expected_model_sha256")
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64 or
            any(character not in "0123456789abcdefABCDEF" for character in expected_hash)):
        raise ValueError("pi0.expected_model_sha256 must be a 64-character hex digest")
    address = pi0.get("server_address")
    if not isinstance(address, str):
        raise ValueError("pi0.server_address must be text")
    host, separator, port = address.rpartition(":")
    if not separator or host != "127.0.0.1" or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("pi0.server_address must be 127.0.0.1:<port>")
    cameras = pi0.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != {"cam_top", "cam_left"}:
        raise ValueError("pi0.cameras must contain exactly cam_top and cam_left")
    top, wrist = cameras["cam_top"], cameras["cam_left"]
    if not isinstance(top, dict) or not isinstance(wrist, dict):
        raise ValueError("camera settings must be mappings")
    dimensions = []
    for name, camera in (("cam_top", top), ("cam_left", wrist)):
        serial = camera.get("serial_number_or_name")
        if not isinstance(serial, str) or not serial:
            raise ValueError(f"pi0.cameras.{name}.serial_number_or_name is required")
        dimensions.append(tuple(_positive_int(camera.get(key), f"pi0.cameras.{name}.{key}")
                                for key in ("width", "height", "fps")))
    if dimensions[0] != dimensions[1]:
        raise ValueError("both cameras must use the same width, height and fps")
    jpeg_quality = _positive_int(qwen.get("jpeg_quality"), "qwen.jpeg_quality", 95)
    if jpeg_quality < 40:
        raise ValueError("qwen.jpeg_quality must be between 40 and 95")
    verification_retries = qwen.get("visual_verification_retries")
    if type(verification_retries) is not int or not 1 <= verification_retries <= 3:
        raise ValueError("qwen.visual_verification_retries must be between 1 and 3")
    fps = _positive_int(pi0.get("fps"), "pi0.fps", 120)
    if dimensions[0][2] < fps:
        raise ValueError("camera fps must be at least pi0.fps")
    for name in ("device", "can_port"):
        if not isinstance(pi0.get(name), str) or not pi0[name].strip():
            raise ValueError(f"pi0.{name} must be nonempty text")
    home_gripper = pi0.get("home_gripper", 0.07)
    if (type(home_gripper) not in (int, float) or not math.isfinite(home_gripper) or
            not 0.0 <= home_gripper <= 0.08):
        raise ValueError("pi0.home_gripper must be between 0.0 and 0.08 metres")
    return LiveSettings(
        project_root=project_root, source_path=source_path.resolve(),
        task_id="main_task", task_command=task_command.strip(),
        db_path=_project_path(project_root, run.get("db"), "run.db"),
        checkpoint=_project_path(project_root, pi0.get("checkpoint"), "pi0.checkpoint"),
        checkpoint_hash=expected_hash.lower(), device=pi0["device"], server_address=address,
        can_port=pi0["can_port"], fps=fps, camera_width=dimensions[0][0],
        camera_height=dimensions[0][1], camera_fps=dimensions[0][2],
        top_camera_serial=top["serial_number_or_name"],
        wrist_camera_serial=wrist["serial_number_or_name"],
        max_joint_step=_positive_float(pi0.get("max_joint_step"), "pi0.max_joint_step"),
        max_gripper_step=_positive_float(pi0.get("max_gripper_step"), "pi0.max_gripper_step"),
        console_status=console_status,
        verification_interval_actions=_positive_int(
            pi0.get("verification_interval_actions"),
            "pi0.verification_interval_actions", 200,
        ),
        wait_pose_min_actions=_positive_int(
            pi0.get("wait_pose_min_actions", 200), "pi0.wait_pose_min_actions", 2000,
        ),
        wait_pose_stable_intervals=_positive_int(
            pi0.get("wait_pose_stable_intervals", 2),
            "pi0.wait_pose_stable_intervals", 5,
        ),
        wait_pose_joint_range=_positive_float(
            pi0.get("wait_pose_joint_range", 0.015), "pi0.wait_pose_joint_range",
        ),
        wait_pose_gripper_range=_positive_float(
            pi0.get("wait_pose_gripper_range", 0.001),
            "pi0.wait_pose_gripper_range",
        ),
        verification_retries=verification_retries, jpeg_quality=jpeg_quality,
        execution_timeout_s=_positive_float(qwen.get("subtask_timeout_s"),
                                             "qwen.subtask_timeout_s"),
        reset_hz=_positive_float(pi0.get("reset_hz", 100.0), "pi0.reset_hz"),
        reset_duration_s=_positive_float(pi0.get("reset_duration_s", 4.0),
                                         "pi0.reset_duration_s"),
        reset_max_joint_step=_positive_float(pi0.get("reset_max_joint_step", 0.01),
                                              "pi0.reset_max_joint_step"),
        home_gripper=float(home_gripper),
    )


class EvidenceRecorder:
    def __init__(self, directory: Path, jpeg_quality: int):
        self.directory = directory
        self.jpeg_quality = jpeg_quality
        self.sequence = 0

    def _jpeg(self, frame: Any) -> bytes:
        if isinstance(frame, bytes):
            if not frame:
                raise LiveSafetyFault("camera returned an empty image")
            return frame
        try:
            from PIL import Image
            image = Image.fromarray(frame)
            destination = io.BytesIO()
            image.save(destination, format="JPEG", quality=self.jpeg_quality)
            data = destination.getvalue()
        except Exception as exc:
            raise LiveSafetyFault(f"cannot encode camera image: {exc}") from exc
        if not data:
            raise LiveSafetyFault("camera JPEG encoding returned no data")
        return data

    def record(self, stage: str, frames: dict[str, Any]) -> EvidencePair:
        if not isinstance(frames, dict) or set(frames) != {"cam_top", "cam_left"}:
            raise LiveSafetyFault("camera capture must contain cam_top and cam_left")
        safe_stage = "".join(character if character.isalnum() or character in "-_" else "_"
                             for character in stage)[:80]
        self.sequence += 1
        top = self._jpeg(frames["cam_top"])
        wrist = self._jpeg(frames["cam_left"])
        self.directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.sequence:04d}_{safe_stage}"
        top_path = self.directory / f"{prefix}_top.jpg"
        wrist_path = self.directory / f"{prefix}_wrist.jpg"
        top_path.write_bytes(top)
        wrist_path.write_bytes(wrist)
        return EvidencePair(
            top_path=str(top_path), wrist_path=str(wrist_path),
            top_sha256=_sha256(top), wrist_sha256=_sha256(wrist),
            top_data_url="data:image/jpeg;base64," + base64.b64encode(top).decode("ascii"),
            wrist_data_url="data:image/jpeg;base64," + base64.b64encode(wrist).decode("ascii"),
        )


def _nonstream_transport(config: QwenConfig, payload: dict[str, Any]) -> dict[str, Any]:
    if not config.api_key.strip():
        raise PlanningError("Qwen API key is empty")
    request = Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.api_key}",
                 "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_s) as response:
            body = response.read(4 * 1024 * 1024 + 1)
    except HTTPError as exc:
        raise PlanningError(f"Qwen request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise PlanningError("Qwen request failed; check network, region and API key") from exc
    if len(body) > 4 * 1024 * 1024:
        raise PlanningError("Qwen response exceeds size limit")
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanningError("Qwen returned invalid JSON transport data") from exc
    if not isinstance(result, dict) or "error" in result:
        raise PlanningError("Qwen returned an error response")
    return result


def _assistant_json(response: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PlanningError("Qwen response has no assistant content") from exc
    if not isinstance(content, str) or len(content) > 64 * 1024:
        raise PlanningError("Qwen assistant content is invalid")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlanningError("Qwen visual result is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise PlanningError("Qwen visual result must be a JSON object")
    return parsed


def _image_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _is_transient_qwen_error(error: PlanningError) -> bool:
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, HTTPError):
            return cause.code in {408, 409, 425, 429} or cause.code >= 500
        if isinstance(cause, (URLError, TimeoutError)):
            return True
        cause = cause.__cause__
    return False


def _interval_is_stationary(
        records: tuple[dict[str, Any], ...], joint_range: float,
        gripper_range: float) -> bool:
    if not records:
        return False
    states: list[list[float]] = []
    for record in records:
        state = record.get("current_state")
        if (not isinstance(state, list) or len(state) != 7 or
                not all(isinstance(value, (int, float)) and math.isfinite(value)
                        for value in state)):
            return False
        states.append([float(value) for value in state])
    ranges = [max(state[index] for state in states) -
              min(state[index] for state in states) for index in range(7)]
    return max(ranges[:6]) <= joint_range and ranges[6] <= gripper_range


def _completed_goal_skill_ids(
        registry: SkillRegistry, verified: set[str]) -> set[str]:
    """Derive completed skill branches from registry effects, stopping at other goals."""
    producers: dict[str, list[ActionSkill]] = {}
    for skill in registry.skills:
        for predicate in skill.adds:
            producers.setdefault(predicate, []).append(skill)
    completed: set[str] = set()
    for goal in verified & T1_GOALS:
        pending = [goal]
        visited_predicates: set[str] = set()
        while pending:
            predicate = pending.pop()
            if predicate in visited_predicates:
                continue
            visited_predicates.add(predicate)
            for skill in producers.get(predicate, []):
                completed.add(skill.id)
                pending.extend(
                    prerequisite for prerequisite in skill.preconditions
                    if prerequisite not in T1_GOALS
                )
    return completed


class QwenVisionAgent:
    def __init__(self, config: QwenConfig, registry: SkillRegistry, task_command: str,
                 planner_transport: Callable[[QwenConfig, dict[str, Any]], dict[str, Any]] | None = None,
                 verifier_transport: Callable[[QwenConfig, dict[str, Any]], dict[str, Any]] | None = None):
        if not isinstance(task_command, str) or not task_command.strip():
            raise ValueError("task_command must be nonempty text")
        self.config = config
        self.registry = registry
        self.task_command = task_command.strip()
        self.planner_transport = planner_transport or _http_transport
        self.verifier_transport = verifier_transport or _nonstream_transport

    def choose_next(self, evidence: EvidencePair, verified: set[str],
                    skill_progress: dict[str, dict[str, int]],
                    history: list[dict[str, Any]]) -> PlannerDecision:
        completed_skill_ids = _completed_goal_skill_ids(self.registry, verified)
        allowed = [
            skill for skill in self.registry.skills
            if skill.t1_allowed and skill.id not in completed_skill_ids
        ]
        allowed_ids = {skill.id for skill in allowed}
        system = (
            "You are the visual high-level planner for a Piper robot. Inspect both current camera "
            "images and choose exactly one next registered skill for the user's total task. "
            "PI0, not you, generates "
            "motor actions. Return only strict JSON with exactly: decision, skill_id, reason. "
            "decision is execute, finish, or fail. For execute skill_id is a registry ID; otherwise "
            "it is null. Never select return_home. Use finish only when every goal is already verified. "
            "You are called once before a skill starts, then only after that skill has been visually "
            "confirmed complete. Mid-skill visual checkpoints are handled outside the planner and are "
            "never attempts or retries. Use fail only when a required target is absent or unrecoverable, "
            "or when the visible scene is unsafe."
        )
        context = {
            "total_task_command": self.task_command,
            "continuation_rule": (
                "Plan from the freshly verified scene state. Never repeat a completed goal branch."
            ),
            "goal_predicates": sorted(T1_GOALS), "verified_predicates": sorted(verified),
            "skill_progress": skill_progress, "recent_history": history[-8:],
            "available_skills": [{"id": skill.id, "policy_task": skill.policy_task,
                                  "preconditions": list(skill.preconditions),
                                  "success_predicate": skill.success_predicate}
                                 for skill in allowed],
        }
        payload = {
            "model": self.config.model, "enable_thinking": True, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": [
                {"type": "text", "text": "Top camera:"}, _image_part(evidence.top_data_url),
                {"type": "text", "text": "Wrist camera:"}, _image_part(evidence.wrist_data_url),
                {"type": "text", "text": json.dumps(context, ensure_ascii=False)},
            ]}],
        }
        started = time.perf_counter()
        response = self.planner_transport(self.config, payload)
        elapsed = time.perf_counter() - started
        parsed = _assistant_json(response)
        if set(parsed) != {"decision", "skill_id", "reason"}:
            raise PlanningError("Qwen planner output has unexpected fields")
        decision, skill_id, reason = parsed["decision"], parsed["skill_id"], parsed["reason"]
        if decision not in {"execute", "finish", "fail"} or not isinstance(reason, str) or not reason.strip():
            raise PlanningError("Qwen planner decision/reason is invalid")
        if len(reason) > 2000:
            raise PlanningError("Qwen planner reason is too long")
        if decision == "execute":
            if not isinstance(skill_id, str):
                raise PlanningError("Qwen execute decision requires a skill_id")
            skill = self.registry.by_id().get(skill_id)
            if skill is None or skill_id not in allowed_ids:
                raise PlanningError(
                    f"Qwen selected an unavailable or completed skill: {skill_id}"
                )
            missing = set(skill.preconditions) - verified
            if missing:
                raise PlanningError(f"Qwen selected {skill_id} with unmet preconditions: {sorted(missing)}")
            if skill.success_predicate in verified:
                raise PlanningError(f"Qwen selected already-complete skill: {skill_id}")
        elif skill_id is not None:
            raise PlanningError("Qwen finish/fail decision requires null skill_id")
        if decision == "finish" and not T1_GOALS <= verified:
            raise PlanningError("Qwen cannot finish before all task goals are verified")
        usage = response.get("usage")
        return PlannerDecision(decision, skill_id, reason,
                               response.get("id") if isinstance(response.get("id"), str) else None,
                               usage if isinstance(usage, dict) else {}, elapsed)

    def verify(self, skill: ActionSkill, before: EvidencePair,
               after: EvidencePair) -> VerificationDecision:
        criteria = {
            "holding_red_block": (
                "True only if the red block is visibly secured between the gripper jaws and lifted "
                "clear of the tabletop in both after views. Contact, surrounding the block, or a "
                "block still touching the table is false."
            ),
            "red_in_blue_box": (
                "True only if the red block is visibly released and resting fully inside the blue box."
            ),
            "holding_green_block": (
                "True only if the green block is visibly secured between the gripper jaws and lifted "
                "clear of the tabletop in both after views. Contact, surrounding the block, or a "
                "block still touching the table is false."
            ),
            "green_in_bowl": (
                "True only if the green block is visibly released and resting inside the bowl."
            ),
            "red_button_triggered": (
                "True only if the before/after evidence shows a clear button actuation or state "
                "change; gripper contact alone is not enough."
            ),
        }[skill.success_predicate]
        system = (
            "You are a strict visual verifier for a robot action. Compare the synchronized before "
            "and after views. Judge only the requested predicate. Return only a JSON object with "
            "exactly predicate, value, confidence, reason. value must be true, false, or unknown. "
            "Use unknown when occlusion, blur, framing, or evidence ambiguity prevents a reliable "
            "judgment. Never infer success merely from the commanded action or gripper proximity."
        )
        text = {"total_task_command": self.task_command,
                "skill_id": skill.id, "policy_task": skill.policy_task,
                "predicate": skill.success_predicate, "strict_success_criteria": criteria}
        payload = {
            "model": self.config.model, "enable_thinking": False, "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": [
                {"type": "text", "text": "Before, top:"}, _image_part(before.top_data_url),
                {"type": "text", "text": "Before, wrist:"}, _image_part(before.wrist_data_url),
                {"type": "text", "text": "After, top:"}, _image_part(after.top_data_url),
                {"type": "text", "text": "After, wrist:"}, _image_part(after.wrist_data_url),
                {"type": "text", "text": json.dumps(text, ensure_ascii=False)},
            ]}],
        }
        started = time.perf_counter()
        response = self.verifier_transport(self.config, payload)
        elapsed = time.perf_counter() - started
        parsed = _assistant_json(response)
        if set(parsed) != {"predicate", "value", "confidence", "reason"}:
            raise PlanningError("Qwen verifier output has unexpected fields")
        if parsed["predicate"] != skill.success_predicate:
            raise PlanningError("Qwen verifier returned the wrong predicate")
        value = parsed["value"]
        # JSON-mode vision models commonly use native booleans even when the
        # prompt asks for the equivalent string enum.  Normalize only these two
        # unambiguous values; every other deviation is still rejected.
        if type(value) is bool:
            value = "true" if value else "false"
        if value not in {"true", "false", "unknown"}:
            raise PlanningError("Qwen verifier returned an invalid value")
        confidence = parsed["confidence"]
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise PlanningError("Qwen verifier confidence must be between 0 and 1")
        if not isinstance(parsed["reason"], str) or not parsed["reason"].strip() or len(parsed["reason"]) > 2000:
            raise PlanningError("Qwen verifier reason is invalid")
        usage = response.get("usage")
        return VerificationDecision(parsed["predicate"], value, float(confidence),
                                    parsed["reason"],
                                    response.get("id") if isinstance(response.get("id"), str) else None,
                                    usage if isinstance(usage, dict) else {}, elapsed)

    def assess_scene_goals(
            self, evidence: EvidencePair) -> dict[str, VerificationDecision]:
        """Assess every persistent goal before planning any robot action."""
        criteria = {
            "red_in_blue_box": (
                "True only if the red block is visibly resting fully inside the blue box. "
                "If either the red block or blue box cannot be located reliably, use unknown."
            ),
            "green_in_bowl": (
                "True only if the green block is visibly released and resting inside the bowl. "
                "If either the green block or bowl cannot be located reliably, use unknown."
            ),
            "red_button_triggered": (
                "True only if the current views show unambiguous visible evidence that the red "
                "button is already triggered. Otherwise use false or unknown; do not infer it."
            ),
        }
        system = (
            "You are a strict scene-state verifier for a robot continuation task. Inspect both "
            "synchronized current views and judge every requested goal predicate. No robot action "
            "has been executed in this run yet. Return only a JSON object with exactly one key, "
            "results. results must contain exactly one object per requested predicate, and each "
            "object must have exactly predicate, value, confidence, reason. value must be true, "
            "false, or unknown. Use unknown for occlusion, blur, framing, or ambiguity."
        )
        text = {
            "total_task_command": self.task_command,
            "goal_predicates": [
                {"predicate": predicate, "strict_success_criteria": criteria[predicate]}
                for predicate in sorted(T1_GOALS)
            ],
        }
        payload = {
            "model": self.config.model, "enable_thinking": False, "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": [
                {"type": "text", "text": "Current top camera:"},
                _image_part(evidence.top_data_url),
                {"type": "text", "text": "Current wrist camera:"},
                _image_part(evidence.wrist_data_url),
                {"type": "text", "text": json.dumps(text, ensure_ascii=False)},
            ]}],
        }
        started = time.perf_counter()
        response = self.verifier_transport(self.config, payload)
        elapsed = time.perf_counter() - started
        parsed = _assistant_json(response)
        if set(parsed) != {"results"} or not isinstance(parsed["results"], list):
            raise PlanningError("Qwen scene verifier output has unexpected fields")
        usage = response.get("usage")
        response_id = response.get("id") if isinstance(response.get("id"), str) else None
        decisions: dict[str, VerificationDecision] = {}
        for item in parsed["results"]:
            if not isinstance(item, dict) or set(item) != {
                    "predicate", "value", "confidence", "reason"}:
                raise PlanningError("Qwen scene verifier result has unexpected fields")
            predicate = item["predicate"]
            if predicate not in T1_GOALS or predicate in decisions:
                raise PlanningError("Qwen scene verifier returned an invalid predicate set")
            value = item["value"]
            if type(value) is bool:
                value = "true" if value else "false"
            if value not in {"true", "false", "unknown"}:
                raise PlanningError("Qwen scene verifier returned an invalid value")
            confidence = item["confidence"]
            if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1):
                raise PlanningError(
                    "Qwen scene verifier confidence must be between 0 and 1"
                )
            reason = item["reason"]
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
                raise PlanningError("Qwen scene verifier reason is invalid")
            decisions[predicate] = VerificationDecision(
                predicate, value, float(confidence), reason, response_id,
                usage if isinstance(usage, dict) else {}, elapsed,
            )
        if set(decisions) != T1_GOALS:
            raise PlanningError("Qwen scene verifier omitted a goal predicate")
        return decisions


class LiveT1Orchestrator:
    qwen_retry_delays_s = (1.0, 2.0)

    def __init__(self, settings: LiveSettings, registry: SkillRegistry,
                 agent: QwenVisionAgent, executor: LiveExecutor, store: Store,
                 run_id: str | None = None):
        self.settings = settings
        self.registry = registry
        self.agent = agent
        self.executor = executor
        self.store = store
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        self.run_dir = settings.project_root / "runs" / self.run_id
        self.recorder = EvidenceRecorder(self.run_dir / "evidence", settings.jpeg_quality)
        self.state: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self._started = False
        self._stopped = False

    def _console(self, event: str, **payload: Any) -> None:
        """Emit one compact, machine-readable status line without sensitive data."""
        if not self.settings.console_status:
            return
        record = {
            "event": event,
            "run_id": self.run_id,
            "task_id": self.settings.task_id,
            **payload,
        }
        print(
            "[HarnessVLA] " + json.dumps(record, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    def _append_execution_log(self, kind: str, payload: dict[str, Any],
                              skill_id: str | None = None) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "wall_time_epoch": time.time(), "run_id": self.run_id,
            "task_id": self.settings.task_id, "skill_id": skill_id, "kind": kind,
            "payload": payload,
        }
        with (self.run_dir / "execution_events.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _capture(self, stage: str) -> EvidencePair:
        try:
            evidence = self.recorder.record(stage, self.executor.capture_frames())
        except LiveSafetyFault:
            raise
        except Exception as exc:
            raise LiveSafetyFault(f"camera capture failed: {exc}") from exc
        self.store.commit(self.state, "camera_evidence", evidence.audit_json(), self.state.get("current_skill"))
        return evidence

    def _commit(self, kind: str, payload: dict[str, Any], skill_id: str | None = None) -> None:
        self.store.commit(self.state, kind, payload, skill_id)
        self._append_execution_log(kind, payload, skill_id)

    def _normal_stop(self) -> None:
        if self._started and not self._stopped:
            self._stopped = True
            self._console(
                "controlled_shutdown_started",
                reason=self.state.get("stop_reason"),
                current_skill=self.state.get("current_skill"),
                next_action="safe_pose_then_disable",
            )
            self._commit("controlled_shutdown_started", {
                "reason": self.state.get("stop_reason"),
                "sequence": "pause_actions_then_safe_pose_then_disable",
            }, self.state.get("current_skill"))
            self.executor.normal_stop()
            self._commit("controlled_shutdown_completed", {
                "sequence": "safe_pose_reached_then_disabled",
            }, self.state.get("current_skill"))
            self._console(
                "controlled_shutdown_completed",
                current_skill=self.state.get("current_skill"),
                robot_enabled=False,
            )

    def _emergency_stop(self) -> None:
        if self._started and not self._stopped:
            self._stopped = True
            self._console(
                "emergency_shutdown_started",
                reason=self.state.get("stop_reason"),
                current_skill=self.state.get("current_skill"),
                next_action="immediate_disable",
            )
            self._commit("emergency_shutdown_started", {
                "reason": self.state.get("stop_reason"),
                "sequence": "immediate_disable_without_motion",
            }, self.state.get("current_skill"))
            self.executor.emergency_stop()

    def _call_qwen(self, operation: str, call: Callable[[], Any]) -> Any:
        for call_number in range(1, len(self.qwen_retry_delays_s) + 2):
            try:
                return call()
            except PlanningError as exc:
                retry_index = call_number - 1
                if (not _is_transient_qwen_error(exc) or
                        retry_index >= len(self.qwen_retry_delays_s)):
                    raise
                delay_s = self.qwen_retry_delays_s[retry_index]
                self._commit("qwen_transport_retry", {
                    "operation": operation,
                    "failed_call": call_number,
                    "next_call": call_number + 1,
                    "delay_s": delay_s,
                    "reason": str(exc),
                }, self.state.get("current_skill"))
                self._console(
                    "qwen_retry",
                    operation=operation,
                    failed_call=call_number,
                    next_call=call_number + 1,
                    delay_s=delay_s,
                    reason=str(exc),
                )
                time.sleep(delay_s)
        raise AssertionError("unreachable Qwen retry state")

    def _assess_initial_scene(
            self, first: EvidencePair) -> tuple[set[str], EvidencePair]:
        """Accept one visual assessment of goals before any PI0 action."""
        evidence = first
        try:
            decisions = self._call_qwen(
                "initial_scene_assessment",
                lambda: self.agent.assess_scene_goals(evidence),
            )
        except PlanningError as exc:
            raise LiveT1Error(
                f"Qwen initial scene assessment failed: {exc}"
            ) from exc
        confirmed = {
            predicate for predicate, decision in decisions.items()
            if decision.value == "true"
        }
        for predicate in sorted(T1_GOALS):
            verdict = decisions[predicate]
            self._commit("qwen_initial_scene_verifier_result", {
                **verdict.audit_json(),
                "phase": "single",
                "evidence": evidence.audit_json(),
                "pi0_actions_sent": 0,
            })
            self._console(
                "qwen_initial_scene_verification",
                phase="single",
                predicate=verdict.predicate,
                value=verdict.value,
                confidence=verdict.confidence,
                reason=verdict.reason,
                latency_s=verdict.latency_s,
                pi0_actions_sent=0,
            )
        self.state["verified_predicates"] = sorted(confirmed)
        self._commit("initial_scene_assessed", {
            "verified_predicates": sorted(confirmed),
            "unverified_predicates": sorted(T1_GOALS - confirmed),
            "source": "fresh_dual_camera_qwen_single_assessment",
            "pi0_actions_sent": 0,
        })
        self._console(
            "initial_scene_assessed",
            verified_predicates=sorted(confirmed),
            unverified_predicates=sorted(T1_GOALS - confirmed),
            next_action="plan_from_current_scene",
            pi0_actions_sent=0,
        )
        return confirmed, evidence

    def run(self) -> dict[str, Any]:
        redacted_config = {
                           "task_id": self.settings.task_id,
                           "task_command": self.settings.task_command,
                           "verification_interval_actions":
                               self.settings.verification_interval_actions,
                           "wait_pose_min_actions": self.settings.wait_pose_min_actions,
                           "wait_pose_stable_intervals":
                               self.settings.wait_pose_stable_intervals,
                           "wait_pose_joint_range": self.settings.wait_pose_joint_range,
                           "wait_pose_gripper_range": self.settings.wait_pose_gripper_range,
                           "model": self.agent.config.model}
        self.state = {
            "run_id": self.run_id, "task_id": self.settings.task_id,
            "status": "starting", "plan": [],
            "verified_predicates": [], "skill_progress": {},
            "current_skill": None,
            "stop_reason": None,
            "config_hash": _sha256(json.dumps(redacted_config, sort_keys=True).encode()),
            "code_hash": code_hash(), "checkpoint_hash": self.settings.checkpoint_hash,
        }
        self.store.create(self.state)
        self._append_execution_log("run_started", {
            "task_id": self.settings.task_id,
            "verification_interval_actions": self.settings.verification_interval_actions,
            "checkpoint_hash": self.settings.checkpoint_hash,
        })
        self._console(
            "run_started",
            task_command=self.settings.task_command,
            verification_interval_actions=self.settings.verification_interval_actions,
            subtask_timeout_s=self.settings.execution_timeout_s,
        )
        verified: set[str] = set()
        skill_progress: dict[str, dict[str, int]] = {}
        try:
            self._started = True
            try:
                self.executor.start()
            except Exception as exc:
                raise LiveSafetyFault(f"hardware startup failed: {exc}") from exc
            self.state["status"] = "running"
            self._commit("hardware_ready", {"motion_authorized": True, "can_port": self.settings.can_port})
            self._console("hardware_ready", can_port=self.settings.can_port)
            planning_evidence = self._capture("initial_planning")
            verified, planning_evidence = self._assess_initial_scene(planning_evidence)
            while True:
                try:
                    decision = self._call_qwen(
                        "planner",
                        lambda: self.agent.choose_next(
                            planning_evidence, verified, skill_progress, self.history,
                        ),
                    )
                except PlanningError as exc:
                    raise LiveT1Error(f"Qwen planning failed: {exc}") from exc
                self._commit("qwen_planner_result", decision.audit_json())
                self._console(
                    "qwen_planner_result",
                    decision=decision.decision,
                    skill_id=decision.skill_id,
                    reason=decision.reason,
                    latency_s=decision.latency_s,
                )
                self.history.append({"type": "planner", "decision": decision.decision,
                                     "skill_id": decision.skill_id, "reason": decision.reason})
                if decision.decision == "fail":
                    self._console("planner_failed", reason=decision.reason)
                    raise LiveT1Error(f"Qwen declared task failure: {decision.reason}")
                if decision.decision == "finish":
                    self.state["status"] = "complete"
                    self.state["stop_reason"] = "all_t1_goals_visually_verified"
                    self._commit("run_completed", {"verified_predicates": sorted(verified)})
                    self._console(
                        "run_completed",
                        verified_predicates=sorted(verified),
                    )
                    self._normal_stop()
                    return dict(self.state)

                skill = self.registry.by_id()[decision.skill_id]
                self.state["current_skill"] = skill.id
                self.state["plan"].append(skill.id)
                progress = skill_progress.setdefault(skill.id, {"checks": 0, "actions": 0})
                self._console(
                    "skill_started",
                    skill_id=skill.id,
                    policy_task=skill.policy_task,
                    success_predicate=skill.success_predicate,
                    actions_already_executed=progress["actions"],
                )
                skill_execution_elapsed_s = 0.0
                consecutive_unknown = 0
                stationary_intervals = 0
                while True:
                    remaining_skill_time = (
                        self.settings.execution_timeout_s - skill_execution_elapsed_s
                    )
                    if remaining_skill_time <= 0:
                        raise LiveT1Error(
                            f"PI0 skill timed out before visual completion: {skill.id}"
                        )
                    check_number = progress["checks"] + 1
                    before = self._capture(f"{skill.id}_before_check_{check_number}")
                    self._console(
                        "pi0_interval_started",
                        skill_id=skill.id,
                        policy_task=skill.policy_task,
                        check_number=check_number,
                        action_start=progress["actions"] + 1,
                        action_target_end=(
                            progress["actions"]
                            + self.settings.verification_interval_actions
                        ),
                        configured_interval_actions=(
                            self.settings.verification_interval_actions
                        ),
                        remaining_skill_time_s=round(remaining_skill_time, 3),
                    )
                    try:
                        report = self.executor.execute(
                            skill, self.settings.verification_interval_actions,
                            remaining_skill_time,
                        )
                    except TimeoutError as exc:
                        raise LiveT1Error(
                            f"PI0 skill reached the controlled execution timeout: {skill.id}"
                        ) from exc
                    except LiveSafetyFault:
                        raise
                    except Exception as exc:
                        raise LiveSafetyFault(
                            f"PI0 execution failed for {skill.id}: {exc}"
                        ) from exc
                    skill_execution_elapsed_s += report.elapsed_s
                    progress["checks"] = check_number
                    progress["actions"] += report.actions_sent
                    self.state["skill_progress"] = {
                        key: dict(value) for key, value in skill_progress.items()
                    }
                    stationary = _interval_is_stationary(
                        report.action_records, self.settings.wait_pose_joint_range,
                        self.settings.wait_pose_gripper_range,
                    )
                    if progress["actions"] >= self.settings.wait_pose_min_actions:
                        stationary_intervals = stationary_intervals + 1 if stationary else 0
                    else:
                        stationary_intervals = 0
                    self._commit("pi0_verification_interval_executed", {
                        "skill_id": skill.id, "policy_task": skill.policy_task,
                        "check_number": check_number, "actions_sent": report.actions_sent,
                        "verification_interval_actions":
                            self.settings.verification_interval_actions,
                        "skill_actions_total": progress["actions"],
                        "skill_execution_elapsed_s": skill_execution_elapsed_s,
                        "elapsed_s": report.elapsed_s,
                        "timed_out": report.timed_out,
                        "wait_pose_stationary": stationary,
                        "wait_pose_stationary_intervals": stationary_intervals,
                        "actions": list(report.action_records),
                    }, skill.id)
                    self._console(
                        "pi0_interval_completed",
                        skill_id=skill.id,
                        policy_task=skill.policy_task,
                        check_number=check_number,
                        actions_sent=report.actions_sent,
                        configured_interval_actions=(
                            self.settings.verification_interval_actions
                        ),
                        skill_actions_total=progress["actions"],
                        interval_elapsed_s=round(report.elapsed_s, 3),
                        skill_execution_elapsed_s=round(skill_execution_elapsed_s, 3),
                        wait_pose_stationary=stationary,
                        wait_pose_stationary_intervals=stationary_intervals,
                        timed_out=report.timed_out,
                    )
                    if report.timed_out:
                        self._commit("skill_execution_timeout", {
                            "skill_id": skill.id,
                            "skill_actions_total": progress["actions"],
                            "skill_execution_elapsed_s": skill_execution_elapsed_s,
                            "shutdown": "controlled_safe_pose_then_disable",
                        }, skill.id)
                        self._console(
                            "skill_timeout",
                            skill_id=skill.id,
                            skill_actions_total=progress["actions"],
                            skill_execution_elapsed_s=round(
                                skill_execution_elapsed_s, 3
                            ),
                            next_action="safe_pose_then_disable",
                        )
                        raise LiveT1Error(
                            f"PI0 skill timed out before visual completion: {skill.id}"
                        )
                    after = self._capture(f"{skill.id}_after_check_{check_number}")
                    try:
                        verdict = self._call_qwen(
                            "verifier",
                            lambda: self.agent.verify(skill, before, after),
                        )
                    except PlanningError as exc:
                        raise LiveT1Error(f"Qwen verification failed: {exc}") from exc
                    self._commit("qwen_verifier_result", {
                        **verdict.audit_json(), "phase": "primary",
                        "before": before.audit_json(), "after": after.audit_json(),
                    }, skill.id)
                    self._console(
                        "qwen_verification",
                        phase="primary",
                        skill_id=skill.id,
                        check_number=check_number,
                        skill_actions_total=progress["actions"],
                        predicate=verdict.predicate,
                        value=verdict.value,
                        confidence=verdict.confidence,
                        reason=verdict.reason,
                        latency_s=verdict.latency_s,
                    )
                    confirmed = verdict.value == "true"
                    final_verdict = verdict
                    for confirmation_index in range(self.settings.verification_retries):
                        if not confirmed:
                            break
                        confirm = self._capture(
                            f"{skill.id}_confirm_{check_number}_{confirmation_index + 1}"
                        )
                        try:
                            final_verdict = self._call_qwen(
                                "verifier_confirmation",
                                lambda: self.agent.verify(skill, before, confirm),
                            )
                        except PlanningError as exc:
                            raise LiveT1Error(f"Qwen confirmation failed: {exc}") from exc
                        self._commit("qwen_verifier_result", {
                            **final_verdict.audit_json(), "phase": "confirmation",
                            "before": before.audit_json(), "after": confirm.audit_json(),
                        }, skill.id)
                        self._console(
                            "qwen_verification",
                            phase="confirmation",
                            confirmation_index=confirmation_index + 1,
                            skill_id=skill.id,
                            check_number=check_number,
                            skill_actions_total=progress["actions"],
                            predicate=final_verdict.predicate,
                            value=final_verdict.value,
                            confidence=final_verdict.confidence,
                            reason=final_verdict.reason,
                            latency_s=final_verdict.latency_s,
                        )
                        confirmed = final_verdict.value == "true"
                        after = confirm
                    if confirmed:
                        verified.difference_update(skill.removes)
                        verified.update(skill.adds)
                        self.state["verified_predicates"] = sorted(verified)
                        self._commit("skill_confirmed", {
                            "skill_id": skill.id, "verified_predicates": sorted(verified),
                            "consecutive_true": 1 + self.settings.verification_retries,
                            "skill_actions_total": progress["actions"],
                        }, skill.id)
                        self._console(
                            "skill_completed",
                            skill_id=skill.id,
                            skill_actions_total=progress["actions"],
                            verified_predicates=sorted(verified),
                        )
                        self.history.append({
                            "type": "verification", "skill_id": skill.id,
                            "value": "true", "reason": final_verdict.reason,
                        })
                        self.state["current_skill"] = None
                        planning_evidence = after
                        break

                    if final_verdict.value == "unknown":
                        consecutive_unknown += 1
                    else:
                        consecutive_unknown = 0
                    wait_pose_stop = (
                        stationary_intervals >= self.settings.wait_pose_stable_intervals
                    )
                    self._commit("skill_not_yet_confirmed", {
                        "skill_id": skill.id, "value": final_verdict.value,
                        "reason": final_verdict.reason, "check_number": check_number,
                        "skill_actions_total": progress["actions"],
                        "next_action": ("controlled_safe_pose_then_disable" if wait_pose_stop
                                        else "continue_same_pi0_skill"),
                    }, skill.id)
                    self._console(
                        "skill_incomplete",
                        skill_id=skill.id,
                        check_number=check_number,
                        skill_actions_total=progress["actions"],
                        value=final_verdict.value,
                        confidence=final_verdict.confidence,
                        reason=final_verdict.reason,
                        next_action=("safe_pose_then_disable" if wait_pose_stop
                                     else "continue_same_pi0_skill"),
                    )
                    self.history.append({
                        "type": "verification", "skill_id": skill.id,
                        "value": final_verdict.value, "reason": final_verdict.reason,
                    })
                    if wait_pose_stop:
                        self._commit("skill_wait_pose_detected", {
                            "skill_id": skill.id,
                            "skill_actions_total": progress["actions"],
                            "stationary_intervals": stationary_intervals,
                            "joint_range_limit": self.settings.wait_pose_joint_range,
                            "gripper_range_limit": self.settings.wait_pose_gripper_range,
                            "shutdown": "controlled_safe_pose_then_disable",
                        }, skill.id)
                        self._console(
                            "skill_wait_pose_detected",
                            skill_id=skill.id,
                            skill_actions_total=progress["actions"],
                            stationary_intervals=stationary_intervals,
                            next_action="safe_pose_then_disable",
                        )
                        raise LiveT1Error(
                            f"PI0 settled at its wait pose without visual completion: {skill.id}"
                        )
                    if consecutive_unknown >= 2:
                        raise LiveT1Error(f"visual result remained unknown: {skill.id}")
        except LiveSafetyFault as exc:
            self.state["status"] = "safety_stopped"
            self.state["stop_reason"] = str(exc)
            self._commit("safety_stop", {"reason": str(exc)}, self.state.get("current_skill"))
            self._console(
                "safety_stop",
                current_skill=self.state.get("current_skill"),
                reason=str(exc),
            )
            self._emergency_stop()
            raise
        except KeyboardInterrupt:
            self.state["status"] = "operator_stopped"
            self.state["stop_reason"] = "operator_interrupt"
            self._commit("operator_stop", {
                "reason": "operator_interrupt",
                "shutdown": "controlled_safe_pose_then_disable",
            }, self.state.get("current_skill"))
            self._console(
                "operator_stop",
                current_skill=self.state.get("current_skill"),
                reason="operator_interrupt",
                next_action="safe_pose_then_disable",
            )
            self._normal_stop()
            raise
        except Exception as exc:
            self.state["status"] = "stopped"
            self.state["stop_reason"] = str(exc)
            self._commit("run_stopped", {"reason": str(exc)}, self.state.get("current_skill"))
            self._console(
                "run_stopped",
                current_skill=self.state.get("current_skill"),
                reason=str(exc),
            )
            self._normal_stop()
            raise
        finally:
            if self._started and not self._stopped:
                self._normal_stop()
            try:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                (self.run_dir / "run_summary.json").write_text(
                    json.dumps(self.state, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except OSError:
                # Incremental SQLite/JSONL/action logs are already durable.  A
                # summary write must never hide the original stop condition.
                pass


class PI0ServerProcess:
    def __init__(self, settings: LiveSettings, log_path: Path):
        self.settings = settings
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()

    def start(self, timeout_s: float = 300.0) -> None:
        host, port = self.settings.server_address.rsplit(":", 1)
        command = [
            sys.executable, "-u", "-m", "harnessvla.pi0_deploy", "server",
            "--checkpoint", str(self.settings.checkpoint), "--device", self.settings.device,
            "--host", host, "--port", port, "--fps", str(self.settings.fps),
            "--expected-model-sha256", self.settings.checkpoint_hash,
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            command, cwd=self.settings.project_root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

        def read_output() -> None:
            assert self.process is not None and self.process.stdout is not None
            with self.log_path.open("a", encoding="utf-8") as destination:
                for line in self.process.stdout:
                    destination.write(line)
                    destination.flush()
                    self._lines.put(line)
            self._lines.put(None)

        self._reader = threading.Thread(target=read_output, daemon=True)
        self._reader.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                if self.process.poll() is not None:
                    break
                continue
            if line is None:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("status") == "ready":
                if message.get("model_sha256") != self.settings.checkpoint_hash:
                    self.stop()
                    raise LiveT1Error("PI0 server reported an unexpected checkpoint hash")
                return
        exit_code = self.process.poll()
        self.stop()
        raise LiveT1Error(f"PI0 server did not become ready (exit={exit_code})")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2.0)
        if self._reader is not None:
            self._reader.join(timeout=2.0)


class PiperPI0Session:
    """One Piper/PI0 client connection with explicit execute/pause phases."""

    def __init__(self, settings: LiveSettings, action_log: Path):
        self.settings = settings
        self.action_log = action_log
        self.client: Any = None
        self.robot: Any = None
        self.receiver: threading.Thread | None = None
        self._stopped = False

    def start(self) -> None:
        import grpc
        from lerobot.async_inference import robot_client as robot_client_module
        from lerobot.async_inference.configs import RobotClientConfig
        from lerobot.async_inference.robot_client import RobotClient
        from lerobot.cameras.realsense import RealSenseCameraConfig
        from lerobot.robots.piper.config_piper import PIPERConfig
        from lerobot.robots.piper.piper import Piper

        from .pi0_deploy import (
            _execute_piper_class,
            _preview_client_class,
            _validate_device,
        )

        contract = inspect_checkpoint(self.settings.checkpoint, hash_model=True)
        if contract.model_sha256 != self.settings.checkpoint_hash:
            raise LiveT1Error("checkpoint SHA-256 does not match pi0.expected_model_sha256")
        expected_shape = (3, self.settings.camera_height, self.settings.camera_width)
        if contract.top_image_shape != expected_shape or contract.wrist_image_shape != expected_shape:
            raise LiveT1Error("camera dimensions do not match the PI0 checkpoint")
        _validate_device(self.settings.device)
        channel = grpc.insecure_channel(self.settings.server_address)
        try:
            grpc.channel_ready_future(channel).result(timeout=10.0)
        except grpc.FutureTimeoutError as exc:
            raise LiveT1Error("local PI0 server is unavailable") from exc
        finally:
            channel.close()
        top_key = contract.top_image_key.removeprefix("observation.images.")
        wrist_key = contract.wrist_image_key.removeprefix("observation.images.")
        cameras = {
            top_key: RealSenseCameraConfig(serial_number_or_name=self.settings.top_camera_serial,
                                           fps=self.settings.camera_fps,
                                           width=self.settings.camera_width,
                                           height=self.settings.camera_height),
            wrist_key: RealSenseCameraConfig(serial_number_or_name=self.settings.wrist_camera_serial,
                                             fps=self.settings.camera_fps,
                                             width=self.settings.camera_width,
                                             height=self.settings.camera_height),
        }
        piper_config = PIPERConfig(
            can_port=self.settings.can_port, cameras=cameras, home_position=[0.0] * 7,
            reset_hz=self.settings.reset_hz, reset_duration_s=self.settings.reset_duration_s,
            max_joint_step_rad=self.settings.reset_max_joint_step,
            open_gripper_on_init=True, gripper_open_range=self.settings.home_gripper,
        )
        execute_type = _execute_piper_class(Piper)
        self.robot = execute_type(
            piper_config, contract, self.action_log, self.settings.max_joint_step,
            self.settings.max_gripper_step,
            self.settings.verification_interval_actions, 10,
        )
        client_config = RobotClientConfig(
            policy_type="pi0", pretrained_name_or_path=str(self.settings.checkpoint),
            robot=piper_config, actions_per_chunk=contract.chunk_size,
            task="HarnessVLA dynamic registered skill", server_address=self.settings.server_address,
            policy_device=self.settings.device, client_device="cpu", chunk_size_threshold=0.5,
            fps=self.settings.fps, aggregate_fn_name="weighted_average",
            debug_visualize_queue_size=False,
        )
        injected_type = _preview_client_class(RobotClient, robot_client_module)

        class PausableClient(injected_type):
            def __init__(inner_self, config, robot):
                super().__init__(config, robot)
                inner_self.execution_paused = threading.Event()
                inner_self.execution_paused.set()
                inner_self.discard_next_chunk = False

            def _aggregate_action_queues(inner_self, incoming_actions, aggregate_fn=None):
                if inner_self.execution_paused.is_set():
                    return
                if inner_self.discard_next_chunk:
                    inner_self.discard_next_chunk = False
                    inner_self.action_chunk_size = 1
                    inner_self.must_go.set()
                    return
                return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

            def actions_available(inner_self):
                return (not inner_self.execution_paused.is_set() and
                        super().actions_available())

            def clear_and_pause(inner_self):
                inner_self.execution_paused.set()
                with inner_self.action_queue_lock:
                    inner_self.action_queue = Queue()
                inner_self.action_chunk_size = -1

            def resume_fresh(inner_self):
                with inner_self.action_queue_lock:
                    inner_self.action_queue = Queue()
                inner_self.action_chunk_size = -1
                inner_self.discard_next_chunk = True
                inner_self.must_go.set()
                inner_self.execution_paused.clear()

        self.client = PausableClient(client_config, self.robot)
        if not self.client.start():
            raise LiveT1Error("PI0 client handshake failed")
        self.receiver = threading.Thread(target=self.client.receive_actions, daemon=True)
        self.receiver.start()
        try:
            self.client.start_barrier.wait(timeout=5.0)
        except threading.BrokenBarrierError as exc:
            raise LiveT1Error("PI0 action receiver failed to start") from exc

    def capture_frames(self) -> dict[str, Any]:
        if self.client is None or self.robot is None:
            raise LiveSafetyFault("Piper session is not started")
        self.client.clear_and_pause()
        try:
            observation = self.robot.get_observation()
            return {"cam_top": observation["cam_top"], "cam_left": observation["cam_left"]}
        except Exception as exc:
            raise LiveSafetyFault(f"dual-camera capture failed: {exc}") from exc

    def execute(self, skill: ActionSkill, action_budget: int, timeout_s: float) -> ExecutionReport:
        if self.client is None or self.robot is None:
            raise LiveSafetyFault("Piper session is not started")
        self.robot.reset_action_budget(action_budget)
        self.client.resume_fresh()
        started = time.monotonic()
        deadline = started + timeout_s
        actions = 0
        timed_out = False
        try:
            while actions < action_budget:
                cycle = time.perf_counter()
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                if self.client.actions_available():
                    self.client.control_loop_action()
                    actions += 1
                if self.client._ready_to_send_observation():
                    observation = self.client.control_loop_observation(skill.policy_task)
                    if observation is None:
                        raise RuntimeError("PI0 observation upload failed")
                time.sleep(max(0.0, self.client.config.environment_dt -
                               (time.perf_counter() - cycle)))
        finally:
            self.client.clear_and_pause()
        records = tuple(self.robot.drain_action_records())
        if len(records) != actions:
            raise LiveSafetyFault("action audit count does not match commands sent")
        return ExecutionReport(actions, records, time.monotonic() - started, timed_out)

    def normal_stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.client is not None:
            self.client.clear_and_pause()
            self.client.stop()  # Piper driver moves to its safe-disable pose, then disables.
        if self.receiver is not None:
            self.receiver.join(timeout=10.0)

    def emergency_stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.client is not None:
            self.client.clear_and_pause()
            self.client.shutdown_event.set()
        if self.robot is not None:
            self.robot.emergency_disconnect()  # immediate disable; deliberately no homing
        if self.client is not None:
            self.client.channel.close()
        if self.receiver is not None:
            self.receiver.join(timeout=10.0)


def run_live_t1(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    settings = load_live_settings(config, source_path)
    registry = load_action_skills()
    if not registry.execution_enabled:
        raise ValueError("action skill registry execution_enabled is false")
    qwen = QwenConfig.load(source_path)
    if not qwen.api_key.strip():
        raise ValueError(f"Qwen API key is empty; fill qwen.api_key in {source_path}")
    run_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    run_dir = settings.project_root / "runs" / run_id
    server = PI0ServerProcess(settings, run_dir / "pi0_server.log")
    store = Store(settings.db_path)
    try:
        server.start()
        executor = PiperPI0Session(settings, run_dir / "pi0_actions.jsonl")
        orchestrator = LiveT1Orchestrator(
            settings, registry,
            QwenVisionAgent(qwen, registry, settings.task_command),
            executor, store, run_id,
        )
        try:
            return orchestrator.run()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise LiveT1Error(f"run_id={run_id}: {exc}") from exc
    finally:
        server.stop()
        store.close()


class DualRealSenseSource:
    """Dual-camera source used by the no-motion staged Qwen check."""

    def __init__(self, settings: LiveSettings):
        self.settings = settings
        self.cameras: dict[str, Any] = {}

    def start(self) -> None:
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

        self.cameras = {
            "cam_top": RealSenseCamera(RealSenseCameraConfig(
                serial_number_or_name=self.settings.top_camera_serial,
                fps=self.settings.camera_fps, width=self.settings.camera_width,
                height=self.settings.camera_height,
            )),
            "cam_left": RealSenseCamera(RealSenseCameraConfig(
                serial_number_or_name=self.settings.wrist_camera_serial,
                fps=self.settings.camera_fps, width=self.settings.camera_width,
                height=self.settings.camera_height,
            )),
        }
        try:
            for camera in self.cameras.values():
                camera.connect()
        except Exception:
            self.stop()
            raise

    def capture_frames(self) -> dict[str, Any]:
        return {name: camera.async_read() for name, camera in self.cameras.items()}

    def stop(self) -> None:
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()


def run_visual_check(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Exercise real cameras plus Qwen planning/verification without touching Piper."""
    settings = load_live_settings(config, source_path, require_motion=False)
    registry = load_action_skills()
    qwen = QwenConfig.load(source_path)
    if not qwen.api_key.strip():
        raise ValueError(f"Qwen API key is empty; fill qwen.api_key in {source_path}")
    run_id = time.strftime("visual_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    recorder = EvidenceRecorder(
        settings.project_root / "runs" / run_id / "evidence", settings.jpeg_quality,
    )
    source = DualRealSenseSource(settings)
    try:
        source.start()
        before = recorder.record("planning", source.capture_frames())
        agent = QwenVisionAgent(qwen, registry, settings.task_command)
        decision = agent.choose_next(before, set(), {}, [])
        result: dict[str, Any] = {
            "run_id": run_id, "motion_authorized": False,
            "planner": decision.audit_json(), "planning_evidence": before.audit_json(),
        }
        if decision.decision == "execute" and decision.skill_id is not None:
            after = recorder.record("verification_no_motion", source.capture_frames())
            verdict = agent.verify(registry.by_id()[decision.skill_id], before, after)
            result["verifier"] = verdict.audit_json()
            result["verification_evidence"] = after.audit_json()
        return result
    finally:
        source.stop()


def run_one_skill_check(config: dict[str, Any], source_path: Path,
                        skill_id: str) -> dict[str, Any]:
    """Run one bounded registered skill, verify twice, then home and disable."""
    settings = load_live_settings(config, source_path)
    registry = load_action_skills()
    skill = registry.by_id().get(skill_id)
    if skill is None or not skill.t1_allowed:
        raise ValueError(f"skill is not available for the T1 live check: {skill_id}")
    qwen = QwenConfig.load(source_path)
    run_id = time.strftime("skill_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    run_dir = settings.project_root / "runs" / run_id
    recorder = EvidenceRecorder(run_dir / "evidence", settings.jpeg_quality)
    server = PI0ServerProcess(settings, run_dir / "pi0_server.log")
    session = PiperPI0Session(settings, run_dir / "pi0_actions.jsonl")
    started = False
    try:
        server.start()
        try:
            session.start()
            started = True
        except Exception:
            session.emergency_stop()
            raise
        before = recorder.record("before", session.capture_frames())
        try:
            report = session.execute(
                skill, settings.verification_interval_actions,
                settings.execution_timeout_s,
            )
        except Exception:
            session.emergency_stop()
            started = False
            raise
        after = recorder.record("after", session.capture_frames())
        agent = QwenVisionAgent(qwen, registry, settings.task_command)
        primary = agent.verify(skill, before, after)
        confirm_evidence = recorder.record("confirm", session.capture_frames())
        confirmation = agent.verify(skill, before, confirm_evidence)
        return {
            "run_id": run_id, "skill_id": skill.id,
            "actions_sent": report.actions_sent, "elapsed_s": report.elapsed_s,
            "primary_verification": primary.audit_json(),
            "confirmation_verification": confirmation.audit_json(),
            "confirmed": primary.value == confirmation.value == "true",
            "before": before.audit_json(), "after": after.audit_json(),
            "confirmation": confirm_evidence.audit_json(),
        }
    finally:
        if started:
            session.normal_stop()
        server.stop()
