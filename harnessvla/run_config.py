"""Launch HarnessVLA modes from one explicit, versioned YAML file."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .cli import main as cli_main
from .pi0_deploy import main as pi0_deploy_main
from .qwen_planner import main as qwen_planner_main

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_CONFIG = PROJECT_ROOT / "config" / "piper_harness.yaml"
RUN_MODES = frozenset({
    "mock_t1", "pi0_check", "pi0_server", "pi0_preview", "pi0_execute", "qwen_plan",
    "qwen_pi0",
})


def _project_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _positive_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def load_run_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("YAML run config requires PyYAML; install project dependencies") from exc
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read run config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML run config {path}: {exc}") from exc
    raw = _object(raw, "run config")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("run config section names must be strings")
    allowed = {"schema_version", "run", "pi0", "qwen"}
    if set(raw) - allowed:
        raise ValueError(f"unknown run config sections: {sorted(set(raw) - allowed)}")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported run config schema_version")
    run = _object(raw.get("run"), "run")
    if not isinstance(run.get("mode"), str) or run["mode"] not in RUN_MODES:
        raise ValueError(f"run.mode must be one of: {', '.join(sorted(RUN_MODES))}")
    return raw


def _pi0_args(config: dict[str, Any], mode: str) -> list[str]:
    pi0 = _object(config.get("pi0"), "pi0")
    checkpoint = _project_path(pi0.get("checkpoint"), "pi0.checkpoint")
    task = _text(pi0.get("task"), "pi0.task")
    device = _text(pi0.get("device"), "pi0.device")
    address = _text(pi0.get("server_address"), "pi0.server_address")
    host, separator, port_text = address.rpartition(":")
    if not separator or host != "127.0.0.1" or not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
        raise ValueError("pi0.server_address must be 127.0.0.1:<port>")
    if mode == "pi0_check":
        args = ["check", "--checkpoint", str(checkpoint), "--task", task, "--device", device]
        for field, flag in (("hash_model", "--hash-model"), ("check_processors", "--processors"),
                            ("smoke_inference", "--smoke-inference")):
            value = pi0.get(field, False)
            if type(value) is not bool:
                raise ValueError(f"pi0.{field} must be true or false")
            if value:
                args.append(flag)
        return args
    fps = _positive_int(pi0.get("fps"), "pi0.fps")
    if mode == "pi0_server":
        return ["server", "--checkpoint", str(checkpoint), "--device", device,
                "--host", host, "--port", port_text, "--fps", str(fps)]
    cameras = _object(pi0.get("cameras"), "pi0.cameras")
    if set(cameras) != {"cam_top", "cam_left"}:
        raise ValueError("pi0.cameras must contain exactly cam_top and cam_left")
    top = _object(cameras["cam_top"], "pi0.cameras.cam_top")
    wrist = _object(cameras["cam_left"], "pi0.cameras.cam_left")
    for name, camera in (("cam_top", top), ("cam_left", wrist)):
        _text(camera.get("serial_number_or_name"), f"pi0.cameras.{name}.serial_number_or_name")
        for dimension in ("width", "height", "fps"):
            _positive_int(camera.get(dimension), f"pi0.cameras.{name}.{dimension}")
    if (top["width"], top["height"], top["fps"]) != (wrist["width"], wrist["height"], wrist["fps"]):
        raise ValueError("pi0 preview requires both cameras to have the same width, height and fps")
    if top["fps"] < fps:
        raise ValueError("camera fps must be at least pi0.fps")
    can_port = _text(pi0.get("can_port"), "pi0.can_port")
    joint_step = _positive_number(pi0.get("max_joint_step"), "pi0.max_joint_step")
    gripper_step = _positive_number(pi0.get("max_gripper_step"), "pi0.max_gripper_step")
    command = "preview-client"
    if mode == "pi0_execute":
        if pi0.get("motion_enabled") is not True:
            raise ValueError("pi0_execute requires pi0.motion_enabled: true")
        command = "execute-client"
    args = [
        command, "--checkpoint", str(checkpoint), "--task", task,
        "--server-address", address, "--device", device, "--can-port", can_port,
        "--top-camera-serial", top["serial_number_or_name"],
        "--wrist-camera-serial", wrist["serial_number_or_name"],
        "--camera-width", str(top["width"]), "--camera-height", str(top["height"]),
        "--camera-fps", str(top["fps"]), "--fps", str(fps),
        "--max-joint-step", str(joint_step), "--max-gripper-step", str(gripper_step),
    ]
    if mode == "pi0_execute":
        max_actions = _positive_int(pi0.get("max_actions"), "pi0.max_actions")
        if max_actions > 50:
            raise ValueError("pi0.max_actions must not exceed 50")
        args.extend(["--max-actions", str(max_actions)])
    return args


def run_from_config(config: dict[str, Any], source_path: Path = DEFAULT_RUN_CONFIG) -> int:
    run = _object(config["run"], "run")
    mode = run["mode"]
    if not isinstance(mode, str) or mode not in RUN_MODES:
        raise ValueError("unsupported run.mode")
    if mode == "mock_t1":
        if run.get("task_id") != "T1":
            raise ValueError("mock_t1 requires run.task_id: T1")
        db = _project_path(run.get("db"), "run.db")
        mock_config = _project_path(run.get("mock_config"), "run.mock_config")
        return cli_main(["--db", str(db), "start", "--config", str(mock_config), "--task", "T1"])
    if mode in {"pi0_check", "pi0_server", "pi0_preview", "pi0_execute"}:
        return pi0_deploy_main(_pi0_args(config, mode))
    qwen = _object(config.get("qwen"), "qwen")
    if not isinstance(qwen.get("api_key"), str) or not qwen["api_key"].strip():
        raise ValueError(f"Qwen API key is empty; fill qwen.api_key in {source_path}")
    if mode == "qwen_pi0":
        from .live_t1 import run_live_t1
        result = run_live_t1(config, source_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    task_id = run.get("task_id")
    if task_id not in {"T1", "T2", "T3"}:
        raise ValueError("qwen_plan requires run.task_id T1, T2 or T3")
    verified = qwen.get("verified_predicates", [])
    if not isinstance(verified, list) or not all(isinstance(item, str) and item for item in verified):
        raise ValueError("qwen.verified_predicates must be a list of predicate names")
    args = ["plan", "--config", str(source_path), "--task-id", task_id]
    for item in verified:
        args.extend(["--verified", item])
    return qwen_planner_main(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run HarnessVLA from YAML; live PI0 motion requires two explicit gates"
    )
    parser.add_argument("--config_path", type=Path, default=DEFAULT_RUN_CONFIG)
    args = parser.parse_args(argv)
    path = args.config_path if args.config_path.is_absolute() else PROJECT_ROOT / args.config_path
    try:
        config = load_run_config(path)
        return run_from_config(config, path)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
