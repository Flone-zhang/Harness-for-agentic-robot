"""PI0 deployment entry point: local server and command-free Piper preview.

LeRobot is imported lazily so mock runs and metadata checks need no robot stack.
This module deliberately has no motor-enable, home, or physical send path.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle  # LeRobot's local gRPC handshake format; bind loopback only.
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

from .pi0 import DEFAULT_CHECKPOINT_DIR, ContractError, PI0Contract, inspect_checkpoint, preview_chunk

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8081


class ActionBudgetReached(RuntimeError):
    """Internal clean-stop signal after the configured live-action budget."""


def _checkpoint(args) -> tuple[Path, PI0Contract]:
    contract = inspect_checkpoint(args.checkpoint, hash_model=getattr(args, "hash_model", False))
    return Path(contract.checkpoint_path), contract


def _prepare_runtime(checkpoint: Path) -> None:
    # Saved tokenizer paths are relative. LeRobot also creates logs during import.
    (checkpoint / "logs").mkdir(exist_ok=True)
    os.chdir(checkpoint)


def _validate_device(device: str):
    import torch

    parsed = torch.device(device)
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for torch {torch.__version__}; use --device cpu for checks")
    return parsed


def _load_policy(checkpoint: Path, device: str):
    from lerobot.policies import PI0Config, make_pre_post_processors
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    _validate_device(device)
    config = PI0Config.from_pretrained(str(checkpoint), local_files_only=True)
    config.device = device
    config.gradient_checkpointing = False
    config.compile_model = False  # predictable startup, no first-frame compilation
    policy = PI0Policy.from_pretrained(
        str(checkpoint), config=config, local_files_only=True, strict=True,
    )
    policy.eval()
    overrides = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides=overrides,
        postprocessor_overrides=overrides,
    )
    return policy, preprocessor, postprocessor


def _processor_check(checkpoint: Path, contract: PI0Contract, task: str, device: str,
                     load_model: bool, smoke_inference: bool) -> dict:
    import torch
    from lerobot.policies import PI0Config, make_pre_post_processors

    selected_device = _validate_device(device)
    if load_model or smoke_inference:
        policy, preprocessor, postprocessor = _load_policy(checkpoint, str(selected_device))
    else:
        policy = None
        config = PI0Config.from_pretrained(str(checkpoint), local_files_only=True)
        config.device = str(selected_device)
        overrides = {"device_processor": {"device": str(selected_device)}}
        preprocessor, postprocessor = make_pre_post_processors(
            config, pretrained_path=str(checkpoint),
            preprocessor_overrides=overrides, postprocessor_overrides=overrides,
        )
    sample = {
        contract.top_image_key: torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        contract.wrist_image_key: torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        contract.state_key: torch.zeros((1, 7), dtype=torch.float32),
        "task": task,
        "robot_type": "piper",
    }
    processed = preprocessor(sample)
    required = (contract.top_image_key, contract.wrist_image_key, contract.state_key,
                "observation.language.tokens", "observation.language.attention_mask")
    missing = [key for key in required if key not in processed]
    if missing:
        raise RuntimeError(f"processor did not produce: {missing}")
    dummy = postprocessor(torch.zeros((1, 7), device=selected_device))
    if tuple(dummy.shape) != (1, 7) or not bool(torch.isfinite(dummy).all()):
        raise RuntimeError("postprocessor produced an invalid 7D action")
    output = {"processors": "ok", "model_strict_load": "ok" if policy else "not_run",
              "smoke_inference": "not_run"}
    if smoke_inference:
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(processed)
        if tuple(chunk.shape) != (1, contract.chunk_size, 7) or not bool(torch.isfinite(chunk).all()):
            raise RuntimeError(f"invalid model output shape or values: {tuple(chunk.shape)}")
        first_action = postprocessor(chunk[:, 0, :])
        if not bool(torch.isfinite(first_action).all()):
            raise RuntimeError("invalid postprocessed model action")
        output["smoke_inference"] = "ok"
        output["chunk_shape"] = list(chunk.shape)
    return output


def run_check(args) -> dict:
    checkpoint, contract = _checkpoint(args)
    result = contract.json()
    result["processors"] = "not_run"
    result["model_strict_load"] = "not_run"
    result["smoke_inference"] = "not_run"
    if args.processors or args.load_model or args.smoke_inference:
        previous_cwd = Path.cwd()
        try:
            _prepare_runtime(checkpoint)
            result.update(_processor_check(
                checkpoint, contract, args.task, args.device,
                args.load_model or args.smoke_inference, args.smoke_inference,
            ))
        finally:
            os.chdir(previous_cwd)
    return result


def _pinned_server_class(PolicyServer, RemotePolicyConfig, services_pb2, grpc):
    class PinnedPolicyServer(PolicyServer):
        def __init__(self, config, checkpoint: Path, device: str, contract: PI0Contract):
            super().__init__(config)
            self._checkpoint = checkpoint
            self._server_device = device
            self._contract = contract

        def ensure_policy_loaded(self):
            if self.policy is None or self.preprocessor is None or self.postprocessor is None:
                self.device = self._server_device
                self.policy, self.preprocessor, self.postprocessor = _load_policy(
                    self._checkpoint, self._server_device,
                )

        def SendPolicyInstructions(self, request, context):  # noqa: N802
            if not self.running:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "server is stopping")
            try:
                specs = pickle.loads(request.data)  # noqa: S301; loopback LeRobot protocol only
                if not isinstance(specs, RemotePolicyConfig) or specs.policy_type != "pi0":
                    raise ValueError("only LeRobot PI0 clients are accepted")
                if Path(specs.pretrained_name_or_path).expanduser().resolve() != self._checkpoint:
                    raise ValueError("client requested a different checkpoint")
                if not 1 <= specs.actions_per_chunk <= self._contract.chunk_size:
                    raise ValueError("actions_per_chunk outside checkpoint budget")
                self.device = self._server_device
                self.policy_type = "pi0"
                self.lerobot_features = specs.lerobot_features
                self.actions_per_chunk = specs.actions_per_chunk
                self.ensure_policy_loaded()
                self.policy.reset()
                return services_pb2.Empty()
            except Exception as exc:
                logging.exception("PI0 handshake rejected")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

    return PinnedPolicyServer


def run_server(args) -> dict:
    if args.host != LOOPBACK_HOST:
        raise ValueError("PI0 server must bind 127.0.0.1; LeRobot handshake uses pickle without authentication")
    if not 1 <= args.port <= 65535 or args.fps <= 0:
        raise ValueError("invalid port or fps")
    checkpoint, contract = _checkpoint(args)
    if not contract.model_sha256:
        raise AssertionError("server requires full model hash")
    if args.expected_model_sha256 and contract.model_sha256 != args.expected_model_sha256.lower():
        raise ValueError("model SHA-256 does not match --expected-model-sha256")
    _prepare_runtime(checkpoint)
    import grpc
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.transport import services_pb2, services_pb2_grpc

    _validate_device(args.device)
    server_config = PolicyServerConfig(
        host=args.host, port=args.port, fps=args.fps,
        inference_latency=1 / args.fps, obs_queue_timeout=args.obs_queue_timeout,
    )
    server_type = _pinned_server_class(PolicyServer, RemotePolicyConfig, services_pb2, grpc)
    policy_server = server_type(server_config, checkpoint, args.device, contract)
    policy_server.ensure_policy_loaded()  # fail before opening the port
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    bound = grpc_server.add_insecure_port(f"{args.host}:{args.port}")
    if bound == 0:
        raise RuntimeError("could not bind local PI0 server port")
    print(json.dumps({"status": "ready", "bind": f"{args.host}:{bound}",
                      "checkpoint": str(checkpoint), "model_sha256": contract.model_sha256,
                      "motion_authorized": False}), flush=True)
    grpc_server.start()
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        policy_server.stop()
        grpc_server.stop(grace=1.0).wait(timeout=2.0)
    return {"status": "stopped"}


def _preview_piper_class(Piper):
    class PreviewPiper(Piper):
        def __init__(self, config, contract: PI0Contract, log_path: Path,
                     max_joint_step: float, max_gripper_step: float, print_every: int):
            super().__init__(config)
            self._contract = contract
            self._log_path = log_path
            self._max_joint_step = max_joint_step
            self._max_gripper_step = max_gripper_step
            self._print_every = print_every
            self._action_count = 0

        def connect(self, calibrate: bool = False) -> None:  # noqa: ARG002
            # The bus constructor opens CAN receive; never bus.connect(enable=True).
            try:
                for camera in self.cameras.values():
                    camera.connect()
                self._is_connected = True
                self._is_calibrated = False
            except Exception:
                self.disconnect()
                raise

        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            # RobotClient calls this method, but this override never calls Piper.send_action.
            try:
                requested = [float(action[key]) for key in self._contract.action_names]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("PI0 action is missing or has nonnumeric joints") from exc
            reading = self.bus.read()
            current = [float(reading[f"joint_{index}"]) for index in range(1, 8)]
            report = preview_chunk(self._contract, current, [requested],
                                   self._max_joint_step, self._max_gripper_step)
            self._action_count += 1
            record = {"wall_time_epoch": time.time(), "action_number": self._action_count,
                      "requested_action": requested, "current_state": current, **report}
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self._action_count == 1 or self._action_count % self._print_every == 0 or report["violations"]:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if report["violations"]:
                raise RuntimeError("PI0 preview action rejected; client stopped without motion")
            return action

        def disconnect(self) -> None:
            # Piper.disconnect may move to a home pose; never call it in preview mode.
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            disconnect_port = getattr(self.bus.piper, "DisconnectPort", None)
            if callable(disconnect_port):
                disconnect_port()
            self._is_connected = False
            self._is_calibrated = False

    return PreviewPiper


def _preview_client_class(RobotClient, robot_client_module):
    class PreviewRobotClient(RobotClient):
        def __init__(self, config, robot):
            original = robot_client_module.make_robot_from_config
            robot_client_module.make_robot_from_config = lambda _config: robot
            try:
                super().__init__(config)
            finally:
                robot_client_module.make_robot_from_config = original

        def _ready_to_send_observation(self):
            # Avoid duplicate first-frame uploads while the initial action chunk is pending.
            if self.action_chunk_size < 0:
                return self.must_go.is_set()
            return super()._ready_to_send_observation()

    return PreviewRobotClient


def _execute_piper_class(Piper):
    class SafeExecutePiper(Piper):
        """Bounded Piper execution using the verified LeRobot driver path."""

        def __init__(self, config, contract: PI0Contract, log_path: Path,
                     max_joint_step: float, max_gripper_step: float,
                     max_actions: int, print_every: int):
            super().__init__(config)
            self._contract = contract
            self._log_path = log_path
            self._max_joint_step = max_joint_step
            self._max_gripper_step = max_gripper_step
            self._max_actions = max_actions
            self._print_every = print_every
            self._action_count = 0
            self._total_action_count = 0
            self._recent_action_records: list[dict] = []

        def reset_action_budget(self, max_actions: int) -> None:
            if type(max_actions) is not int or max_actions < 1:
                raise ValueError("action budget must be a positive integer")
            self._max_actions = max_actions
            self._action_count = 0
            self._recent_action_records.clear()

        def drain_action_records(self) -> list[dict]:
            records = list(self._recent_action_records)
            self._recent_action_records.clear()
            return records

        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            if self._action_count >= self._max_actions:
                raise ActionBudgetReached(
                    f"live action budget reached: {self._max_actions} commands"
                )
            try:
                requested = [float(action[key]) for key in self._contract.action_names]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("PI0 action is missing or has nonnumeric joints") from exc
            if len(requested) != 7 or not all(math.isfinite(value) for value in requested):
                raise RuntimeError("PI0 action must contain seven finite values")

            reading = self.bus.read()
            try:
                current = [float(reading[f"joint_{index}"]) for index in range(1, 8)]
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("Piper state is missing or has nonnumeric joints") from exc
            if not all(math.isfinite(value) for value in current):
                raise RuntimeError("Piper state contains a non-finite value")

            low = list(self._contract.training_quantile_low)
            high = list(self._contract.training_quantile_high)
            low[6] = max(0.0, low[6])
            high[6] = min(0.08, high[6])
            safe = [min(max(value, low[index]), high[index])
                    for index, value in enumerate(requested)]
            for index in range(6):
                safe[index] = min(max(safe[index], current[index] - self._max_joint_step),
                                  current[index] + self._max_joint_step)
            safe[6] = min(max(safe[6], max(0.0, current[6] - self._max_gripper_step)),
                          min(0.08, current[6] + self._max_gripper_step))
            safe_action = {key: safe[index]
                           for index, key in enumerate(self._contract.action_names)}

            delivered = super().send_action(safe_action)
            self._action_count += 1
            self._total_action_count += 1
            record = {
                "wall_time_epoch": time.time(),
                "action_number": self._action_count,
                "total_action_number": self._total_action_count,
                "action_budget": self._max_actions,
                "requested_action": requested,
                "current_state": current,
                "safe_action": safe,
                "clamped": any(abs(a - b) > 1e-6 for a, b in zip(requested, safe)),
                "status": "command_sent",
                "motion_authorized": True,
            }
            self._recent_action_records.append(record)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self._action_count == 1 or self._action_count % self._print_every == 0:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            return delivered

    return SafeExecutePiper


def _local_server_address(value: str) -> str:
    host, separator, port = value.rpartition(":")
    if not separator or host != LOOPBACK_HOST or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("server address must be 127.0.0.1:<port>")
    return value


def run_preview_client(args) -> dict:
    checkpoint, contract = _checkpoint(args)
    preview_log = args.preview_log.expanduser().resolve()
    _local_server_address(args.server_address)
    if not args.task.strip():
        raise ValueError("--task must be nonempty and match training-style task wording")
    if not all((args.can_port, args.top_camera_serial, args.wrist_camera_serial)):
        raise ValueError("CAN interface and both camera serials must be provided explicitly")
    if args.fps <= 0 or args.camera_fps <= 0 or args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("fps and camera dimensions must be positive")
    if args.camera_fps < args.fps:
        raise ValueError("camera FPS must be at least the control FPS")
    expected_shape = (3, args.camera_height, args.camera_width)
    if expected_shape != contract.top_image_shape or expected_shape != contract.wrist_image_shape:
        raise ValueError("camera width/height must match both checkpoint image feature shapes")
    if not all(math.isfinite(value) and value > 0 for value in
               (args.max_joint_step, args.max_gripper_step, args.connect_timeout)):
        raise ValueError("preview thresholds and timeout must be positive and finite")
    if args.print_every < 1:
        raise ValueError("print_every must be >= 1")
    _prepare_runtime(checkpoint)
    import grpc
    from lerobot.async_inference import robot_client as robot_client_module
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.robots.piper.config_piper import PIPERConfig
    from lerobot.robots.piper.piper import Piper

    _validate_device(args.device)
    channel = grpc.insecure_channel(args.server_address)
    try:
        grpc.channel_ready_future(channel).result(timeout=args.connect_timeout)
    except grpc.FutureTimeoutError as exc:
        raise ConnectionError(f"PI0 server unavailable at {args.server_address}") from exc
    finally:
        channel.close()

    top_key = contract.top_image_key.removeprefix("observation.images.")
    wrist_key = contract.wrist_image_key.removeprefix("observation.images.")
    cameras = {
        top_key: RealSenseCameraConfig(serial_number_or_name=args.top_camera_serial,
                                      fps=args.camera_fps, width=args.camera_width, height=args.camera_height),
        wrist_key: RealSenseCameraConfig(serial_number_or_name=args.wrist_camera_serial,
                                        fps=args.camera_fps, width=args.camera_width, height=args.camera_height),
    }
    piper_config = PIPERConfig(
        can_port=args.can_port, cameras=cameras,
        home_position=[0.0] * 7, reset_hz=100.0, reset_duration_s=4.0,
        max_joint_step_rad=0.01, open_gripper_on_init=True, gripper_open_range=0.07,
    )
    preview_type = _preview_piper_class(Piper)
    robot = preview_type(piper_config, contract, preview_log,
                         args.max_joint_step, args.max_gripper_step, args.print_every)
    client_config = RobotClientConfig(
        policy_type="pi0", pretrained_name_or_path=str(checkpoint), robot=piper_config,
        actions_per_chunk=contract.chunk_size, task=args.task,
        server_address=args.server_address, policy_device=args.device,
        client_device="cpu", chunk_size_threshold=0.5, fps=args.fps,
        aggregate_fn_name="weighted_average", debug_visualize_queue_size=False,
    )
    client_type = _preview_client_class(RobotClient, robot_client_module)
    client = client_type(client_config, robot)
    receiver = None
    try:
        if not client.start():
            raise ConnectionError("PI0 client handshake failed")
        print("PREVIEW ONLY: cameras/CAN receive active; motors not enabled; no commands sent", flush=True)
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        client.control_loop(task=args.task)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        if receiver is not None:
            receiver.join(timeout=3.0)
    return {"status": "stopped", "preview_log": str(preview_log),
            "motion_authorized": False}


def run_execute_client(args) -> dict:
    checkpoint, contract = _checkpoint(args)
    execute_log = args.execute_log.expanduser().resolve()
    _local_server_address(args.server_address)
    if not args.task.strip():
        raise ValueError("--task must be nonempty and match training-style task wording")
    if not all((args.can_port, args.top_camera_serial, args.wrist_camera_serial)):
        raise ValueError("CAN interface and both camera serials must be provided explicitly")
    if args.fps <= 0 or args.camera_fps <= 0 or args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("fps and camera dimensions must be positive")
    if args.camera_fps < args.fps:
        raise ValueError("camera FPS must be at least the control FPS")
    expected_shape = (3, args.camera_height, args.camera_width)
    if expected_shape != contract.top_image_shape or expected_shape != contract.wrist_image_shape:
        raise ValueError("camera width/height must match both checkpoint image feature shapes")
    if not all(math.isfinite(value) and value > 0 for value in (
            args.max_joint_step, args.max_gripper_step, args.connect_timeout,
            args.reset_hz, args.reset_duration, args.reset_max_joint_step)):
        raise ValueError("execute thresholds, reset settings and timeout must be positive and finite")
    if not 0.0 <= args.home_gripper <= 0.08:
        raise ValueError("home gripper must be between 0.0 and 0.08 metres")
    if args.max_actions < 1 or args.max_actions > contract.chunk_size:
        raise ValueError(f"max actions must be between 1 and checkpoint chunk size {contract.chunk_size}")
    if args.print_every < 1:
        raise ValueError("print_every must be >= 1")

    _prepare_runtime(checkpoint)
    import grpc
    from lerobot.async_inference import robot_client as robot_client_module
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.robots.piper.config_piper import PIPERConfig
    from lerobot.robots.piper.piper import Piper

    _validate_device(args.device)
    channel = grpc.insecure_channel(args.server_address)
    try:
        grpc.channel_ready_future(channel).result(timeout=args.connect_timeout)
    except grpc.FutureTimeoutError as exc:
        raise ConnectionError(f"PI0 server unavailable at {args.server_address}") from exc
    finally:
        channel.close()

    top_key = contract.top_image_key.removeprefix("observation.images.")
    wrist_key = contract.wrist_image_key.removeprefix("observation.images.")
    cameras = {
        top_key: RealSenseCameraConfig(serial_number_or_name=args.top_camera_serial,
                                      fps=args.camera_fps, width=args.camera_width,
                                      height=args.camera_height),
        wrist_key: RealSenseCameraConfig(serial_number_or_name=args.wrist_camera_serial,
                                        fps=args.camera_fps, width=args.camera_width,
                                        height=args.camera_height),
    }
    piper_config = PIPERConfig(
        can_port=args.can_port, cameras=cameras, home_position=[0.0] * 7,
        reset_hz=args.reset_hz, reset_duration_s=args.reset_duration,
        max_joint_step_rad=args.reset_max_joint_step,
        open_gripper_on_init=True, gripper_open_range=args.home_gripper,
    )
    execute_type = _execute_piper_class(Piper)
    robot = execute_type(
        piper_config, contract, execute_log, args.max_joint_step,
        args.max_gripper_step, args.max_actions, args.print_every,
    )
    client_config = RobotClientConfig(
        policy_type="pi0", pretrained_name_or_path=str(checkpoint), robot=piper_config,
        actions_per_chunk=contract.chunk_size, task=args.task,
        server_address=args.server_address, policy_device=args.device,
        client_device="cpu", chunk_size_threshold=0.5, fps=args.fps,
        aggregate_fn_name="weighted_average", debug_visualize_queue_size=False,
    )
    client_type = _preview_client_class(RobotClient, robot_client_module)
    client = client_type(client_config, robot)
    receiver = None
    try:
        if not client.start():
            raise ConnectionError("PI0 client handshake failed")
        print(
            f"EXECUTE: Piper enabled for at most {args.max_actions} bounded commands; "
            "use the physical emergency stop for unsafe motion",
            flush=True,
        )
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        try:
            client.control_loop(task=args.task)
        except ActionBudgetReached as exc:
            print(str(exc), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        if receiver is not None:
            receiver.join(timeout=3.0)
    return {
        "status": "action_budget_reached" if robot._action_count >= args.max_actions else "stopped",
        "actions_sent": robot._action_count,
        "action_budget": args.max_actions,
        "execute_log": str(execute_log),
        "motion_authorized": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HarnessVLA PI0 deployment")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="validate a new checkpoint in weights/pi0")
    check.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    check.add_argument("--hash-model", action="store_true")
    check.add_argument("--processors", action="store_true", help="run local LeRobot processor check")
    check.add_argument("--load-model", action="store_true", help="strictly load model weights")
    check.add_argument("--smoke-inference", action="store_true", help="run one synthetic inference chunk")
    check.add_argument("--device", default="cpu")
    check.add_argument("--task", default="check local PI0 deployment")
    check.set_defaults(handler=run_check)

    server = commands.add_parser("server", help="serve a pinned checkpoint on loopback only")
    server.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    server.add_argument("--device", default="cuda:0")
    server.add_argument("--host", default=LOOPBACK_HOST)
    server.add_argument("--port", type=int, default=DEFAULT_PORT)
    server.add_argument("--fps", type=int, default=20)
    server.add_argument("--obs-queue-timeout", type=float, default=2.0)
    server.add_argument("--expected-model-sha256", help="optional pin for the new model file")
    server.set_defaults(handler=run_server, hash_model=True)

    preview = commands.add_parser("preview-client", help="read Piper/cameras; log actions without motor commands")
    preview.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    preview.add_argument("--task", required=True)
    preview.add_argument("--server-address", default=f"{LOOPBACK_HOST}:{DEFAULT_PORT}")
    preview.add_argument("--device", default="cuda:0")
    preview.add_argument("--can-port", required=True)
    preview.add_argument("--top-camera-serial", required=True)
    preview.add_argument("--wrist-camera-serial", required=True)
    preview.add_argument("--camera-width", type=int, default=640)
    preview.add_argument("--camera-height", type=int, default=480)
    preview.add_argument("--camera-fps", type=int, default=60)
    preview.add_argument("--fps", type=int, default=20)
    preview.add_argument("--connect-timeout", type=float, default=5.0)
    preview.add_argument("--max-joint-step", type=float, default=0.05)
    preview.add_argument("--max-gripper-step", type=float, default=0.005)
    preview.add_argument("--print-every", type=int, default=10)
    preview.add_argument("--preview-log", type=Path, default=Path("runs/pi0_preview.jsonl"))
    preview.set_defaults(handler=run_preview_client, hash_model=False)

    execute = commands.add_parser(
        "execute-client", help="enable Piper and run one bounded PI0 skill"
    )
    execute.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    execute.add_argument("--task", required=True)
    execute.add_argument("--server-address", default=f"{LOOPBACK_HOST}:{DEFAULT_PORT}")
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument("--can-port", required=True)
    execute.add_argument("--top-camera-serial", required=True)
    execute.add_argument("--wrist-camera-serial", required=True)
    execute.add_argument("--camera-width", type=int, default=640)
    execute.add_argument("--camera-height", type=int, default=480)
    execute.add_argument("--camera-fps", type=int, default=60)
    execute.add_argument("--fps", type=int, default=20)
    execute.add_argument("--connect-timeout", type=float, default=5.0)
    execute.add_argument("--max-joint-step", type=float, default=0.02)
    execute.add_argument("--max-gripper-step", type=float, default=0.002)
    execute.add_argument("--max-actions", type=int, default=20)
    execute.add_argument("--reset-hz", type=float, default=100.0)
    execute.add_argument("--reset-duration", type=float, default=4.0)
    execute.add_argument("--reset-max-joint-step", type=float, default=0.01)
    execute.add_argument("--home-gripper", type=float, default=0.07)
    execute.add_argument("--print-every", type=int, default=10)
    execute.add_argument("--execute-log", type=Path, default=Path("runs/pi0_execute.jsonl"))
    execute.set_defaults(handler=run_execute_client, hash_model=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ContractError, ValueError, RuntimeError, ImportError, OSError, ConnectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
