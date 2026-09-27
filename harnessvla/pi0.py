"""Read-only contract and recorded-action checks for the specified local PI0 export.

No LeRobot import, model loading, camera access, CAN access or action delivery occurs here.
"""

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ACTION_NAMES = tuple(f"joint_{index}.pos" for index in range(1, 8))
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "weights" / "pi0"
TOP_KEYS = ("observation.images.cam_top", "observation.images.base_0_rgb")
WRIST_KEYS = ("observation.images.cam_left", "observation.images.left_wrist_0_rgb")
OPTIONAL_IMAGES = {"observation.images.right_wrist_0_rgb"}
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_6_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


class ContractError(ValueError):
    pass


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path.name} must contain a JSON object")
    return value


def _processor_step(pipeline: dict, name: str) -> dict:
    matches = [step for step in pipeline.get("steps", []) if step.get("registry_name") == name]
    if len(matches) != 1:
        raise ContractError(f"expected exactly one {name} processor")
    return matches[0].get("config", {})


def _read_f32_vector(path: Path, tensor_name: str, length: int) -> tuple[float, ...]:
    """Read only a small named F32 safetensors statistic, never model weights."""
    try:
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ContractError(f"short safetensors header: {path.name}")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= 16 * 1024 * 1024:
                raise ContractError(f"invalid safetensors header size: {path.name}")
            header_raw = source.read(header_size)
            if len(header_raw) != header_size:
                raise ContractError(f"truncated safetensors header: {path.name}")
            header = json.loads(header_raw)
            tensor = header.get(tensor_name)
            if not isinstance(tensor, dict) or tensor.get("dtype") != "F32" or tensor.get("shape") != [length]:
                raise ContractError(f"missing F32[{length}] {tensor_name} in {path.name}")
            offsets = tensor.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(x, int) for x in offsets):
                raise ContractError(f"invalid offsets for {tensor_name}")
            start, end = offsets
            if start < 0 or end - start != length * 4 or end > path.stat().st_size - 8 - header_size:
                raise ContractError(f"invalid tensor span for {tensor_name}")
            source.seek(8 + header_size + start)
            raw = source.read(length * 4)
            if len(raw) != length * 4:
                raise ContractError(f"truncated tensor {tensor_name}")
            values = struct.unpack(f"<{length}f", raw)
    except (OSError, json.JSONDecodeError, struct.error) as exc:
        raise ContractError(f"cannot read {tensor_name}: {exc}") from exc
    if not all(math.isfinite(value) for value in values):
        raise ContractError(f"non-finite {tensor_name}")
    return values


@dataclass(frozen=True)
class PI0Contract:
    checkpoint_path: str
    top_image_key: str
    wrist_image_key: str
    top_image_shape: tuple[int, int, int]
    wrist_image_shape: tuple[int, int, int]
    state_key: str
    state_dimension: int
    action_dimension: int
    action_names: tuple[str, ...]
    action_representation: str
    joint_unit_reference: str
    gripper_unit_reference: str
    unit_verification: str
    chunk_size: int
    control_fps_reference: int
    training_quantile_low: tuple[float, ...]
    training_quantile_high: tuple[float, ...]
    model_size_bytes: int
    metadata_sha256: str
    model_sha256: str | None
    t1_skill_status: str

    def json(self) -> dict[str, Any]:
        return asdict(self)


def inspect_checkpoint(directory: str | Path = DEFAULT_CHECKPOINT_DIR, *, hash_model: bool = False) -> PI0Contract:
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise ContractError(f"checkpoint directory does not exist: {path}")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ContractError(f"checkpoint missing files: {', '.join(missing)}")
    config = _json_file(path / "config.json")
    preprocessor = _json_file(path / "policy_preprocessor.json")
    postprocessor = _json_file(path / "policy_postprocessor.json")
    if config.get("type") != "pi0":
        raise ContractError("checkpoint type is not pi0")
    if config.get("n_obs_steps") != 1:
        raise ContractError("deployment requires n_obs_steps=1")
    features = config.get("input_features", {})
    if features.get("observation.state", {}).get("shape") != [7] or features.get("observation.state", {}).get("type") != "STATE":
        raise ContractError("expected observation.state shape [7]")
    image_keys = {key for key, feature in features.items() if feature.get("type") == "VISUAL"}
    tops = [key for key in TOP_KEYS if key in image_keys]
    wrists = [key for key in WRIST_KEYS if key in image_keys]
    if len(tops) != 1 or len(wrists) != 1 or image_keys - {tops[0], wrists[0]} - OPTIONAL_IMAGES:
        raise ContractError(f"unsupported camera feature mapping: {sorted(image_keys)}")
    for key in (tops[0], wrists[0]):
        shape = features[key].get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 3 or any(
            type(dimension) is not int or dimension < 1 for dimension in shape
        ):
            raise ContractError(f"invalid RGB image shape for {key}")
    if config.get("output_features", {}).get("action", {}).get("shape") != [7]:
        raise ContractError("expected action shape [7]")
    if tuple(config.get("action_feature_names", ())) != ACTION_NAMES:
        raise ContractError("action ordering differs from joint_1.pos ... joint_7.pos")
    if config.get("use_relative_actions") is not False:
        raise ContractError("only absolute joint-position actions are supported")
    if _processor_step(preprocessor, "relative_actions_processor").get("enabled") is not False:
        raise ContractError("preprocessor unexpectedly enables relative actions")
    if _processor_step(postprocessor, "absolute_actions_processor").get("enabled") is not False:
        raise ContractError("postprocessor unexpectedly transforms action representation")
    _processor_step(postprocessor, "unnormalizer_processor")
    chunk_size = config.get("chunk_size")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ContractError("invalid chunk_size")
    action_steps = config.get("n_action_steps")
    if type(action_steps) is not int or not 1 <= action_steps <= chunk_size:
        raise ContractError("invalid n_action_steps")
    stats_path = path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    low = _read_f32_vector(stats_path, "action.q01", 7)
    high = _read_f32_vector(stats_path, "action.q99", 7)
    if any(left > right for left, right in zip(low, high)):
        raise ContractError("training action quantiles are reversed")
    digest = hashlib.sha256()
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        digest.update(name.encode())
        digest.update((path / name).read_bytes())
    model_hash = None
    if hash_model:
        model_digest = hashlib.sha256()
        with (path / "model.safetensors").open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                model_digest.update(block)
        model_hash = model_digest.hexdigest()
    return PI0Contract(
        checkpoint_path=str(path), top_image_key=tops[0], wrist_image_key=wrists[0],
        top_image_shape=tuple(features[tops[0]]["shape"]),
        wrist_image_shape=tuple(features[wrists[0]]["shape"]),
        state_key="observation.state", state_dimension=7, action_dimension=7,
        action_names=ACTION_NAMES, action_representation="absolute_joint_position_plus_gripper",
        joint_unit_reference="rad", gripper_unit_reference="m",
        unit_verification="reference_script_only_not_hardware_verified", chunk_size=chunk_size,
        control_fps_reference=20, training_quantile_low=low, training_quantile_high=high,
        model_size_bytes=(path / "model.safetensors").stat().st_size,
        metadata_sha256=digest.hexdigest(), model_sha256=model_hash,
        t1_skill_status="unverified_no_skill_evidence",
    )


def preview_chunk(
    contract: PI0Contract,
    current_state: list[float],
    chunk: list[list[float]],
    max_joint_step_rad: float = 0.05,
    max_gripper_step_m: float = 0.005,
) -> dict[str, Any]:
    """Reject structurally bad or out-of-reference actions; never authorize motion."""
    if not isinstance(current_state, list) or len(current_state) != 7 or not all(
        type(value) in (int, float) and math.isfinite(value) for value in current_state
    ):
        raise ContractError("current state must be seven finite numbers")
    if not isinstance(chunk, list) or not 1 <= len(chunk) <= contract.chunk_size:
        raise ContractError(f"chunk length must be 1..{contract.chunk_size}")
    if not math.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0:
        raise ContractError("max_joint_step_rad must be positive and finite")
    if not math.isfinite(max_gripper_step_m) or max_gripper_step_m <= 0:
        raise ContractError("max_gripper_step_m must be positive and finite")
    if not 0 <= current_state[6] <= 0.08:
        raise ContractError("current gripper state is outside 0..0.08 m")
    violations = []
    previous = current_state
    for index, action in enumerate(chunk):
        if not isinstance(action, list) or len(action) != 7 or not all(
            type(value) in (int, float) and math.isfinite(value) for value in action
        ):
            raise ContractError(f"action {index} must contain seven finite numbers")
        for joint, (value, low, high) in enumerate(zip(action, contract.training_quantile_low,
                                                       contract.training_quantile_high), start=1):
            if not low <= value <= high:
                violations.append(f"action[{index}] joint_{joint} outside checkpoint q01..q99")
        if not 0 <= action[6] <= 0.08:
            violations.append(f"action[{index}] gripper outside 0..0.08 m")
        if any(abs(action[joint] - previous[joint]) > max_joint_step_rad for joint in range(6)):
            violations.append(f"action[{index}] joint step exceeds preview threshold")
        if abs(action[6] - previous[6]) > max_gripper_step_m:
            violations.append(f"action[{index}] gripper step exceeds preview threshold")
        previous = action
    return {
        "status": "rejected" if violations else "preview_only_within_reference_envelope",
        "action_count": len(chunk), "violations": violations,
        "motion_authorized": False,
        "note": "Training quantiles and script step defaults are not calibrated site safety limits.",
    }
