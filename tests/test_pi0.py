import json
import io
import hashlib
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from harnessvla.cli import main
from harnessvla.pi0 import ContractError, inspect_checkpoint, preview_chunk
from harnessvla.pi0_deploy import build_parser as build_deploy_parser, run_preview_client, run_server


def write_stats(path: Path, low: list[float], high: list[float]) -> None:
    header = {
        "action.q01": {"dtype": "F32", "shape": [7], "data_offsets": [0, 28]},
        "action.q99": {"dtype": "F32", "shape": [7], "data_offsets": [28, 56]},
    }
    header_bytes = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes +
                     struct.pack("<7f", *low) + struct.pack("<7f", *high))


class PI0ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        (self.path / "tokenizer").mkdir()
        config = {
            "type": "pi0", "chunk_size": 2, "n_action_steps": 2,
            "n_obs_steps": 1, "use_relative_actions": False,
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [7]},
                "observation.images.cam_top": {"type": "VISUAL", "shape": [3, 480, 640]},
                "observation.images.cam_left": {"type": "VISUAL", "shape": [3, 480, 640]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [7]}},
            "action_feature_names": [f"joint_{index}.pos" for index in range(1, 8)],
        }
        (self.path / "config.json").write_text(json.dumps(config))
        (self.path / "policy_preprocessor.json").write_text(json.dumps({
            "steps": [{"registry_name": "relative_actions_processor", "config": {"enabled": False}}],
        }))
        (self.path / "policy_postprocessor.json").write_text(json.dumps({
            "steps": [
                {"registry_name": "unnormalizer_processor", "config": {}},
                {"registry_name": "absolute_actions_processor", "config": {"enabled": False}},
            ],
        }))
        (self.path / "model.safetensors").write_bytes(b"fixture")
        (self.path / "policy_preprocessor_step_6_normalizer_processor.safetensors").write_bytes(b"fixture")
        (self.path / "tokenizer/tokenizer.json").write_text("{}")
        (self.path / "tokenizer/tokenizer_config.json").write_text("{}")
        write_stats(self.path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                    [-1.0] * 6 + [0.0], [1.0] * 6 + [0.08])

    def tearDown(self):
        self.temp.cleanup()

    def test_inspect_contract_without_loading_model(self):
        contract = inspect_checkpoint(self.path)
        self.assertEqual(contract.action_dimension, 7)
        self.assertEqual(contract.chunk_size, 2)
        self.assertEqual(contract.top_image_key, "observation.images.cam_top")
        self.assertEqual(contract.top_image_shape, (3, 480, 640))
        self.assertEqual(contract.model_sha256, None)
        self.assertEqual(contract.t1_skill_status, "unverified_no_skill_evidence")

    def test_optional_full_model_hash(self):
        contract = inspect_checkpoint(self.path, hash_model=True)
        self.assertEqual(contract.model_sha256, hashlib.sha256(b"fixture").hexdigest())

    def test_server_rejects_wrong_pinned_hash_before_importing_lerobot(self):
        args = build_deploy_parser().parse_args([
            "server", "--checkpoint", str(self.path),
            "--expected-model-sha256", "0" * 64,
        ])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            run_server(args)

    def test_preview_rejects_camera_shape_before_hardware_import(self):
        args = build_deploy_parser().parse_args([
            "preview-client", "--checkpoint", str(self.path), "--task", "fixture task",
            "--can-port", "can_test", "--top-camera-serial", "top",
            "--wrist-camera-serial", "wrist", "--camera-width", "320",
        ])
        with self.assertRaisesRegex(ValueError, "image feature shapes"):
            run_preview_client(args)

    def test_reject_relative_or_wrong_action_contract(self):
        config_path = self.path / "config.json"
        config = json.loads(config_path.read_text())
        config["use_relative_actions"] = True
        config_path.write_text(json.dumps(config))
        with self.assertRaises(ContractError):
            inspect_checkpoint(self.path)
        config["use_relative_actions"] = False
        config["output_features"]["action"]["shape"] = [8]
        config_path.write_text(json.dumps(config))
        with self.assertRaises(ContractError):
            inspect_checkpoint(self.path)

    def test_reject_incomplete_checkpoint(self):
        (self.path / "model.safetensors").unlink()
        with self.assertRaisesRegex(ContractError, "missing files"):
            inspect_checkpoint(self.path)

    def test_preview_is_never_motion_authority(self):
        contract = inspect_checkpoint(self.path)
        state = [0.0] * 6 + [0.02]
        good = [[0.01] * 6 + [0.021], [0.02] * 6 + [0.022]]
        report = preview_chunk(contract, state, good)
        self.assertEqual(report["status"], "preview_only_within_reference_envelope")
        self.assertFalse(report["motion_authorized"])
        bad = [[1.1] + [0.0] * 5 + [0.1]]
        report = preview_chunk(contract, state, bad)
        self.assertEqual(report["status"], "rejected")
        self.assertTrue(any("q01..q99" in item for item in report["violations"]))
        self.assertTrue(any("gripper" in item for item in report["violations"]))

    def test_preview_rejects_malformed_nonfinite_and_large_chunks(self):
        contract = inspect_checkpoint(self.path)
        state = [0.0] * 7
        with self.assertRaises(ContractError):
            preview_chunk(contract, state, [[float("nan")] + [0.0] * 6])
        with self.assertRaises(ContractError):
            preview_chunk(contract, state, [[0.0] * 7] * 3)
        with self.assertRaises(ContractError):
            preview_chunk(contract, state, [[0.0] * 6])

    def test_checkpoint_cli_does_not_create_run_database(self):
        db = self.path / "should_not_exist.sqlite3"
        with redirect_stdout(io.StringIO()) as output:
            exit_code = main(["--db", str(db), "pi0-checkpoint", "--checkpoint", str(self.path)])
        self.assertEqual(exit_code, 0)
        self.assertFalse(db.exists())
        self.assertEqual(json.loads(output.getvalue())["action_dimension"], 7)


if __name__ == "__main__":
    unittest.main()
