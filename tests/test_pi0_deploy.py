import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harnessvla.pi0_deploy import (ActionBudgetReached, _execute_piper_class,
                                   _local_server_address, _preview_piper_class,
                                   build_parser, main, run_server)


class PI0DeployTests(unittest.TestCase):
    def test_default_checkpoint_is_reserved_project_directory(self):
        args = build_parser().parse_args(["check"])
        self.assertTrue(str(args.checkpoint).endswith("/HarnessVLA/weights/pi0"))

    def test_no_weights_fails_before_importing_lerobot(self):
        # Do not assume the user-managed default checkpoint directory is empty.
        # An isolated empty directory keeps this failure-path test deterministic
        # after real deployment weights have been installed.
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            code = main(["check", "--checkpoint", directory])
        self.assertEqual(code, 2)
        self.assertIn("missing files", errors.getvalue())

    def test_server_rejects_non_loopback_before_checkpoint_load(self):
        args = build_parser().parse_args(["server", "--host", "0.0.0.0"])
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            run_server(args)

    def test_preview_server_address_must_be_local(self):
        self.assertEqual(_local_server_address("127.0.0.1:8081"), "127.0.0.1:8081")
        for invalid in ("0.0.0.0:8081", "example.com:8081", "127.0.0.1:0", "127.0.0.1:bad"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _local_server_address(invalid)

    def test_preview_requires_explicit_hardware_identifiers(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["preview-client", "--task", "example"])
        args = build_parser().parse_args([
            "preview-client", "--task", "example", "--can-port", "can_test",
            "--top-camera-serial", "top", "--wrist-camera-serial", "wrist",
        ])
        self.assertFalse(hasattr(args, "execute"))

    def test_preview_override_never_calls_parent_motor_methods(self):
        class FakePiper:
            def __init__(self, config):
                state = {f"joint_{index}": 0.0 for index in range(1, 7)}
                state["joint_7"] = 0.02
                self.bus = SimpleNamespace(read=lambda: state, piper=SimpleNamespace())
                self.cameras = {}

            def send_action(self, action):
                raise AssertionError("parent motor method must not be called")

            def disconnect(self):
                raise AssertionError("parent disconnect may home the arm")

        contract = SimpleNamespace(
            action_names=tuple(f"joint_{index}.pos" for index in range(1, 8)),
            training_quantile_low=(-1.0,) * 6 + (0.0,),
            training_quantile_high=(1.0,) * 6 + (0.08,),
            chunk_size=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "preview.jsonl"
            robot = _preview_piper_class(FakePiper)(None, contract, log, 0.05, 0.005, 10)
            robot.connect()
            action = {f"joint_{index}.pos": 0.01 for index in range(1, 7)}
            action["joint_7.pos"] = 0.021
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(robot.send_action(action), action)
            robot.disconnect()
            record = json.loads(log.read_text().strip())
            self.assertFalse(record["motion_authorized"])
            self.assertEqual(record["status"], "preview_only_within_reference_envelope")

    def test_execute_override_clamps_calls_parent_and_stops_at_budget(self):
        class FakePiper:
            def __init__(self, config):
                state = {f"joint_{index}": 0.0 for index in range(1, 7)}
                state["joint_7"] = 0.02
                self.bus = SimpleNamespace(read=lambda: state)
                self.delivered = []

            def send_action(self, action):
                self.delivered.append(action)
                return action

        contract = SimpleNamespace(
            action_names=tuple(f"joint_{index}.pos" for index in range(1, 8)),
            training_quantile_low=(-1.0,) * 6 + (0.0,),
            training_quantile_high=(1.0,) * 6 + (0.08,),
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "execute.jsonl"
            robot = _execute_piper_class(FakePiper)(
                None, contract, log, 0.02, 0.002, 1, 10,
            )
            action = {f"joint_{index}.pos": 0.5 for index in range(1, 7)}
            action["joint_7.pos"] = 0.08
            with contextlib.redirect_stdout(io.StringIO()):
                delivered = robot.send_action(action)
            self.assertEqual(len(robot.delivered), 1)
            self.assertAlmostEqual(delivered["joint_1.pos"], 0.02)
            self.assertAlmostEqual(delivered["joint_7.pos"], 0.022)
            with self.assertRaises(ActionBudgetReached):
                robot.send_action(action)
            record = json.loads(log.read_text().strip())
            self.assertTrue(record["motion_authorized"])
            self.assertTrue(record["clamped"])
            self.assertEqual(record["action_budget"], 1)


if __name__ == "__main__":
    unittest.main()
