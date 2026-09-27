import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from harnessvla.live_t1 import load_live_settings
from harnessvla.run_config import (
    DEFAULT_RUN_CONFIG,
    load_run_config,
    main,
    run_from_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RunConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_run_config(PROJECT_ROOT / "config" / "piper_harness.example.yaml")

    def test_only_runtime_sections_and_selected_can_remain(self):
        self.assertEqual(set(self.config), {"schema_version", "run", "pi0", "qwen"})
        self.assertEqual(self.config["pi0"]["can_port"], "can_left_f")
        self.assertEqual(
            self.config["pi0"]["cameras"]["cam_top"]["serial_number_or_name"],
            "REPLACE_WITH_TOP_CAMERA_SERIAL",
        )
        self.assertEqual(
            self.config["pi0"]["cameras"]["cam_left"]["serial_number_or_name"],
            "REPLACE_WITH_WRIST_CAMERA_SERIAL",
        )

    def test_direct_python_script_runs_selected_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config["run"]["db"] = str(Path(directory) / "harness.sqlite3")
            selected_yaml = Path(directory) / "selected.yaml"
            selected_yaml.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "run_harness.py"),
                 f"--config_path={selected_yaml}"],
                cwd=directory, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"status": "complete"', result.stdout)
            self.assertTrue((Path(directory) / "harness.sqlite3").exists())

    def test_preview_routes_only_to_command_free_client(self):
        config = copy.deepcopy(self.config)
        config["run"]["mode"] = "pi0_preview"
        with patch("harnessvla.run_config.pi0_deploy_main", return_value=0) as deploy:
            self.assertEqual(run_from_config(config), 0)
        argv = deploy.call_args.args[0]
        self.assertEqual(argv[0], "preview-client")
        self.assertEqual(argv[argv.index("--can-port") + 1], "can_left_f")
        self.assertEqual(argv[argv.index("--camera-fps") + 1], "30")
        self.assertNotIn("--execute", argv)
        self.assertNotIn("can_right_f", argv)

    def test_execute_mode_requires_explicit_second_gate(self):
        config = copy.deepcopy(self.config)
        config["run"]["mode"] = "pi0_execute"
        with patch("harnessvla.run_config.pi0_deploy_main") as deploy:
            with self.assertRaisesRegex(ValueError, "motion_enabled"):
                run_from_config(config)
        deploy.assert_not_called()

    def test_execute_mode_routes_to_bounded_live_client(self):
        config = copy.deepcopy(self.config)
        config["run"]["mode"] = "pi0_execute"
        config["pi0"]["motion_enabled"] = True
        with patch("harnessvla.run_config.pi0_deploy_main", return_value=0) as deploy:
            self.assertEqual(run_from_config(config), 0)
        argv = deploy.call_args.args[0]
        self.assertEqual(argv[0], "execute-client")
        self.assertEqual(argv[argv.index("--max-actions") + 1], "20")
        self.assertEqual(argv[argv.index("--can-port") + 1], "can_left_f")

    def test_qwen_mode_uses_selected_yaml(self):
        config = copy.deepcopy(self.config)
        config["run"]["mode"] = "qwen_plan"
        config["qwen"]["api_key"] = "test-key"
        config["qwen"]["verified_predicates"] = ["red_in_blue_box"]
        with patch("harnessvla.run_config.qwen_planner_main", return_value=0) as planner:
            self.assertEqual(run_from_config(config), 0)
        argv = planner.call_args.args[0]
        self.assertEqual(argv[:1], ["plan"])
        self.assertIn("red_in_blue_box", argv)
        self.assertIn(str(DEFAULT_RUN_CONFIG), argv)

    def test_closed_loop_mode_routes_to_live_t1(self):
        config = copy.deepcopy(self.config)
        config["run"]["mode"] = "qwen_pi0"
        config["qwen"]["api_key"] = "test-key"
        result = {"run_id": "fixture", "status": "complete"}
        with patch("harnessvla.live_t1.run_live_t1", return_value=result) as live, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(run_from_config(config), 0)
        live.assert_called_once_with(config, DEFAULT_RUN_CONFIG)
        self.assertEqual(json.loads(output.getvalue()), result)

    def test_live_mode_requires_and_loads_yaml_total_task_command(self):
        config = copy.deepcopy(self.config)
        config["run"]["mode"] = "qwen_pi0"
        config["pi0"]["motion_enabled"] = True
        loaded = load_live_settings(config, DEFAULT_RUN_CONFIG)
        self.assertEqual(loaded.task_command, config["run"]["task_command"])
        del config["run"]["task_command"]
        with self.assertRaisesRegex(ValueError, "run.task_command"):
            load_live_settings(config, DEFAULT_RUN_CONFIG)

    def test_selected_yaml_supplies_qwen_key_without_printing_it(self):
        config = copy.deepcopy(self.config)
        config["run"]["mode"] = "qwen_plan"
        config["qwen"]["api_key"] = "test-secret-from-yaml"
        skill_ids = ["pick_red_block", "place_red_in_teal_box", "pick_green_block",
                     "place_green_in_bowl", "press_red_button"]
        response = {"choices": [{"message": {"content": json.dumps({"skill_ids": skill_ids})}}]}
        with tempfile.TemporaryDirectory() as directory:
            selected_yaml = Path(directory) / "selected.yaml"
            selected_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
            with (patch("harnessvla.qwen_planner._http_transport", return_value=response) as transport,
                  patch("sys.stdout", new_callable=io.StringIO) as output):
                self.assertEqual(main([f"--config_path={selected_yaml}"]), 0)
            self.assertEqual(transport.call_args.args[0].api_key, "test-secret-from-yaml")
            self.assertNotIn("test-secret-from-yaml", output.getvalue())
            self.assertFalse(json.loads(output.getvalue())["motion_authorized"])

    def test_empty_qwen_key_stops_before_planner(self):
        config = copy.deepcopy(self.config)
        config["run"]["mode"] = "qwen_plan"
        with patch("harnessvla.run_config.qwen_planner_main") as planner:
            with self.assertRaisesRegex(ValueError, "qwen.api_key"):
                run_from_config(config)
        planner.assert_not_called()

    def test_unsafe_yaml_tag_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            selected_yaml = Path(directory) / "unsafe.yaml"
            selected_yaml.write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
            with patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(main([f"--config_path={selected_yaml}"]), 2)

    def test_unused_recording_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            selected_yaml = Path(directory) / "extra.yaml"
            config = copy.deepcopy(self.config)
            config["recording_reference"] = {"dataset": {"num_episodes": 500}}
            selected_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown run config sections"):
                load_run_config(selected_yaml)


if __name__ == "__main__":
    unittest.main()
