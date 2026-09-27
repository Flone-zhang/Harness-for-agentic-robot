import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harnessvla.cli import main
from harnessvla.config import Config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class YAMLConfigTests(unittest.TestCase):
    def test_shipped_yaml_matches_mock_json(self):
        yaml_config = Config.load(PROJECT_ROOT / "config" / "pycharm.yaml")
        json_config = Config.load(PROJECT_ROOT / "config" / "mock.json")
        self.assertEqual(yaml_config, json_config)

    def test_pycharm_style_start_uses_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "run.sqlite3"
            config_path = PROJECT_ROOT / "config" / "pycharm.yaml"
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                code = main(["--db", str(database), "start", "--config", str(config_path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "complete")
            self.assertTrue(database.exists())

    def test_yaml_cannot_enable_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.yaml"
            path.write_text("mode: piper\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only mock mode"):
                Config.load(path)

    def test_yaml_uses_safe_loader_and_requires_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid YAML"):
                Config.load(path)
            path.write_text("- mode: mock\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping"):
                Config.load(path)

    def test_yaml_rejects_invalid_budget_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text('mode: mock\naction_budget: "8"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "action_budget"):
                Config.load(path)


if __name__ == "__main__":
    unittest.main()
