import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harnessvla.policy_skills import (SkillRegistryError, load_action_skills,
                                      validate_t1_sequence)
from harnessvla.qwen_planner import (PlanningError, QwenConfig, QwenPlanner,
                                     _http_transport, main)

FULL_T1 = [
    "pick_red_block", "place_red_in_teal_box", "pick_green_block",
    "place_green_in_bowl", "press_red_button",
]


def config(api_key="test-key"):
    return QwenConfig(
        api_key=api_key, model="qwen3.8-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_s=5, enable_thinking=True, action_budget=20, subtask_timeout_s=60,
    )


def response(skill_ids):
    return {"id": "fixture-response", "choices": [{"message": {
        "content": json.dumps({"skill_ids": skill_ids})}}],
        "usage": {"total_tokens": 42}}


class SkillRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_action_skills()

    def test_user_corrected_task_names_registered_for_constrained_execution(self):
        self.assertTrue(self.registry.execution_enabled)
        self.assertEqual([skill.policy_task for skill in self.registry.skills], [
            "pick up the red block",
            "place the held object in the blue box",
            "pick up the green block",
            "place the held object in the bowl",
            "press the red button",
            "return to home position",
        ])
        self.assertFalse(self.registry.by_id()["return_home"].t1_allowed)

    def test_valid_t1_plan_preserves_policy_prompts(self):
        plan = validate_t1_sequence(FULL_T1, set(), self.registry, 20, 60)
        self.assertEqual(len(plan), 5)
        self.assertEqual(plan[3].instruction, "place the held object in the bowl")
        self.assertEqual(plan[-1].success_predicate, "red_button_triggered")

    def test_t2_suffix_skips_verified_red_goal(self):
        plan = validate_t1_sequence(
            ["pick_green_block", "place_green_in_bowl", "press_red_button"],
            {"red_in_blue_box"}, self.registry, 20, 60,
        )
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[0].skill, "pick_green_block")

    def test_reject_unknown_reordered_incomplete_and_home(self):
        cases = [
            ["not_registered"],
            ["place_red_in_teal_box"] + FULL_T1[2:],
            FULL_T1[:-1],
            FULL_T1 + ["return_home"],
            ["pick_red_block", "pick_red_block"] + FULL_T1[1:],
        ]
        for skill_ids in cases:
            with self.subTest(skill_ids=skill_ids), self.assertRaises(SkillRegistryError):
                validate_t1_sequence(skill_ids, set(), self.registry, 20, 60)


class QwenPlannerTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_action_skills()

    def test_valid_fake_qwen_response_is_constrained(self):
        seen = {}

        def fake_transport(_config, payload):
            seen.update(payload)
            return response(FULL_T1)

        result = QwenPlanner(config(), self.registry, fake_transport).plan("T1", set(), "blue box visible")
        self.assertEqual(len(result.subtasks), 5)
        self.assertFalse(result.motion_authorized)
        self.assertEqual(result.response_id, "fixture-response")
        self.assertEqual(result.usage["total_tokens"], 42)
        self.assertEqual(seen["model"], "qwen3.8-flash")
        self.assertTrue(seen["enable_thinking"])
        self.assertTrue(seen["stream"])
        self.assertEqual(seen["stream_options"], {"include_usage": True})
        self.assertNotIn("response_format", seen)
        self.assertNotIn("max_tokens", seen)
        self.assertNotIn("return_home", seen["messages"][1]["content"])

    def test_non_thinking_mode_requests_json_object(self):
        raw = dict(config().__dict__)
        raw["enable_thinking"] = False
        seen = {}

        def fake_transport(_config, payload):
            seen.update(payload)
            return response(FULL_T1)

        QwenPlanner(QwenConfig(**raw), self.registry, fake_transport).plan("T1", set())
        self.assertEqual(seen["response_format"], {"type": "json_object"})

    def test_qwen_cannot_invent_or_omit_skills(self):
        for ids in (["unknown_skill"], FULL_T1[:-1], ["press_red_button"]):
            with self.subTest(ids=ids), self.assertRaises(PlanningError):
                QwenPlanner(config(), self.registry, lambda _config, _payload: response(ids)).plan("T1", set())

    def test_missing_key_stops_before_transport(self):
        called = []

        def transport(_config, _payload):
            called.append(True)
            return response(FULL_T1)

        with self.assertRaisesRegex(PlanningError, "API key is empty"):
            QwenPlanner(config(""), self.registry, transport).plan("T1", set())
        self.assertEqual(called, [])

    def test_non_json_qwen_output_is_rejected(self):
        malformed = {"choices": [{"message": {"content": "```json\n{}\n```"}}]}
        with self.assertRaisesRegex(PlanningError, "strict JSON"):
            QwenPlanner(config(), self.registry, lambda _config, _payload: malformed).plan("T1", set())

    def test_config_rejects_non_alibaba_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen.json"
            raw = dict(config().__dict__)
            raw["base_url"] = "https://example.com/compatible-mode/v1"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(PlanningError, "Alibaba Cloud"):
                QwenConfig.load(path)
            raw["base_url"] = None
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(PlanningError, "base_url must be a string"):
                QwenConfig.load(path)

    def test_default_qwen_config_reads_yaml_section(self):
        path = Path(__file__).resolve().parent.parent / "config" / "piper_harness.example.yaml"
        loaded = QwenConfig.load(path)
        self.assertEqual(loaded.model, "qwen3.8-flash")
        self.assertTrue(loaded.enable_thinking)
        self.assertEqual(loaded.api_key, "")

    def test_http_request_uses_compatible_endpoint_without_logging_key(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                answer = json.dumps({"skill_ids": FULL_T1})
                events = [
                    {"id": "fixture-response", "choices": [{"delta": {"reasoning_content": "ignore this"},
                                                             "finish_reason": None}]},
                    {"choices": [{"delta": {"content": answer[:20]}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": answer[20:]}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"total_tokens": 42}},
                ]
                for event in events:
                    yield ("data: " + json.dumps(event) + "\n").encode()
                    yield b"\n"
                yield b"data: [DONE]\n"

        captured = {}

        def fake_open(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("harnessvla.qwen_planner.urlopen", fake_open):
            result = _http_transport(config(), {"model": "qwen3.8-flash", "messages": []})
        self.assertEqual(result["id"], "fixture-response")
        self.assertEqual(result["usage"], {"total_tokens": 42})
        self.assertEqual(json.loads(result["choices"][0]["message"]["content"]), {"skill_ids": FULL_T1})
        self.assertEqual(captured["request"].full_url,
                         "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(captured["request"].get_header("Authorization"), "Bearer test-key")
        self.assertEqual(captured["request"].get_header("Accept"), "text/event-stream")
        self.assertEqual(captured["timeout"], 5)

    def test_incomplete_or_truncated_stream_is_rejected(self):
        class FakeResponse:
            def __init__(self, lines):
                self.lines = lines

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter(self.lines)

        incomplete = [b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n']
        truncated = [b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"length"}]}\n',
                     b'data: [DONE]\n']
        for lines in (incomplete, truncated):
            with self.subTest(lines=lines), patch("harnessvla.qwen_planner.urlopen", return_value=FakeResponse(lines)):
                with self.assertRaises(PlanningError):
                    _http_transport(config(), {"model": "qwen3.8-flash", "messages": []})

    def test_offline_validate_needs_no_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps({"skill_ids": FULL_T1}))
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                code = main(["validate", "--file", str(path)])
            self.assertEqual(code, 0)
            self.assertFalse(json.loads(output.getvalue())["motion_authorized"])


if __name__ == "__main__":
    unittest.main()
