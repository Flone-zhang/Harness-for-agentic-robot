import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from harnessvla.live_t1 import (
    EvidencePair,
    ExecutionReport,
    LiveSafetyFault,
    LiveSettings,
    LiveT1Error,
    LiveT1Orchestrator,
    PlannerDecision,
    QwenVisionAgent,
    VerificationDecision,
    _interval_is_stationary,
)
from harnessvla.policy_skills import T1_GOALS, load_action_skills
from harnessvla.qwen_planner import PlanningError, QwenConfig
from harnessvla.store import Store

TASK_COMMAND = (
    "put the red block in the blue box, and put the green block in the bowl, "
    "then press the red button"
)


def qwen_config():
    return QwenConfig(
        api_key="test-key", model="qwen3.8-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_s=5, enable_thinking=True, action_budget=20,
        subtask_timeout_s=60,
    )


def evidence():
    return EvidencePair(
        "top.jpg", "wrist.jpg", "a" * 64, "b" * 64,
        "data:image/jpeg;base64,dG9w", "data:image/jpeg;base64,d3Jpc3Q=",
    )


def settings(root: Path):
    return LiveSettings(
        project_root=root, source_path=root / "config.yaml", task_id="main_task",
        task_command=TASK_COMMAND,
        db_path=root / "events.sqlite3",
        checkpoint=root / "weights", checkpoint_hash="a" * 64, device="cuda:0",
        server_address="127.0.0.1:8081", can_port="can_left_f", fps=20,
        camera_width=640, camera_height=480, camera_fps=30,
        top_camera_serial="top", wrist_camera_serial="wrist",
        max_joint_step=0.02, max_gripper_step=0.002,
        console_status=False,
        verification_interval_actions=50,
        wait_pose_min_actions=200, wait_pose_stable_intervals=2,
        wait_pose_joint_range=0.015, wait_pose_gripper_range=0.001,
        verification_retries=1, jpeg_quality=85, execution_timeout_s=60,
        reset_hz=100, reset_duration_s=4, reset_max_joint_step=0.01,
        home_gripper=0.07,
    )


class FakeExecutor:
    def __init__(self, camera_fault=False):
        self.camera_fault = camera_fault
        self.started = False
        self.normal_stops = 0
        self.emergency_stops = 0
        self.executed = []
        self.captures = 0

    def start(self):
        self.started = True

    def capture_frames(self):
        self.captures += 1
        if self.camera_fault:
            raise RuntimeError("camera unplugged")
        return {"cam_top": b"top-jpeg", "cam_left": b"wrist-jpeg"}

    def execute(self, skill, action_budget, timeout_s):
        self.executed.append(skill.id)
        records = tuple({"requested_action": [0] * 7, "safe_action": [0] * 7,
                         "current_state": [0] * 7,
                         "action_number": index + 1} for index in range(action_budget))
        return ExecutionReport(action_budget, records, 1.0)

    def normal_stop(self):
        self.normal_stops += 1

    def emergency_stop(self):
        self.emergency_stops += 1


class SuccessfulAgent:
    order = ["pick_red_block", "place_red_in_teal_box", "pick_green_block",
             "place_green_in_bowl", "press_red_button"]

    def __init__(self, initial_verified=frozenset()):
        self.config = qwen_config()
        self.plan_calls = 0
        self.initial_verified = set(initial_verified)
        self.assessment_calls = 0

    def choose_next(self, _evidence, verified, _attempts, _history):
        self.plan_calls += 1
        if T1_GOALS <= verified:
            return PlannerDecision("finish", None, "all goals visible", "plan-finish", {}, 0.1)
        if "red_in_blue_box" not in verified:
            skill_id = ("place_red_in_teal_box" if "holding_red_block" in verified
                        else "pick_red_block")
        elif "green_in_bowl" not in verified:
            skill_id = ("place_green_in_bowl" if "holding_green_block" in verified
                        else "pick_green_block")
        else:
            skill_id = "press_red_button"
        return PlannerDecision("execute", skill_id, "next visible step", "plan", {}, 0.1)

    def verify(self, skill, _before, _after):
        return VerificationDecision(skill.success_predicate, "true", 0.99, "clearly visible",
                                    "verify", {}, 0.1)

    def assess_scene_goals(self, _evidence):
        self.assessment_calls += 1
        return {
            predicate: VerificationDecision(
                predicate,
                "true" if predicate in self.initial_verified else "false",
                0.99,
                "initial scene result",
                "initial-verify",
                {},
                0.1,
            )
            for predicate in T1_GOALS
        }


class DelayedSuccessfulAgent(SuccessfulAgent):
    def __init__(self):
        super().__init__()
        self.failed_pick_checks = 0

    def verify(self, skill, _before, _after):
        if skill.id == "pick_red_block" and self.failed_pick_checks < 2:
            self.failed_pick_checks += 1
            return VerificationDecision(skill.success_predicate, "false", 0.95, "not held yet",
                                        "verify", {}, 0.1)
        return super().verify(skill, _before, _after)


class TransientVerifierAgent(SuccessfulAgent):
    def __init__(self):
        super().__init__()
        self.verifier_calls = 0

    def verify(self, skill, before, after):
        self.verifier_calls += 1
        if self.verifier_calls == 1:
            try:
                raise URLError("temporary network failure")
            except URLError as exc:
                raise PlanningError("Qwen request failed") from exc
        return super().verify(skill, before, after)


class NeverCompleteAgent(SuccessfulAgent):
    def verify(self, skill, _before, _after):
        return VerificationDecision(skill.success_predicate, "false", 0.99,
                                    "skill is not complete", "verify", {}, 0.1)


class TimeoutExecutor(FakeExecutor):
    def execute(self, skill, _action_budget, _timeout_s):
        self.executed.append(skill.id)
        records = tuple({"requested_action": [0] * 7, "safe_action": [0] * 7,
                         "current_state": [0] * 7, "action_number": index + 1}
                        for index in range(7))
        return ExecutionReport(7, records, 60.0, timed_out=True)


class InterruptingExecutor(FakeExecutor):
    def execute(self, _skill, _action_budget, _timeout_s):
        raise KeyboardInterrupt


class LiveT1OrchestratorTests(unittest.TestCase):
    def test_scene_aware_mode_skips_red_branch_on_single_true(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = FakeExecutor()
            store = Store(root / "events.sqlite3")
            agent = SuccessfulAgent({"red_in_blue_box"})
            orchestrator = LiveT1Orchestrator(
                settings(root),
                load_action_skills(), agent, executor, store,
                run_id="scene_continue_run",
            )
            result = orchestrator.run()
            events = store.events("scene_continue_run")
            store.close()
        self.assertEqual(result["task_id"], "main_task")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(
            executor.executed,
            ["pick_green_block", "place_green_in_bowl", "press_red_button"],
        )
        initial_checks = [event for event in events
                          if event["kind"] == "qwen_initial_scene_verifier_result"]
        self.assertEqual(len(initial_checks), 3)
        self.assertEqual(agent.assessment_calls, 1)
        self.assertTrue(all(event["payload"]["phase"] == "single"
                            for event in initial_checks))
        self.assertTrue(all(event["payload"]["pi0_actions_sent"] == 0
                            for event in initial_checks))
        self.assertEqual(sum(event["kind"] == "initial_scene_assessed"
                             for event in events), 1)
        self.assertTrue(T1_GOALS <= set(result["verified_predicates"]))

    def test_console_status_reports_yaml_interval_skill_and_qwen_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = FakeExecutor()
            store = Store(root / "events.sqlite3")
            orchestrator = LiveT1Orchestrator(
                replace(
                    settings(root),
                    console_status=True,
                    verification_interval_actions=100,
                ),
                load_action_skills(), SuccessfulAgent(), executor, store,
                run_id="console_run",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                orchestrator.run()
            store.close()
        lines = [line for line in output.getvalue().splitlines()
                 if line.startswith("[HarnessVLA] ")]
        records = [json.loads(line.removeprefix("[HarnessVLA] ")) for line in lines]
        events = [record["event"] for record in records]
        self.assertIn("skill_started", events)
        self.assertIn("pi0_interval_completed", events)
        self.assertIn("qwen_verification", events)
        self.assertIn("skill_completed", events)
        self.assertIn("controlled_shutdown_completed", events)
        started = next(record for record in records if record["event"] == "run_started")
        progress = next(record for record in records
                        if record["event"] == "pi0_interval_completed")
        verdict = next(record for record in records
                       if record["event"] == "qwen_verification")
        self.assertEqual(started["verification_interval_actions"], 100)
        self.assertEqual(progress["configured_interval_actions"], 100)
        self.assertEqual(progress["skill_actions_total"], 100)
        self.assertEqual(verdict["value"], "true")
        self.assertNotIn("test-key", output.getvalue())

    def test_full_t1_requires_double_visual_confirmation_and_homes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = FakeExecutor()
            store = Store(root / "events.sqlite3")
            agent = SuccessfulAgent()
            orchestrator = LiveT1Orchestrator(
                settings(root), load_action_skills(), agent, executor, store,
                run_id="fixture_run",
            )
            result = orchestrator.run()
            events = store.events("fixture_run")
            store.close()
            execution_log = root / "runs" / "fixture_run" / "execution_events.jsonl"
            execution_log_text = execution_log.read_text()
            summary_exists = (root / "runs" / "fixture_run" / "run_summary.json").exists()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(T1_GOALS <= set(result["verified_predicates"]))
        self.assertEqual(executor.executed, SuccessfulAgent.order)
        self.assertEqual(agent.plan_calls, 6)
        self.assertEqual(executor.normal_stops, 1)
        self.assertEqual(executor.emergency_stops, 0)
        self.assertEqual(sum(event["kind"] == "qwen_verifier_result" for event in events), 10)
        action_events = [event for event in events
                         if event["kind"] == "pi0_verification_interval_executed"]
        self.assertEqual(len(action_events), 5)
        self.assertEqual(len(action_events[0]["payload"]["actions"]), 50)
        self.assertIn('"kind": "run_completed"', execution_log_text)
        self.assertTrue(summary_exists)

    def test_false_checks_continue_same_pi0_skill_without_becoming_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = FakeExecutor()
            store = Store(root / "events.sqlite3")
            agent = DelayedSuccessfulAgent()
            orchestrator = LiveT1Orchestrator(
                settings(root), load_action_skills(), agent, executor, store,
                run_id="continue_run",
            )
            result = orchestrator.run()
            state = store.load("continue_run")
            store.close()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(executor.executed[:3], ["pick_red_block"] * 3)
        self.assertEqual(state["skill_progress"]["pick_red_block"],
                         {"actions": 150, "checks": 3})
        self.assertEqual(agent.plan_calls, 6)
        self.assertEqual(executor.normal_stops, 1)
        self.assertEqual(executor.emergency_stops, 0)

    def test_camera_fault_immediately_disables_without_homing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = FakeExecutor(camera_fault=True)
            store = Store(root / "events.sqlite3")
            orchestrator = LiveT1Orchestrator(
                settings(root), load_action_skills(), SuccessfulAgent(), executor, store,
                run_id="fault_run",
            )
            with self.assertRaises(LiveSafetyFault):
                orchestrator.run()
            state = store.load("fault_run")
            store.close()
        self.assertEqual(state["status"], "safety_stopped")
        self.assertEqual(executor.normal_stops, 0)
        self.assertEqual(executor.emergency_stops, 1)

    def test_ctrl_c_homes_before_disabling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = InterruptingExecutor()
            store = Store(root / "events.sqlite3")
            orchestrator = LiveT1Orchestrator(
                settings(root), load_action_skills(), SuccessfulAgent(), executor, store,
                run_id="operator_interrupt_run",
            )
            with self.assertRaises(KeyboardInterrupt):
                orchestrator.run()
            state = store.load("operator_interrupt_run")
            events = store.events("operator_interrupt_run")
            store.close()
        self.assertEqual(state["status"], "operator_stopped")
        self.assertEqual(executor.normal_stops, 1)
        self.assertEqual(executor.emergency_stops, 0)
        operator_stop = next(event for event in events if event["kind"] == "operator_stop")
        self.assertEqual(operator_stop["payload"]["shutdown"],
                         "controlled_safe_pose_then_disable")
        self.assertEqual(sum(event["kind"] == "controlled_shutdown_completed"
                             for event in events), 1)

    def test_transient_qwen_network_failure_retries_without_more_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = FakeExecutor()
            store = Store(root / "events.sqlite3")
            agent = TransientVerifierAgent()
            orchestrator = LiveT1Orchestrator(
                settings(root), load_action_skills(), agent, executor, store,
                run_id="retry_run",
            )
            with patch("harnessvla.live_t1.time.sleep") as sleep:
                result = orchestrator.run()
            events = store.events("retry_run")
            store.close()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(executor.executed, SuccessfulAgent.order)
        retries = [event for event in events if event["kind"] == "qwen_transport_retry"]
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["payload"]["operation"], "verifier")
        sleep.assert_called_once_with(1.0)

    def test_wait_pose_stall_homes_before_disabling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = FakeExecutor()
            store = Store(root / "events.sqlite3")
            orchestrator = LiveT1Orchestrator(
                settings(root), load_action_skills(), NeverCompleteAgent(), executor, store,
                run_id="wait_pose_run",
            )
            with self.assertRaisesRegex(LiveT1Error, "settled at its wait pose"):
                orchestrator.run()
            state = store.load("wait_pose_run")
            events = store.events("wait_pose_run")
            store.close()
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(len(executor.executed), 5)
        self.assertEqual(executor.normal_stops, 1)
        self.assertEqual(executor.emergency_stops, 0)
        self.assertEqual(sum(event["kind"] == "skill_wait_pose_detected"
                             for event in events), 1)
        self.assertEqual(sum(event["kind"] == "controlled_shutdown_completed"
                             for event in events), 1)

    def test_skill_timeout_homes_before_disabling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = TimeoutExecutor()
            store = Store(root / "events.sqlite3")
            orchestrator = LiveT1Orchestrator(
                settings(root), load_action_skills(), NeverCompleteAgent(), executor, store,
                run_id="timeout_run",
            )
            with self.assertRaisesRegex(LiveT1Error, "timed out before visual completion"):
                orchestrator.run()
            state = store.load("timeout_run")
            events = store.events("timeout_run")
            store.close()
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(executor.normal_stops, 1)
        self.assertEqual(executor.emergency_stops, 0)
        self.assertEqual(sum(event["kind"] == "skill_execution_timeout"
                             for event in events), 1)
        self.assertEqual(sum(event["kind"] == "controlled_shutdown_completed"
                             for event in events), 1)


class WaitPoseDetectionTests(unittest.TestCase):
    def test_stationary_threshold_uses_measured_state_range(self):
        stationary = tuple(
            {"current_state": [0.001 * (index % 10), 0, 0, 0, 0, 0, 0.0005]}
            for index in range(50)
        )
        moving = tuple(
            {"current_state": [0.02 * index / 49, 0, 0, 0, 0, 0, 0]}
            for index in range(50)
        )
        self.assertTrue(_interval_is_stationary(stationary, 0.015, 0.001))
        self.assertFalse(_interval_is_stationary(moving, 0.015, 0.001))


class VisionAgentTests(unittest.TestCase):
    def test_planner_cannot_repeat_a_completed_goal_branch(self):
        response = {"choices": [{"message": {"content": json.dumps({
            "decision": "execute", "skill_id": "pick_red_block",
            "reason": "repeat the completed red step",
        })}}]}
        agent = QwenVisionAgent(
            qwen_config(), load_action_skills(), TASK_COMMAND,
            planner_transport=lambda _config, _payload: response,
        )
        with self.assertRaisesRegex(PlanningError, "unavailable or completed skill"):
            agent.choose_next(evidence(), {"red_in_blue_box"}, {}, [])

    def test_initial_scene_assessment_uses_yaml_task_and_both_cameras(self):
        payloads = []

        def verifier_transport(_config, payload):
            payloads.append(payload)
            return {"id": "initial", "choices": [{"message": {"content": json.dumps({
                "results": [
                    {"predicate": "red_in_blue_box", "value": "true",
                     "confidence": 0.98, "reason": "red is inside blue"},
                    {"predicate": "green_in_bowl", "value": "false",
                     "confidence": 0.97, "reason": "green is on the table"},
                    {"predicate": "red_button_triggered", "value": "unknown",
                     "confidence": 0.6, "reason": "button state is ambiguous"},
                ],
            })}}]}

        agent = QwenVisionAgent(
            qwen_config(), load_action_skills(), TASK_COMMAND,
            verifier_transport=verifier_transport,
        )
        verdicts = agent.assess_scene_goals(evidence())
        self.assertEqual(verdicts["red_in_blue_box"].value, "true")
        self.assertEqual(verdicts["green_in_bowl"].value, "false")
        self.assertEqual(len(payloads), 1)
        parts = payloads[0]["messages"][1]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in parts), 2)
        self.assertFalse(payloads[0]["enable_thinking"])
        self.assertEqual(payloads[0]["response_format"], {"type": "json_object"})
        self.assertIn(TASK_COMMAND, parts[-1]["text"])

    def test_planner_and_verifier_send_dual_camera_images(self):
        registry = load_action_skills()
        payloads = []

        def planner_transport(_config, payload):
            payloads.append(payload)
            return {"id": "p", "choices": [{"message": {"content": json.dumps({
                "decision": "execute", "skill_id": "pick_red_block", "reason": "red visible",
            })}}], "usage": {"total_tokens": 10}}

        def verifier_transport(_config, payload):
            payloads.append(payload)
            return {"id": "v", "choices": [{"message": {"content": json.dumps({
                "predicate": "holding_red_block", "value": "true",
                "confidence": 0.9, "reason": "block moved with gripper",
            })}}]}

        agent = QwenVisionAgent(
            qwen_config(), registry, TASK_COMMAND, planner_transport, verifier_transport,
        )
        decision = agent.choose_next(evidence(), set(), {}, [])
        verdict = agent.verify(registry.by_id()["pick_red_block"], evidence(), evidence())
        self.assertEqual(decision.skill_id, "pick_red_block")
        self.assertEqual(verdict.value, "true")
        planner_parts = payloads[0]["messages"][1]["content"]
        verifier_parts = payloads[1]["messages"][1]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in planner_parts), 2)
        self.assertEqual(sum(part["type"] == "image_url" for part in verifier_parts), 4)
        self.assertTrue(payloads[0]["enable_thinking"])
        self.assertIn(TASK_COMMAND, planner_parts[-1]["text"])
        self.assertFalse(payloads[1]["enable_thinking"])
        self.assertEqual(payloads[1]["response_format"], {"type": "json_object"})
        verifier_text = payloads[1]["messages"][1]["content"][-1]["text"]
        self.assertIn(TASK_COMMAND, verifier_text)
        self.assertIn("lifted clear of the tabletop", verifier_text)

    def test_planner_cannot_skip_an_unverified_precondition(self):
        response = {"choices": [{"message": {"content": json.dumps({
            "decision": "execute", "skill_id": "place_red_in_teal_box", "reason": "try place",
        })}}]}
        agent = QwenVisionAgent(
            qwen_config(), load_action_skills(), TASK_COMMAND,
            lambda _config, _payload: response,
        )
        with self.assertRaisesRegex(PlanningError, "unmet preconditions"):
            agent.choose_next(evidence(), set(), {}, [])

    def test_verifier_normalizes_unambiguous_json_boolean(self):
        response = {"choices": [{"message": {"content": json.dumps({
            "predicate": "holding_red_block", "value": False,
            "confidence": 0.8, "reason": "gripper remains empty",
        })}}]}
        agent = QwenVisionAgent(
            qwen_config(), load_action_skills(), TASK_COMMAND,
            verifier_transport=lambda _config, _payload: response,
        )
        verdict = agent.verify(load_action_skills().by_id()["pick_red_block"],
                               evidence(), evidence())
        self.assertEqual(verdict.value, "false")


if __name__ == "__main__":
    unittest.main()
