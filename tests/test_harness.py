import tempfile
import unittest
from pathlib import Path

from harnessvla.config import Config, fixed_t1_plan, validate_plan
from harnessvla.contracts import Subtask
from harnessvla.core import Harness, replay_decisions
from harnessvla.mock import observe
from harnessvla.safety import JointCommand, PiperAdapter, SafetyError, SafetyLimits, validate_joint_command
from harnessvla.store import Store


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "events.sqlite3"
        self.store = Store(self.path)
        self.harness = Harness(self.store)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_t1_requires_three_independent_predicates(self):
        state = self.harness.start(Config())
        done = self.harness.run(state["run_id"])
        self.assertEqual(done["status"], "complete")
        self.assertEqual(done["total_action_steps"], 3)
        self.assertEqual(set(done["goals"]), {
            "red_in_blue_box", "green_in_bowl", "red_button_triggered",
        })
        self.assertTrue(all(value["source"] == "mock_world_sensor" and value["evidence_ref"]
                            for value in done["goals"].values()))
        self.assertEqual(replay_decisions(self.store, state["run_id"]), {"ok": True, "checked": 3})
        events = self.store.events(state["run_id"])
        self.assertEqual([event["kind"] for event in events], ["run_started"] + ["watchdog_decision"] * 3)
        self.assertEqual([event["subtask_id"] for event in events[1:]],
                         ["red_to_blue", "green_to_bowl", "press_button"])
        self.assertEqual(sorted(event["monotonic_ns"] for event in events),
                         [event["monotonic_ns"] for event in events])

    def test_empty_grasp_recovers_without_false_completion(self):
        state = self.harness.start(Config(), [{"subtask_id": "red_to_blue", "kind": "empty_grasp"}], "T3")
        first = self.harness.step(state["run_id"])
        self.assertEqual(first["index"], 0)
        event = self.store.events(state["run_id"])[-1]
        self.assertEqual(event["payload"]["decision"]["action"], "recover")
        done = self.harness.run(state["run_id"])
        self.assertEqual(done["status"], "complete")
        self.assertEqual(done["total_action_steps"], 4)
        self.assertTrue(replay_decisions(self.store, state["run_id"])["ok"])

    def test_black_frame_waits_without_action(self):
        state = self.harness.start(Config(), [{"subtask_id": "red_to_blue", "kind": "black_frame"}])
        first = self.harness.step(state["run_id"])
        self.assertEqual(first["total_action_steps"], 0)
        self.assertEqual(self.store.events(state["run_id"])[-1]["payload"]["decision"]["action"], "wait")
        self.assertEqual(self.harness.run(state["run_id"])["status"], "complete")
        self.assertTrue(replay_decisions(self.store, state["run_id"])["ok"])

    def test_occlusion_exhausts_observation_waits(self):
        state = self.harness.start(Config(), [{"subtask_id": "red_to_blue", "kind": "occlusion", "remaining": 3}])
        done = self.harness.run(state["run_id"])
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["terminal_reason"], "observation_wait_exhausted")
        self.assertEqual(done["total_action_steps"], 0)
        self.assertTrue(replay_decisions(self.store, state["run_id"])["ok"])

    def test_false_completion_does_not_advance(self):
        state = self.harness.start(Config(), [{"subtask_id": "red_to_blue", "kind": "false_completion"}])
        first = self.harness.step(state["run_id"])
        self.assertEqual(first["index"], 0)
        event = self.store.events(state["run_id"])[-1]["payload"]
        self.assertEqual(event["execution_result"]["status"], "success")
        self.assertEqual(event["predicate_result"]["value"], "false")
        self.assertEqual(event["decision"]["action"], "replay")

    def test_retry_exhaustion_is_bounded(self):
        config = Config(max_attempts=1, max_replans=0)
        state = self.harness.start(config, [{"subtask_id": "red_to_blue", "kind": "false_completion"}])
        done = self.harness.run(state["run_id"])
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["terminal_reason"], "attempts_and_replans_exhausted")
        self.assertEqual(done["index"], 0)

    def test_budget_exhaustion_is_not_success(self):
        config = Config(action_budget=1, max_attempts=2)
        state = self.harness.start(config, [{"subtask_id": "red_to_blue", "kind": "false_completion"}])
        done = self.harness.run(state["run_id"])
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["terminal_reason"], "action_budget_exhausted")

    def test_unknown_signal_never_advances(self):
        state = self.harness.start(Config(), [{"subtask_id": "press_button", "kind": "missing_signal"}])
        done = self.harness.run(state["run_id"])
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["terminal_reason"], "predicate_unverifiable")
        self.assertEqual(done["index"], 2)
        self.assertNotIn("red_button_triggered", done["goals"])

    def test_box_move_invalidates_fact_and_replans(self):
        state = self.harness.start(Config(), [{"subtask_id": "red_to_blue", "kind": "box_moved"}])
        first = self.harness.step(state["run_id"])
        self.assertEqual(first["replans"], 1)
        self.assertEqual(first["facts"]["box_pose"]["value"], "moved")
        event = self.store.events(state["run_id"])[-1]["payload"]
        self.assertEqual(event["decision"]["action"], "replan")
        self.assertEqual(event["invalidated_fact"]["value"], "initial")
        self.assertEqual(self.harness.run(state["run_id"])["status"], "complete")
        self.assertTrue(replay_decisions(self.store, state["run_id"])["ok"])

    def test_drop_replans_then_retries(self):
        state = self.harness.start(Config(), [{"subtask_id": "red_to_blue", "kind": "drop"}])
        first = self.harness.step(state["run_id"])
        self.assertEqual(first["replans"], 1)
        self.assertEqual(first["world"]["current_object_pose"], None)
        self.assertEqual(self.harness.run(state["run_id"])["status"], "complete")
        self.assertTrue(replay_decisions(self.store, state["run_id"])["ok"])

    def test_restart_resume_and_fact_conflict(self):
        state = self.harness.start(Config())
        paused = self.harness.run(state["run_id"], pause_after_goals=1)
        self.assertEqual(paused["status"], "paused")
        self.store.close()
        self.store = Store(self.path)
        self.harness = Harness(self.store)
        done = self.harness.resume(state["run_id"], mock_box_pose="new_pose")
        self.assertEqual(done["status"], "complete")
        self.assertEqual(done["total_action_steps"], 3)
        resumed = next(event for event in self.store.events(state["run_id"]) if event["kind"] == "run_resumed")
        self.assertEqual(resumed["payload"]["invalidated_fact"]["value"], "initial")
        self.assertEqual(resumed["payload"]["checks"]["red_in_blue_box"]["value"], "true")

    def test_resume_refuses_unverified_prior_goal(self):
        state = self.harness.start(Config())
        self.harness.run(state["run_id"], pause_after_goals=1)
        changed = self.store.load(state["run_id"])
        changed["world"]["red_in_blue_box"] = False
        self.store.commit(changed, "mock_scene_changed", {"red_in_blue_box": False}, None)
        paused = self.harness.resume(state["run_id"])
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["total_action_steps"], 1)

    def test_expired_scene_fact_is_refreshed_on_resume(self):
        state = self.harness.start(Config())
        self.harness.run(state["run_id"], pause_after_goals=1)
        changed = self.store.load(state["run_id"])
        changed["facts"]["box_pose"]["observed_monotonic_ns"] = 0
        self.store.commit(changed, "mock_clock_advanced", {}, None)
        done = self.harness.resume(state["run_id"])
        self.assertEqual(done["status"], "complete")
        resumed = next(event for event in self.store.events(state["run_id"]) if event["kind"] == "run_resumed")
        self.assertEqual(resumed["payload"]["expired_facts"][0]["key"], "box_pose")
        self.assertTrue(done["facts"]["box_pose"]["valid"])

    def test_unknown_skill_rejected_before_execution(self):
        plan = fixed_t1_plan(Config())
        plan[0] = Subtask("red_to_blue", "not_registered", "x", [], "red_in_blue_box",
                          "red_in_blue_box", 1, 1.0)
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_safety_gate_and_disabled_piper(self):
        limits = SafetyLimits((-1.0,) * 6, (1.0,) * 6, 0.2, 0.0, 0.08, 1.0)
        obs = observe({})
        good = JointCommand((0.0,) * 6, 0.1, 0.02)
        validate_joint_command(good, obs, limits, obs.monotonic_ns + 100)
        with self.assertRaises(SafetyError):
            validate_joint_command(good, obs, limits, obs.monotonic_ns + 2_000_000_000)
        with self.assertRaises(SafetyError):
            validate_joint_command(JointCommand((float("nan"),) * 6, 0.1, 0.02),
                                   obs, limits, obs.monotonic_ns + 100)
        with self.assertRaises(SafetyError):
            validate_joint_command(JointCommand((2.0,) * 6, 0.1, 0.02),
                                   obs, limits, obs.monotonic_ns + 100)
        with self.assertRaises(SafetyError):
            PiperAdapter().send(good)


if __name__ == "__main__":
    unittest.main()
