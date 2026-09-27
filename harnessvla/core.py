"""Single-owner, evidence-gated task state machine for offline mock runs."""

import hashlib
import time
import uuid
from typing import Any

from .config import Config, fixed_t1_plan, validate_plan
from .contracts import DecisionInput, Fact, PredicateResult, Subtask
from .mock import execute, observe, predicate
from .store import Store, code_hash
from .watchdog import decide

SCENARIO_KINDS = {"empty_grasp", "drop", "box_moved", "black_frame", "occlusion", "false_completion", "missing_signal"}


def validate_scenario(scenario: list[dict[str, Any]], plan: list[Subtask]) -> list[dict[str, Any]]:
    ids = {item.subtask_id for item in plan}
    cleaned = []
    for entry in scenario:
        if set(entry) - {"subtask_id", "kind", "on_attempt", "remaining"}:
            raise ValueError(f"unknown scenario keys: {entry}")
        if entry.get("subtask_id") not in ids or entry.get("kind") not in SCENARIO_KINDS:
            raise ValueError(f"invalid scenario: {entry}")
        on_attempt = entry.get("on_attempt", 1)
        remaining = entry.get("remaining", 1)
        if not isinstance(on_attempt, int) or on_attempt < 1 or not isinstance(remaining, int) or remaining < 1:
            raise ValueError(f"invalid scenario counters: {entry}")
        cleaned.append({"subtask_id": entry["subtask_id"], "kind": entry["kind"],
                        "on_attempt": on_attempt, "remaining": remaining})
    return cleaned


class Harness:
    def __init__(self, store: Store):
        self.store = store

    @staticmethod
    def _assert_code_compatible(state: dict) -> None:
        if state["code_hash"] != code_hash():
            raise RuntimeError("code changed since run start; do not resume under a different implementation")

    def start(self, config: Config, scenario: list[dict[str, Any]] | None = None,
              task_id: str = "T1") -> dict[str, Any]:
        if task_id not in {"T1", "T2", "T3"}:
            raise ValueError("task_id must be T1, T2 or T3")
        plan = fixed_t1_plan(config)
        validate_plan(plan)
        scenario = validate_scenario(scenario or [], plan)
        now_ns = time.monotonic_ns()
        state = {
            "run_id": str(uuid.uuid4()), "task_id": task_id, "status": "running",
            "index": 0, "attempt": 0, "replans": 0, "observation_waits": 0,
            "subtask_action_steps": 0, "total_action_steps": 0,
            "pending_verification": False,
            "created_at_epoch": time.time(), "subtask_started_epoch": time.time(),
            "plan": [item.json() for item in plan], "goals": {},
            "world": {"red_in_blue_box": False, "green_in_bowl": False,
                      "red_button_triggered": False, "box_pose": "initial"},
            "facts": {"box_pose": Fact("box_pose", "initial", "scene", "mock_initial_state",
                                        now_ns, 30.0, True).json()},
            "scenario": scenario, "config": config.json(),
            "config_hash": config.hash(), "code_hash": code_hash(),
            "checkpoint_hash": hashlib.sha256(config.checkpoint_id.encode()).hexdigest(),
            "terminal_reason": None,
        }
        self.store.create(state)
        return state

    def _scenario_fault(self, state: dict, item: Subtask) -> str | None:
        for entry in list(state["scenario"]):
            if entry["subtask_id"] == item.subtask_id and entry["on_attempt"] == state["attempt"] + 1:
                entry["remaining"] -= 1
                if entry["remaining"] == 0:
                    state["scenario"].remove(entry)
                return entry["kind"]
        return None

    @staticmethod
    def _update_box_fact(state: dict, observation) -> dict | None:
        value = observation.signals.get("box_pose")
        previous = state["facts"].get("box_pose")
        if observation.image_quality != "ok" or value is None:
            return None
        if previous and previous["valid"] and previous["value"] == value:
            return None
        invalidated = None
        if previous and previous["valid"]:
            previous["valid"] = False
            invalidated = dict(previous)
        state["facts"]["box_pose"] = Fact(
            "box_pose", value, "scene", f"mock_observation:{observation.observation_id}",
            observation.monotonic_ns, 30.0, True,
        ).json()
        return invalidated

    @staticmethod
    def _expire_facts(state: dict, now_ns: int) -> list[dict]:
        expired = []
        for fact in state["facts"].values():
            ttl = fact.get("expires_after_s")
            if fact["valid"] and ttl is not None and (now_ns - fact["observed_monotonic_ns"]) / 1e9 > ttl:
                fact["valid"] = False
                expired.append(dict(fact))
        return expired

    @staticmethod
    def _decision_input(state: dict, config: Config, result, pred: PredicateResult,
                        quality: str, fault: str | None) -> DecisionInput:
        return DecisionInput(
            safety_fault=None,
            image_quality=quality,
            predicate=pred.value,
            execution_status=result.status if result else "not_run",
            fault=fault,
            attempt=state["attempt"], max_attempts=config.max_attempts,
            replans=state["replans"], max_replans=config.max_replans,
            observation_waits=state["observation_waits"],
            max_observation_waits=config.max_observation_waits,
            progress=result.progress if result else 0.0,
        )

    def _apply_decision(self, state: dict, item: Subtask, decision, pred: PredicateResult) -> dict | None:
        final_checks = None
        if decision.action == "end":
            if pred.value != "true" or not pred.evidence_ref:
                raise AssertionError("completion requires independent evidence")
            state["goals"][item.success_predicate] = pred.json()
            state["index"] += 1
            state["attempt"] = 0
            state["replans"] = 0
            state["observation_waits"] = 0
            state["subtask_action_steps"] = 0
            state["pending_verification"] = False
            state["subtask_started_epoch"] = time.time()
            if state["index"] == len(state["plan"]):
                final_observation = observe(state["world"])
                final_checks = {
                    prior["success_predicate"]: predicate(prior["success_predicate"], final_observation).json()
                    for prior in state["plan"]
                }
                if all(check["value"] == "true" and check["evidence_ref"] for check in final_checks.values()):
                    state["goals"].update(final_checks)
                    state["status"] = "complete"
                    state["terminal_reason"] = "all_three_predicates_verified"
                else:
                    state["status"] = "failed"
                    state["terminal_reason"] = "final_goal_revalidation_failed"
        elif decision.action == "wait":
            state["observation_waits"] += 1
        elif decision.action in {"replay", "recover"}:
            state["observation_waits"] = 0
            state["pending_verification"] = False
        elif decision.action == "replan":
            # Offline fixed-plan suffix revalidation; no Qwen service is claimed.
            state["replans"] += 1
            state["attempt"] = 0
            state["observation_waits"] = 0
            state["pending_verification"] = False
        elif decision.action == "fail":
            state["status"] = "failed"
            state["terminal_reason"] = decision.reason
        return final_checks

    def step(self, run_id: str) -> dict[str, Any]:
        state = self.store.load(run_id)
        if state["status"] != "running":
            return state
        self._assert_code_compatible(state)
        config = Config.load_dict(state["config"])
        item = Subtask(**state["plan"][state["index"]])
        if time.time() - state["created_at_epoch"] > config.task_timeout_s:
            state["status"], state["terminal_reason"] = "failed", "task_timeout"
            self.store.commit(state, "task_failed", {"reason": "task_timeout"}, item.subtask_id)
            return state
        if time.time() - state["subtask_started_epoch"] > item.timeout_s:
            state["status"], state["terminal_reason"] = "failed", "subtask_timeout"
            self.store.commit(state, "task_failed", {"reason": "subtask_timeout"}, item.subtask_id)
            return state
        if any(key not in state["goals"] for key in item.prerequisites):
            state["status"], state["terminal_reason"] = "failed", "prerequisite_not_verified"
            self.store.commit(state, "task_failed", {"reason": "prerequisite_not_verified"}, item.subtask_id)
            return state

        fault = None if state["pending_verification"] else self._scenario_fault(state, item)
        if fault == "box_moved":
            state["world"]["box_pose"] = "moved"
        quality = fault if fault in {"black_frame", "occlusion"} else "ok"
        pre_observation = observe(state["world"], quality)
        expired_facts = self._expire_facts(state, pre_observation.monotonic_ns)
        invalidated_fact = self._update_box_fact(state, pre_observation)
        result = None
        post_observation = pre_observation

        if quality == "ok" and fault != "box_moved" and not state["pending_verification"]:
            if state["subtask_action_steps"] >= item.action_budget:
                from .contracts import ExecutionResult
                result = ExecutionResult("budget_exhausted", 0, pre_observation.observation_id, 0.0)
            else:
                state["attempt"] += 1
                result = execute(item, state["world"], pre_observation, fault)
                state["subtask_action_steps"] += result.action_steps
                state["total_action_steps"] += result.action_steps
                post_observation = observe(state["world"])

        pred = predicate(item.success_predicate, post_observation)
        if pred.value == "unknown" and result is not None:
            state["pending_verification"] = True
        if quality == "ok" and pred.value != "unknown":
            state["observation_waits"] = 0
        if result and state["subtask_action_steps"] > item.action_budget:
            raise AssertionError("action budget exceeded")
        decision_input = self._decision_input(state, config, result, pred, quality, fault)
        decision = decide(decision_input)
        final_checks = self._apply_decision(state, item, decision, pred)
        self.store.commit(state, "watchdog_decision", {
            "pre_observation": pre_observation.json(),
            "observation": post_observation.json(),
            "execution_result": result.json() if result else None,
            "predicate_result": pred.json(),
            "decision_input": decision_input.json(),
            "decision": decision.json(),
            "invalidated_fact": invalidated_fact,
            "expired_facts": expired_facts,
            "final_checks": final_checks,
            "state_after": {"status": state["status"], "index": state["index"],
                            "attempt": state["attempt"], "replans": state["replans"]},
        }, item.subtask_id)
        return state

    def run(self, run_id: str, pause_after_goals: int | None = None) -> dict[str, Any]:
        if pause_after_goals is not None and not 1 <= pause_after_goals <= 2:
            raise ValueError("pause_after_goals must be 1 or 2")
        for _ in range(1000):
            state = self.store.load(run_id)
            if state["status"] != "running":
                return state
            if pause_after_goals is not None and state["index"] >= pause_after_goals:
                state["status"] = "paused"
                self.store.commit(state, "run_paused", {"after_goals": state["index"]}, None)
                return state
            self.step(run_id)
        state = self.store.load(run_id)
        state["status"], state["terminal_reason"] = "failed", "orchestrator_step_guard"
        self.store.commit(state, "task_failed", {"reason": "orchestrator_step_guard"}, None)
        return state

    def resume(self, run_id: str, mock_box_pose: str | None = None) -> dict[str, Any]:
        state = self.store.load(run_id)
        if state["status"] != "paused":
            raise ValueError("only a paused run can be resumed")
        self._assert_code_compatible(state)
        if mock_box_pose is not None:
            if not mock_box_pose or len(mock_box_pose) > 80:
                raise ValueError("invalid mock box pose label")
            state["world"]["box_pose"] = mock_box_pose
        observation = observe(state["world"])
        expired_facts = self._expire_facts(state, observation.monotonic_ns)
        invalidated = self._update_box_fact(state, observation)
        checks = {}
        for item_data in state["plan"][:state["index"]]:
            item = Subtask(**item_data)
            checks[item.success_predicate] = predicate(item.success_predicate, observation).json()
        if any(value["value"] != "true" for value in checks.values()):
            self.store.commit(state, "resume_revalidation_failed", {
                "observation": observation.json(), "checks": checks,
                "invalidated_fact": invalidated,
                "expired_facts": expired_facts,
            }, None)
            return state
        state["status"] = "running"
        self.store.commit(state, "run_resumed", {
            "observation": observation.json(), "checks": checks,
            "invalidated_fact": invalidated,
            "expired_facts": expired_facts,
        }, None)
        return self.run(run_id)


def replay_decisions(store: Store, run_id: str) -> dict[str, Any]:
    checked = 0
    for event in store.events(run_id):
        if event["kind"] != "watchdog_decision":
            continue
        payload = event["payload"]
        actual = decide(DecisionInput(**payload["decision_input"]))
        if actual.json() != payload["decision"]:
            return {"ok": False, "checked": checked, "mismatch_seq": event["seq"]}
        checked += 1
    return {"ok": True, "checked": checked}
