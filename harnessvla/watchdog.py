"""Pure transition rules: replayable without a model or external service."""

from .contracts import Decision, DecisionInput


def decide(data: DecisionInput) -> Decision:
    if data.safety_fault:
        return Decision("fail", f"safety_fault:{data.safety_fault}", data.attempt)
    if data.image_quality != "ok":
        if data.observation_waits < data.max_observation_waits:
            return Decision("wait", f"invalid_observation:{data.image_quality}", data.attempt)
        return Decision("fail", "observation_wait_exhausted", data.attempt)
    if data.predicate == "true":
        return Decision("end", "predicate_verified", data.attempt)
    if data.execution_status == "budget_exhausted":
        return Decision("fail", "action_budget_exhausted", data.attempt)
    if data.execution_status == "unsupported":
        return Decision("fail", "skill_unsupported", data.attempt)
    if data.fault in {"drop", "box_moved"}:
        if data.replans < data.max_replans:
            return Decision("replan", f"scene_changed:{data.fault}", data.attempt)
        return Decision("fail", "replan_exhausted", data.attempt)
    if data.predicate == "unknown":
        if data.observation_waits < data.max_observation_waits:
            return Decision("wait", "predicate_unknown", data.attempt)
        return Decision("fail", "predicate_unverifiable", data.attempt)
    if data.attempt < data.max_attempts:
        if data.fault == "empty_grasp":
            return Decision("recover", "verified_mock_regrasp", data.attempt)
        return Decision("replay", "predicate_false_or_execution_failed", data.attempt)
    if data.replans < data.max_replans:
        return Decision("replan", "attempts_exhausted", data.attempt)
    return Decision("fail", "attempts_and_replans_exhausted", data.attempt)
