"""Offline command-line entry points. All commands are mock/read-only."""

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .core import Harness, replay_decisions
from .pi0 import DEFAULT_CHECKPOINT_DIR, inspect_checkpoint, preview_chunk
from .store import Store


def _summary(state: dict) -> dict:
    return {
        "run_id": state["run_id"], "task_id": state["task_id"],
        "status": state["status"], "current_index": state["index"],
        "verified_goals": list(state["goals"]),
        "total_action_steps": state["total_action_steps"],
        "terminal_reason": state["terminal_reason"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HarnessVLA offline mock runner")
    parser.add_argument("--db", type=Path, default=Path("runs/harnessvla.sqlite3"))
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="start fixed T1 mock plan")
    start.add_argument("--config", type=Path)
    start.add_argument("--scenario", type=Path)
    start.add_argument("--pause-after", type=int, choices=(1, 2))
    start.add_argument("--task", choices=("T1", "T2", "T3"), help="defaults to T2 when paused, T3 with scenario, else T1")
    resume = commands.add_parser("resume", help="re-observe and continue a paused mock run")
    resume.add_argument("run_id")
    resume.add_argument("--mock-box-pose", help="synthetic scene change before revalidation")
    status = commands.add_parser("status")
    status.add_argument("run_id")
    events = commands.add_parser("events")
    events.add_argument("run_id")
    replay = commands.add_parser("replay")
    replay.add_argument("run_id")
    checkpoint = commands.add_parser("pi0-checkpoint", help="inspect a local PI0 export without loading model weights")
    checkpoint.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    checkpoint.add_argument("--hash-model", action="store_true", help="stream the entire model file to calculate SHA-256")
    preview = commands.add_parser("pi0-preview", help="check a recorded PI0 action chunk; never sends commands")
    preview.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    preview.add_argument("--state-json", type=Path, required=True, help="JSON array of six joint radians plus gripper metres")
    preview.add_argument("--chunk-json", type=Path, required=True, help="JSON array of 7D absolute actions")
    preview.add_argument("--max-joint-step", type=float, default=0.05)
    preview.add_argument("--max-gripper-step", type=float, default=0.005)
    args = parser.parse_args(argv)
    store = None
    try:
        if args.command == "pi0-checkpoint":
            output = inspect_checkpoint(args.checkpoint, hash_model=args.hash_model).json()
        elif args.command == "pi0-preview":
            contract = inspect_checkpoint(args.checkpoint)
            current_state = json.loads(args.state_json.read_text(encoding="utf-8"))
            chunk = json.loads(args.chunk_json.read_text(encoding="utf-8"))
            output = preview_chunk(contract, current_state, chunk,
                                   args.max_joint_step, args.max_gripper_step)
        else:
            store = Store(args.db)
            harness = Harness(store)
            if args.command == "start":
                config = Config.load(args.config)
                scenario = [] if args.scenario is None else json.loads(args.scenario.read_text(encoding="utf-8"))
                task_id = args.task or ("T2" if args.pause_after else "T3" if scenario else "T1")
                state = harness.start(config, scenario, task_id)
                output = _summary(harness.run(state["run_id"], args.pause_after))
            elif args.command == "resume":
                output = _summary(harness.resume(args.run_id, args.mock_box_pose))
            elif args.command == "status":
                output = _summary(store.load(args.run_id))
            elif args.command == "events":
                output = store.events(args.run_id)
            else:
                output = replay_decisions(store, args.run_id)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
