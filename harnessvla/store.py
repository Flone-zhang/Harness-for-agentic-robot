"""SQLite event log and atomic run snapshots; one orchestrator owns state changes."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                state_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                subtask_id TEXT,
                monotonic_ns INTEGER NOT NULL,
                wall_time_epoch REAL NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                checkpoint_hash TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS events_run_seq ON events(run_id, seq);
        """)

    def close(self) -> None:
        self.connection.close()

    def create(self, state: dict[str, Any]) -> None:
        state["version"] = 0
        state["last_monotonic_ns"] = time.monotonic_ns()
        with self.connection:
            self.connection.execute(
                "INSERT INTO runs(run_id, version, state_json) VALUES (?, ?, ?)",
                (state["run_id"], 0, json.dumps(state, sort_keys=True)),
            )
            self._insert_event(state, "run_started", {"plan": state["plan"]}, None)

    def load(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown run_id: {run_id}")
        return json.loads(row["state_json"])

    def commit(
        self,
        state: dict[str, Any],
        kind: str,
        payload: dict[str, Any],
        subtask_id: str | None,
    ) -> None:
        old_version = state["version"]
        previous_ns = state["last_monotonic_ns"]
        state["last_monotonic_ns"] = max(time.monotonic_ns(), previous_ns + 1)
        state["version"] = old_version + 1
        try:
            with self.connection:
                cursor = self.connection.execute(
                    "UPDATE runs SET version = ?, state_json = ? WHERE run_id = ? AND version = ?",
                    (state["version"], json.dumps(state, sort_keys=True), state["run_id"], old_version),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("run changed concurrently; reload before retrying")
                self._insert_event(state, kind, payload, subtask_id)
        except Exception:
            state["version"] = old_version
            state["last_monotonic_ns"] = previous_ns
            raise

    def _insert_event(
        self, state: dict[str, Any], kind: str, payload: dict[str, Any], subtask_id: str | None
    ) -> None:
        self.connection.execute(
            """INSERT INTO events
            (run_id, task_id, subtask_id, monotonic_ns, wall_time_epoch, kind,
             payload_json, config_hash, code_hash, checkpoint_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state["run_id"], state["task_id"], subtask_id,
                state["last_monotonic_ns"], time.time(), kind,
                json.dumps(payload, sort_keys=True),
                state["config_hash"], state["code_hash"], state["checkpoint_hash"],
            ),
        )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result
