"""
SQLite backend for durable execution state.

Schema:
  workflows     — registered workflow definitions
  instances     — workflow execution instances
  steps         — individual step executions within an instance
  checkpoints   — serialized input/output for each step
"""

import sqlite3
import json
import time
from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class StepRecord:
    id: int
    instance_id: str
    step_name: str
    status: str  # pending, running, completed, failed
    attempt: int
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    error_message: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class DurableBackend:
    """SQLite-backed persistence for durable execution."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                definition_json TEXT NOT NULL,
                created_at REAL NOT NULL DEFAULT (unixepoch())
            );

            CREATE TABLE IF NOT EXISTS instances (
                id TEXT PRIMARY KEY,
                workflow_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                input_data TEXT,
                output_data TEXT,
                started_at REAL,
                completed_at REAL,
                created_at REAL NOT NULL DEFAULT (unixepoch()),
                FOREIGN KEY (workflow_name) REFERENCES workflows(name)
            );

            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempt INTEGER NOT NULL DEFAULT 0,
                input_data TEXT,
                output_data TEXT,
                error_message TEXT,
                started_at REAL,
                completed_at REAL,
                FOREIGN KEY (instance_id) REFERENCES instances(id)
            );

            CREATE INDEX IF NOT EXISTS idx_steps_instance
                ON steps(instance_id, step_name);
        """)
        self.conn.commit()

    # === Workflow CRUD ===

    def register_workflow(self, name: str, definition: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO workflows (name, definition_json) VALUES (?, ?)",
            (name, json.dumps(definition)),
        )
        self.conn.commit()

    def get_workflow(self, name: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT definition_json FROM workflows WHERE name = ?", (name,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    # === Instance CRUD ===

    def create_instance(
        self, instance_id: str, workflow_name: str, input_data: Any = None
    ) -> str:
        self.conn.execute(
            "INSERT INTO instances (id, workflow_name, input_data, status) VALUES (?, ?, ?, 'pending')",
            (instance_id, workflow_name, json.dumps(input_data)),
        )
        self.conn.commit()
        return instance_id

    def update_instance_status(
        self,
        instance_id: str,
        status: str,
        output_data: Any = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
    ):
        fields = ["status = ?"]
        params: list = [status]
        if output_data is not None:
            fields.append("output_data = ?")
            params.append(json.dumps(output_data))
        if started_at is not None:
            fields.append("started_at = ?")
            params.append(started_at)
        if completed_at is not None:
            fields.append("completed_at = ?")
            params.append(completed_at)
        params.append(instance_id)
        self.conn.execute(
            f"UPDATE instances SET {', '.join(fields)} WHERE id = ?", params
        )
        self.conn.commit()

    def get_instance(self, instance_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM instances WHERE id = ?", (instance_id,)
        ).fetchone()
        return dict(row) if row else None

    # === Step CRUD ===

    def get_step(self, instance_id: str, step_name: str) -> Optional[StepRecord]:
        row = self.conn.execute(
            "SELECT * FROM steps WHERE instance_id = ? AND step_name = ? ORDER BY attempt DESC LIMIT 1",
            (instance_id, step_name),
        ).fetchone()
        if not row:
            return None
        return StepRecord(
            id=row["id"],
            instance_id=row["instance_id"],
            step_name=row["step_name"],
            status=row["status"],
            attempt=row["attempt"],
            input_data=json.loads(row["input_data"]) if row["input_data"] else None,
            output_data=json.loads(row["output_data"]) if row["output_data"] else None,
            error_message=row["error_message"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def record_step_start(
        self, instance_id: str, step_name: str, attempt: int, input_data: Any
    ):
        self.conn.execute(
            """INSERT INTO steps (instance_id, step_name, status, attempt, input_data, started_at)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (instance_id, step_name, attempt, json.dumps(input_data), time.time()),
        )
        self.conn.commit()

    def record_step_complete(
        self, instance_id: str, step_name: str, output_data: Any
    ):
        self.conn.execute(
            """UPDATE steps SET status = 'completed', output_data = ?, completed_at = ?
               WHERE instance_id = ? AND step_name = ? AND status = 'running'""",
            (json.dumps(output_data), time.time(), instance_id, step_name),
        )
        self.conn.commit()

    def record_step_failed(
        self, instance_id: str, step_name: str, error_message: str
    ):
        self.conn.execute(
            """UPDATE steps SET status = 'failed', error_message = ?, completed_at = ?
               WHERE instance_id = ? AND step_name = ? AND status = 'running'""",
            (error_message, time.time(), instance_id, step_name),
        )
        self.conn.commit()

    def get_all_steps(self, instance_id: str) -> list[StepRecord]:
        rows = self.conn.execute(
            """SELECT s.* FROM steps s
               WHERE s.instance_id = ?
               AND s.id IN (
                   SELECT MAX(id) FROM steps WHERE instance_id = ? GROUP BY step_name
               )
               ORDER BY s.id""",
            (instance_id, instance_id),
        ).fetchall()
        results = []
        for row in rows:
            results.append(
                StepRecord(
                    id=row["id"],
                    instance_id=row["instance_id"],
                    step_name=row["step_name"],
                    status=row["status"],
                    attempt=row["attempt"],
                    input_data=json.loads(row["input_data"])
                    if row["input_data"]
                    else None,
                    output_data=json.loads(row["output_data"])
                    if row["output_data"]
                    else None,
                    error_message=row["error_message"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                )
            )
        return results

    # === Utilities ===

    def get_stats(self) -> dict:
        total_instances = self.conn.execute(
            "SELECT COUNT(*) FROM instances"
        ).fetchone()[0]
        total_steps = self.conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        completed = self.conn.execute(
            "SELECT COUNT(*) FROM instances WHERE status = 'completed'"
        ).fetchone()[0]
        failed = self.conn.execute(
            "SELECT COUNT(*) FROM instances WHERE status = 'failed'"
        ).fetchone()[0]
        return {
            "total_instances": total_instances,
            "total_steps": total_steps,
            "completed": completed,
            "failed": failed,
        }

    def close(self):
        self.conn.close()
