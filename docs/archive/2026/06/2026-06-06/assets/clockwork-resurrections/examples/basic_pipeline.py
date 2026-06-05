"""
Example 1: A data pipeline with crash recovery.

Simulates: fetching data from an API, processing each item,
aggregating results, then generating a report. If any step fails,
the entire pipeline can be resumed from the failure point.
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from durable_mini import DurableEngine, Workflow

# Use a persistent database so we can demonstrate crash recovery
DB_PATH = "/tmp/durable_pipeline_example.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

engine = DurableEngine(DB_PATH)


# === Define pipeline steps ===


@engine.step(description="Fetch user IDs from the data source")
def fetch_users():
    """Simulates an API call to fetch user IDs."""
    time.sleep(0.1)  # Simulate network latency
    return [101, 102, 103, 104, 105]


@engine.step(description="Enrich each user with profile data")
def enrich_user(user_id: int):
    """Simulates fetching profile data for a user."""
    time.sleep(0.05)
    return {
        "id": user_id,
        "name": f"User_{user_id}",
        "score": user_id * 7 % 100,
        "active": user_id % 2 == 0,
    }


@engine.step(description="Aggregate statistics from enriched users")
def compute_stats(users: list) -> dict:
    """Compute aggregate statistics."""
    scores = [u["score"] for u in users]
    active = sum(1 for u in users if u["active"])
    return {
        "total_users": len(users),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "active_users": active,
        "inactive_users": len(users) - active,
    }


@engine.step(description="Generate a summary report")
def generate_report(stats: dict) -> str:
    """Generate a human-readable report."""
    return f"""
╔══════════════════════════════╗
║     DATA PIPELINE REPORT    ║
╠══════════════════════════════╣
║ Total Users:    {stats['total_users']:>11} ║
║ Avg Score:      {stats['avg_score']:>11.1f} ║
║ Max Score:      {stats['max_score']:>11} ║
║ Active Users:   {stats['active_users']:>11} ║
║ Inactive Users: {stats['inactive_users']:>11} ║
╚══════════════════════════════╝
"""


# === Build and run the workflow ===

workflow = Workflow(
    "user_data_pipeline",
    description="Fetch, enrich, and report on user data — with durable execution",
)

workflow.add_step(fetch_users)
workflow.add_step(enrich_user, depends_on=["fetch_users"], fan_out=True)
workflow.add_step(compute_stats, depends_on=["enrich_user"])
workflow.add_step(generate_report, depends_on=["compute_stats"])


def run_demo():
    print("=" * 60)
    print("  Durable Mini — Data Pipeline Example")
    print("=" * 60)

    # First run: normal execution
    print("\n[Run 1] Normal execution...")
    result = engine.run(workflow, instance_id="demo-pipeline-001")
    print(f"  Status: {result.status}")
    print(f"  Duration: {result.duration_seconds:.3f}s")
    print(f"  Steps: {result.steps_completed}/{result.steps_total}")

    if result.status == "completed":
        print(result.output["generate_report"])

    # Show step-by-step status
    print("\n[Step Status]")
    status = engine.status("demo-pipeline-001")
    for step in status["steps"]:
        icon = "✅" if step["status"] == "completed" else "❌"
        print(f"  {icon} {step['name']}: {step['status']} (attempt {step['attempt']})")

    # Second run: resume (all steps already completed — should be rejected)
    print("\n[Run 2] Resume attempt (all completed — should reject)...")
    try:
        result2 = engine.run(workflow, instance_id="demo-pipeline-001")
        print(f"  Status: {result2.status}")
    except ValueError as e:
        print(f"  Rejected: {e}")
        print("  ✅ Correct: completed workflows are immutable.")

    # Third run: new instance to show idempotent behavior
    print("\n[Run 3] New instance, same workflow...")
    result3 = engine.run(workflow, instance_id="demo-pipeline-002")
    print(f"  Status: {result3.status}")
    print(f"  Duration: {result3.duration_seconds:.3f}s")

    print("\n✨ Durable execution: completed steps are never re-run.")

    engine.close()
    os.remove(DB_PATH)


if __name__ == "__main__":
    run_demo()
