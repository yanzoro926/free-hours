"""
Crash Recovery Demo — The core value proposition of durable execution.

Demonstrates:
1. A workflow that fails mid-execution (simulated crash)
2. The database retains all completed step outputs
3. Resuming the workflow skips completed steps
4. The final result is identical to an uninterrupted run
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from durable_mini import DurableEngine, Workflow

DB_PATH = "/tmp/durable_crash_demo.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

engine = DurableEngine(DB_PATH)

# Track execution to prove steps aren't re-run
execution_log: list[str] = []


@engine.step(description="Load raw data from source")
def load_data():
    execution_log.append("load_data executed")
    return list(range(1, 11))  # [1, 2, ..., 10]


@engine.step(description="Validate each data item")
def validate(item: int) -> dict:
    execution_log.append(f"validate({item}) executed")
    time.sleep(0.01)
    return {"value": item, "valid": item != 7, "squared": item**2}


@engine.step(description="Filter out invalid items")
def filter_invalid(results: list[dict]) -> list[dict]:
    execution_log.append("filter_invalid executed")
    return [r for r in results if r["valid"]]


@engine.step(description="Compute aggregate statistics")
def compute_aggregates(filtered: list[dict]) -> dict:
    execution_log.append("compute_aggregates executed")
    values = [r["value"] for r in filtered]
    squares = [r["squared"] for r in filtered]
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values),
        "sum_squares": sum(squares),
        "items": values,
    }


@engine.step(description="Generate final report (FRAGILE — crashes 50% of the time)")
def generate_report(stats: dict) -> str:
    execution_log.append("generate_report executed")
    # Simulate transient failure
    if random.random() < 0.5:
        raise RuntimeError("💥 Transient error in report generation!")
    return f"Report: {stats['count']} items, sum={stats['sum']}, mean={stats['mean']:.1f}"


# Build workflow
workflow = Workflow(
    "crash_recovery_demo",
    description="Demonstrates crash recovery with transient failures",
)
workflow.add_step(load_data)
workflow.add_step(validate, depends_on=["load_data"], fan_out=True)
workflow.add_step(filter_invalid, depends_on=["validate"])
workflow.add_step(compute_aggregates, depends_on=["filter_invalid"])
workflow.add_step(generate_report, depends_on=["compute_aggregates"])


def run_demo():
    print("=" * 60)
    print("  Durable Mini — Crash Recovery Demo")
    print("=" * 60)
    print()
    print("Scenario: A 5-step data pipeline where the last step")
    print("crashes randomly. With durable execution, we simply")
    print("resume — the first 4 steps are NEVER re-executed.")
    print()

    instance_id = "crash-demo-001"

    # Keep trying until it succeeds (simulating crash + retry)
    attempt = 0
    while True:
        attempt += 1
        execution_log.clear()
        print(f"[Attempt {attempt}] ", end="", flush=True)

        result = engine.run(workflow, instance_id=instance_id)

        if result.status == "completed":
            print(f"✅ SUCCESS after {attempt} attempt(s)")
            print(f"  Duration: {result.duration_seconds:.3f}s")
            print(f"  Result: {result.output['generate_report']}")
            break
        else:
            print(f"❌ FAILED: {result.error.split(chr(10))[0]}")
            print(f"  Resuming... (completed steps will be skipped)")

    print()
    print("═" * 40)
    print("EXECUTION AUDIT")
    print("═" * 40)
    for entry in execution_log:
        print(f"  → {entry}")

    # Count how many times each step actually ran
    from collections import Counter

    step_counts = Counter(
        entry.split("(")[0] if "(" in entry else entry.split()[0]
        for entry in execution_log
    )
    print()
    print("📊 Steps executed in final attempt:")
    for step_name in ["load_data", "validate", "filter_invalid", "compute_aggregates", "generate_report"]:
        count = step_counts.get(step_name, 0)
        print(f"  {step_name}: {count} time(s)")

    # Show database state
    print()
    print("═" * 40)
    print("DATABASE STATE")
    print("═" * 40)
    status = engine.status(instance_id)
    for step in status["steps"]:
        icon = "✅" if step["status"] == "completed" else "🔄"
        print(f"  {icon} {step['name']}: {step['status']} (attempt {step['attempt']})")

    print()
    print("💡 Durable execution: crash → resume → complete.")
    print("   No step is ever re-executed after succeeding.")

    engine.close()
    os.remove(DB_PATH)


if __name__ == "__main__":
    run_demo()
