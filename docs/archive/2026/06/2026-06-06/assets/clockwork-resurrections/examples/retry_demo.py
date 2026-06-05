"""
Example 4: Retry with Exponential Backoff.

Demonstrates automatic retry of failed steps within a single engine.run() call.
No manual resume needed — the engine retries with exponential backoff.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from durable_mini import DurableEngine, Workflow
from durable_mini.retry import RetryPolicy, FAST_RETRY

DB_PATH = "/tmp/durable_retry_demo.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

engine = DurableEngine(DB_PATH)

# Track attempts across retries
attempt_log: list[tuple[str, int]] = []


@engine.step(description="Fetch data (always succeeds)")
def fetch_data():
    return {"user_id": 42, "items": [1, 2, 3]}


@engine.step(description="Call flaky external API (fails first 2 times)")
def call_api(data: dict) -> dict:
    """Simulates a flaky API that fails the first 2 calls."""
    attempt_log.append(("call_api", len(attempt_log) + 1))

    # Fail on first 2 attempts, succeed on 3rd
    call_count = sum(1 for name, _ in attempt_log if name == "call_api")
    if call_count <= 2:
        raise ConnectionError(f"API unavailable (attempt {call_count})")

    return {"status": "ok", "user": data["user_id"], "processed": len(data["items"])}


@engine.step(description="Save result to database")
def save_result(result: dict) -> str:
    return f"Saved: user={result['user']}, items={result['processed']}"


# Build workflow — call_api has retries=3 (tries up to 4 times total)
workflow = Workflow(
    "retry_pipeline",
    description="Flaky API call with automatic retry",
)
workflow.add_step(fetch_data)
workflow.add_step(call_api, depends_on=["fetch_data"], retries=3)
workflow.add_step(save_result, depends_on=["call_api"])


def run_demo():
    print("=" * 60)
    print("  Durable Mini — Retry with Backoff")
    print("=" * 60)
    print()
    print("Scenario: call_api fails on first 2 attempts,")
    print("but succeeds on the 3rd. With retries=3, the engine")
    print("automatically retries without manual intervention.")
    print()

    start = time.time()
    result = engine.run(workflow, instance_id="retry-demo-001")
    elapsed = time.time() - start

    print(f"Status: {result.status}")
    print(f"Duration: {elapsed:.2f}s")
    print()

    if result.status == "completed":
        print(f"Final output: {result.output['save_result']}")

    print()
    print("Attempt log:")
    for name, n in attempt_log:
        print(f"  [{n}] {name} executed")

    print()
    status = engine.status("retry-demo-001")
    print("Step status:")
    for step in status["steps"]:
        icon = "✅" if step["status"] == "completed" else "❌"
        print(f"  {icon} {step['name']}: {step['status']} (attempt {step['attempt']})")

    print()
    print("💡 The engine retried call_api 3 times automatically.")
    print("   fetch_data and save_result each executed exactly once.")

    engine.close()
    os.remove(DB_PATH)


if __name__ == "__main__":
    run_demo()
