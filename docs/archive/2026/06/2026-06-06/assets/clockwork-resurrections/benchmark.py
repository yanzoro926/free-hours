#!/usr/bin/env python3
"""
Benchmark: Durable execution overhead.

Measures the performance cost of checkpointing every step to SQLite
versus running the same logic without durability.
"""

import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from durable_mini import DurableEngine, Workflow


def create_workflow(engine: DurableEngine, depth: int, fan_out: int = 0) -> Workflow:
    """Create a workflow with `depth` sequential steps and optional fan-out."""

    def make_step(i):
        if i == 0:
            @engine.step(description="Root step")
            def root_step():
                _ = sum(range(1000))
                return 1
            root_step.__name__ = f"step_0"
            return root_step
        else:
            @engine.step(description=f"Step {i}")
            def step(val):
                _ = sum(range(1000))
                if isinstance(val, list):
                    return [v * 2 for v in val]
                return val * 2 if isinstance(val, (int, float)) else val
            step.__name__ = f"step_{i}"
            return step

    wf = Workflow(f"bench_depth_{depth}")

    # Root step
    root = make_step(0)
    wf.add_step(root)

    for i in range(1, depth):
        prev = f"step_{i - 1}"

        if fan_out > 0 and i == depth // 2:
            # Fan-out in the middle
            fo_step = make_step(i)
            wf.add_step(fo_step, depends_on=[prev], fan_out=True)
            # Generate list input
            @engine.step()
            def gen_list():
                return list(range(fan_out))

            gen_list.__name__ = f"gen_list_{i}"
            # Actually, need a source for fan_out
            # For simplicity, skip fan_out in benchmark
            step = make_step(i)
            wf.add_step(step, depends_on=[prev])
        else:
            step = make_step(i)
            wf.add_step(step, depends_on=[prev])

    return wf


def benchmark_durable(depth: int, iterations: int = 3) -> dict:
    """Benchmark durable execution with SQLite checkpointing."""
    times = []
    for _ in range(iterations):
        db_path = tempfile.mktemp(suffix=".db")
        engine = DurableEngine(db_path)
        wf = create_workflow(engine, depth)

        start = time.perf_counter()
        result = engine.run(wf)
        elapsed = time.perf_counter() - start

        assert result.status == "completed", f"Failed: {result.error}"
        times.append(elapsed)
        engine.close()
        os.remove(db_path)

    return {
        "depth": depth,
        "iterations": iterations,
        "times": times,
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }


def benchmark_non_durable(depth: int, iterations: int = 3) -> dict:
    """Benchmark without any durability (raw Python function calls)."""
    times = []

    # Build equivalent non-durable pipeline
    def run_pipeline(val):
        for _ in range(depth):
            _ = sum(range(1000))
            val = val * 2 if isinstance(val, (int, float)) else val
        return val

    for _ in range(iterations):
        start = time.perf_counter()
        run_pipeline(1)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "depth": depth,
        "iterations": iterations,
        "times": times,
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }


def main():
    print("=" * 65)
    print("  Durable Mini — Overhead Benchmark")
    print("=" * 65)
    print()
    print(f"{'Depth':>6}  {'Durable (ms)':>14}  {'Raw (ms)':>14}  {'Overhead':>10}")
    print("-" * 55)

    for depth in [5, 10, 20, 50, 100]:
        d = benchmark_durable(depth, iterations=5)
        nd = benchmark_non_durable(depth, iterations=5)

        overhead = (d["mean"] - nd["mean"]) / nd["mean"] * 100
        print(
            f"{depth:>6}  {d['mean']*1000:>12.3f}  "
            f"{nd['mean']*1000:>12.3f}  {overhead:>8.1f}%"
        )

    # Detailed analysis at depth=20
    print()
    print("=" * 65)
    print("  Detailed Analysis (depth=20, 10 iterations)")
    print("=" * 65)

    d20 = benchmark_durable(20, iterations=10)
    nd20 = benchmark_non_durable(20, iterations=10)

    print(f"\nDurable:")
    print(f"  Mean:   {d20['mean']*1000:.3f} ms")
    print(f"  Min:    {d20['min']*1000:.3f} ms")
    print(f"  Max:    {d20['max']*1000:.3f} ms")
    print(f"  StdDev: {((sum((t-d20['mean'])**2 for t in d20['times'])/len(d20['times']))**0.5)*1000:.3f} ms")

    print(f"\nNon-Durable:")
    print(f"  Mean:   {nd20['mean']*1000:.3f} ms")
    print(f"  Min:    {nd20['min']*1000:.3f} ms")
    print(f"  Max:    {nd20['max']*1000:.3f} ms")

    overhead_per_step = (d20["mean"] - nd20["mean"]) / 20 * 1000
    print(f"\nOverhead per step: {overhead_per_step:.3f} ms")
    print(f"Total overhead:     {(d20['mean'] - nd20['mean'])*1000:.3f} ms")
    print(
        f"Overhead ratio:     {(d20['mean'] / nd20['mean'] - 1) * 100:.1f}%"
    )

    print()
    print("💡 The overhead comes from SQLite INSERT/UPDATE per step.")
    print("   For I/O-bound steps (API calls, DB queries), this is negligible.")
    print("   For CPU-bound steps, use fewer, larger steps to amortize cost.")


if __name__ == "__main__":
    main()
