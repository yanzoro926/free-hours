"""
Test suite for Durable Mini.

Covers:
- Basic workflow execution
- Crash recovery and resume
- Fan-out (parallel) steps
- Idempotent replay
- DAG validation (cycle detection, missing deps)
- Visualization
"""

import os
import sys
import json
import time
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from durable_mini import DurableEngine, Workflow
from durable_mini.backend import DurableBackend


class TestBackend(unittest.TestCase):
    """SQLite backend tests."""

    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.backend = DurableBackend(self.db_path)

    def tearDown(self):
        self.backend.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_register_and_get_workflow(self):
        self.backend.register_workflow("test_wf", {"steps": {"a": {}}})
        wf = self.backend.get_workflow("test_wf")
        self.assertEqual(wf["steps"]["a"], {})

    def test_create_instance(self):
        self.backend.register_workflow("test_wf", {"steps": {}})
        iid = self.backend.create_instance("inst-1", "test_wf", {"key": "val"})
        inst = self.backend.get_instance("inst-1")
        self.assertEqual(inst["workflow_name"], "test_wf")
        self.assertEqual(inst["status"], "pending")
        self.assertEqual(json.loads(inst["input_data"]), {"key": "val"})

    def test_step_lifecycle(self):
        self.backend.register_workflow("test_wf", {"steps": {}})
        self.backend.create_instance("inst-1", "test_wf")

        # Start
        self.backend.record_step_start("inst-1", "step_a", 1, {"x": 1})
        step = self.backend.get_step("inst-1", "step_a")
        self.assertEqual(step.status, "running")
        self.assertEqual(step.attempt, 1)
        self.assertEqual(step.input_data, {"x": 1})

        # Complete
        self.backend.record_step_complete("inst-1", "step_a", {"result": 42})
        step = self.backend.get_step("inst-1", "step_a")
        self.assertEqual(step.status, "completed")
        self.assertEqual(step.output_data, {"result": 42})

    def test_step_failure(self):
        self.backend.register_workflow("test_wf", {"steps": {}})
        self.backend.create_instance("inst-1", "test_wf")
        self.backend.record_step_start("inst-1", "step_a", 1, None)
        self.backend.record_step_failed("inst-1", "step_a", "BOOM")
        step = self.backend.get_step("inst-1", "step_a")
        self.assertEqual(step.status, "failed")
        self.assertIn("BOOM", step.error_message)

    def test_stats(self):
        self.backend.register_workflow("test_wf", {"steps": {}})
        self.backend.create_instance("inst-1", "test_wf")
        self.backend.create_instance("inst-2", "test_wf")
        self.backend.update_instance_status("inst-1", "completed")
        self.backend.update_instance_status("inst-2", "failed")
        stats = self.backend.get_stats()
        self.assertEqual(stats["total_instances"], 2)
        self.assertEqual(stats["completed"], 1)
        self.assertEqual(stats["failed"], 1)


class TestWorkflow(unittest.TestCase):
    """Workflow DAG tests."""

    def test_simple_linear(self):
        def a():
            pass

        def b():
            pass

        def c():
            pass

        wf = Workflow("linear")
        wf.add_step(a)
        wf.add_step(b, depends_on=["a"])
        wf.add_step(c, depends_on=["b"])
        order = wf.topological_order()
        self.assertEqual(order, ["a", "b", "c"])

    def test_diamond_dag(self):
        def a():
            pass

        def b():
            pass

        def c():
            pass

        def d():
            pass

        wf = Workflow("diamond")
        wf.add_step(a)
        wf.add_step(b, depends_on=["a"])
        wf.add_step(c, depends_on=["a"])
        wf.add_step(d, depends_on=["b", "c"])
        order = wf.topological_order()
        self.assertEqual(order[0], "a")
        self.assertEqual(order[3], "d")
        self.assertIn("b", order[1:3])
        self.assertIn("c", order[1:3])

    def test_cycle_detection(self):
        def a():
            pass

        def b():
            pass

        wf = Workflow("cycle")
        wf.add_step(a, depends_on=["b"])
        wf.add_step(b, depends_on=["a"])
        with self.assertRaises(ValueError):
            wf.topological_order()

    def test_missing_dependency(self):
        def a():
            pass

        wf = Workflow("missing")
        wf.add_step(a, depends_on=["nonexistent"])
        with self.assertRaises(ValueError):
            wf.validate()

    def test_duplicate_step(self):
        def a():
            pass

        wf = Workflow("dup")
        wf.add_step(a)
        with self.assertRaises(ValueError):
            wf.add_step(a)

    def test_fanout_requires_dep(self):
        def a():
            pass

        wf = Workflow("fanout_no_dep")
        wf.add_step(a, fan_out=True)
        with self.assertRaises(ValueError):
            wf.validate()

    def test_to_dict(self):
        def a():
            pass

        def b():
            pass

        wf = Workflow("test", "desc")
        wf.add_step(a).add_step(b, depends_on=["a"], fan_out=True)
        d = wf.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["description"], "desc")
        self.assertTrue(d["steps"]["b"]["fan_out"])


class TestEngine(unittest.TestCase):
    """Durable execution engine tests."""

    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.engine = DurableEngine(self.db_path)

    def tearDown(self):
        self.engine.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_basic_execution(self):
        @self.engine.step()
        def step_a():
            return 42

        @self.engine.step()
        def step_b(val):
            return val * 2

        wf = Workflow("basic")
        wf.add_step(step_a)
        wf.add_step(step_b, depends_on=["step_a"])
        result = self.engine.run(wf)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output["step_b"], 84)

    def test_fan_out_execution(self):
        @self.engine.step()
        def source():
            return [1, 2, 3]

        @self.engine.step()
        def double(item: int):
            return item * 2

        @self.engine.step()
        def total(results):
            return sum(results)

        wf = Workflow("fanout")
        wf.add_step(source)
        wf.add_step(double, depends_on=["source"], fan_out=True)
        wf.add_step(total, depends_on=["double"])
        result = self.engine.run(wf)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output["total"], 12)

    def test_crash_recovery(self):
        call_count = {"b": 0}

        @self.engine.step()
        def step_a():
            return "data"

        @self.engine.step()
        def step_b(data):
            call_count["b"] += 1
            if call_count["b"] == 1:
                raise RuntimeError("Simulated crash")
            return f"processed:{data}"

        @self.engine.step()
        def step_c(processed):
            return f"final:{processed}"

        wf = Workflow("crash")
        wf.add_step(step_a)
        wf.add_step(step_b, depends_on=["step_a"])
        wf.add_step(step_c, depends_on=["step_b"])

        # First run fails
        result1 = self.engine.run(wf)
        self.assertEqual(result1.status, "failed")

        # Resume (same instance)
        result2 = self.engine.run(wf, instance_id=result1.instance_id)
        self.assertEqual(result2.status, "completed")
        self.assertEqual(result2.output["step_c"], "final:processed:data")

        # step_b was called twice (first failed, second succeeded)
        # step_a and step_c were called once each
        self.assertEqual(call_count["b"], 2)

        # Verify step_a only ran once
        status = self.engine.status(result1.instance_id)
        step_a_rec = next(s for s in status["steps"] if s["name"] == "step_a")
        self.assertEqual(step_a_rec["attempt"], 1)

    def test_idempotent_replay(self):
        """Re-running a completed instance should be rejected."""
        @self.engine.step()
        def step_a():
            return 1

        wf = Workflow("idempotent")
        wf.add_step(step_a)
        result = self.engine.run(wf, instance_id="fixed-id")
        self.assertEqual(result.status, "completed")

        with self.assertRaises(ValueError):
            self.engine.run(wf, instance_id="fixed-id")

    def test_failed_instance_resume(self):
        """A failed instance can be resumed."""
        call_count = {"a": 0}

        @self.engine.step()
        def flaky():
            call_count["a"] += 1
            if call_count["a"] == 1:
                raise RuntimeError("Fail!")
            return "ok"

        wf = Workflow("flaky")
        wf.add_step(flaky)

        result1 = self.engine.run(wf, instance_id="flaky-1")
        self.assertEqual(result1.status, "failed")

        result2 = self.engine.run(wf, instance_id="flaky-1")
        self.assertEqual(result2.status, "completed")
        self.assertEqual(call_count["a"], 2)

    def test_multiple_inputs(self):
        @self.engine.step()
        def step_a():
            return "hello"

        @self.engine.step()
        def step_b():
            return "world"

        @self.engine.step()
        def combine(step_a, step_b):
            return f"{step_a} {step_b}"

        wf = Workflow("multi_input")
        wf.add_step(step_a)
        wf.add_step(step_b)
        wf.add_step(combine, depends_on=["step_a", "step_b"])
        result = self.engine.run(wf)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output["combine"], "hello world")

    def test_ctx_argument(self):
        """Steps can accept an ExecutionContext for metadata."""

        @self.engine.step()
        def with_ctx(ctx):
            return ctx.instance_id

        wf = Workflow("ctx_test")
        wf.add_step(with_ctx)
        result = self.engine.run(wf)
        self.assertIsNotNone(result.output["with_ctx"])
        self.assertTrue(len(result.output["with_ctx"]) > 0)

    def test_fanout_crash_recovery(self):
        """Fan-out steps should also be recoverable."""
        call_count = {}

        @self.engine.step()
        def source():
            return [10, 20, 30]

        @self.engine.step()
        def process(item):
            key = f"process_{item}"
            call_count[key] = call_count.get(key, 0) + 1
            if call_count[key] == 1 and item == 20:
                raise RuntimeError("Simulated crash on item 20")
            return item * 10

        @self.engine.step()
        def collect(results):
            return sum(results)

        wf = Workflow("fanout_crash")
        wf.add_step(source)
        wf.add_step(process, depends_on=["source"], fan_out=True)
        wf.add_step(collect, depends_on=["process"])

        # Run 1: fails at process[1] (item=20)
        result1 = self.engine.run(wf)
        self.assertEqual(result1.status, "failed")

        # Run 2: resume — process[0] (item=10) should be skipped
        result2 = self.engine.run(wf, instance_id=result1.instance_id)
        self.assertEqual(result2.status, "completed")
        self.assertEqual(result2.output["collect"], 600)

        # process[0] ran only once (first attempt)
        self.assertEqual(call_count.get("process_10", 0), 1)
        # process[1] ran twice (failed + succeeded)
        self.assertEqual(call_count.get("process_20", 0), 2)
        # process[2] ran once
        self.assertEqual(call_count.get("process_30", 0), 1)

    def test_status(self):
        @self.engine.step()
        def step_a():
            return 1

        wf = Workflow("status_test")
        wf.add_step(step_a)

        result = self.engine.run(wf, instance_id="status-1")
        status = self.engine.status("status-1")
        self.assertEqual(status["instance_id"], "status-1")
        self.assertEqual(status["status"], "completed")
        self.assertEqual(len(status["steps"]), 1)
        self.assertEqual(status["steps"][0]["status"], "completed")

    def test_nonexistent_instance(self):
        status = self.engine.status("nonexistent")
        self.assertIn("error", status)


class TestVisualizer(unittest.TestCase):
    """Visualization tests."""

    def setUp(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.engine = DurableEngine(self.db_path)

    def tearDown(self):
        self.engine.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_dag_output(self):
        from visualize import WorkflowVisualizer

        @self.engine.step()
        def step_a():
            return [1, 2]

        @self.engine.step()
        def step_b(item):
            return item * 2

        wf = Workflow("viz_test")
        wf.add_step(step_a)
        wf.add_step(step_b, depends_on=["step_a"], fan_out=True)

        result = self.engine.run(wf, instance_id="viz-1")

        viz = WorkflowVisualizer(self.db_path)
        dag_output = viz.dag("viz-1")
        self.assertIn("viz_test", dag_output)
        self.assertIn("✅", dag_output)
        self.assertIn("step_a", dag_output)
        self.assertIn("step_b", dag_output)
        viz.close()

    def test_html_output(self):
        from visualize import WorkflowVisualizer

        @self.engine.step()
        def step_a():
            return 1

        wf = Workflow("html_test")
        wf.add_step(step_a)
        self.engine.run(wf, instance_id="html-1")

        viz = WorkflowVisualizer(self.db_path)
        html = viz.html("html-1")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("html_test", html)
        self.assertIn("step_a", html)
        viz.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
