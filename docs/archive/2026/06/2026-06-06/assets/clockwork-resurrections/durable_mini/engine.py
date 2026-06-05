"""
Core durable execution engine.

The engine orchestrates workflow execution with checkpointing:
1. Before running a step, check if it already completed (DB lookup)
2. If completed: skip and retrieve stored output
3. If not: execute, store result in DB
4. On crash: resume from last incomplete step

Key insight: Each step is idempotent from the engine's perspective
because inputs and outputs are deterministically checkpointed.
"""

import time
import uuid
import traceback
from typing import Any, Optional, Callable
from dataclasses import dataclass, field

from .backend import DurableBackend, StepRecord
from .workflow import Workflow, StepDef


@dataclass
class ExecutionContext:
    """Context passed to each step during execution."""

    instance_id: str
    workflow_name: str
    step_name: str
    attempt: int
    engine: "DurableEngine"


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    instance_id: str
    workflow_name: str
    status: str
    output: Any = None
    error: Optional[str] = None
    steps_completed: int = 0
    steps_total: int = 0
    duration_seconds: float = 0.0


class DurableEngine:
    """
    The durable execution engine.

    Example:
        engine = DurableEngine("my_workflows.db")

        @engine.step()
        def fetch():
            return [1, 2, 3]

        @engine.step()
        def process(item):
            return item * 2

        wf = Workflow("pipeline")
        wf.add_step(fetch).add_step(process, depends_on=["fetch"], fan_out=True)

        result = engine.run(wf)
    """

    def __init__(self, db_path: str = "durable.db"):
        self.backend = DurableBackend(db_path)
        self._pending_callbacks: dict[
            str, Callable
        ] = {}  # on_complete callbacks per instance

    def run(
        self,
        workflow: Workflow,
        input_data: Any = None,
        instance_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Execute a workflow durably.

        If instance_id is provided and the instance already exists,
        this is treated as a resume — completed steps are skipped.
        """
        workflow.validate()

        if instance_id is None:
            instance_id = str(uuid.uuid4())[:12]

        # Register workflow if new
        existing = self.backend.get_workflow(workflow.name)
        if existing is None:
            self.backend.register_workflow(workflow.name, workflow.to_dict())

        # Check existing instance
        existing_instance = self.backend.get_instance(instance_id)
        is_resume = existing_instance is not None

        if not is_resume:
            self.backend.create_instance(instance_id, workflow.name, input_data)
            self.backend.update_instance_status(
                instance_id, "running", started_at=time.time()
            )
        elif existing_instance["status"] == "completed":
            raise ValueError(
                f"Instance {instance_id} already completed"
            )
        else:
            # Resume: mark as running again
            self.backend.update_instance_status(
                instance_id, "running", started_at=time.time()
            )

        start_time = time.time()
        topological_order = workflow.topological_order()

        try:
            step_outputs: dict[str, Any] = {"__input__": input_data}

            for step_name in topological_order:
                step_def = workflow.steps[step_name]

                if step_def.fan_out:
                    self._execute_fanout_step(
                        workflow, step_def, step_outputs, instance_id
                    )
                else:
                    output = self._execute_step(
                        workflow, step_def, step_outputs, instance_id
                    )
                    step_outputs[step_name] = output

            # All steps completed
            duration = time.time() - start_time
            self.backend.update_instance_status(
                instance_id, "completed", output_data=step_outputs, completed_at=time.time()
            )

            return WorkflowResult(
                instance_id=instance_id,
                workflow_name=workflow.name,
                status="completed",
                output=step_outputs,
                steps_completed=len(topological_order),
                steps_total=len(topological_order),
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            self.backend.update_instance_status(
                instance_id, "failed", completed_at=time.time()
            )
            return WorkflowResult(
                instance_id=instance_id,
                workflow_name=workflow.name,
                status="failed",
                error=str(e),
                duration_seconds=duration,
            )

    def _execute_step(
        self,
        workflow: Workflow,
        step_def: StepDef,
        step_outputs: dict[str, Any],
        instance_id: str,
    ) -> Any:
        """Execute a single step with checkpointing and retries."""
        # Check if already completed (crash recovery)
        existing = self.backend.get_step(instance_id, step_def.name)
        if existing and existing.status == "completed":
            return existing.output_data

        # Prepare input from dependencies
        input_data = self._resolve_inputs(step_def, step_outputs)

        max_attempts = step_def.retries + 1
        last_error = None
        base_attempt = existing.attempt if existing else 0

        for attempt_num in range(1, max_attempts + 1):
            db_attempt = base_attempt + attempt_num
            self.backend.record_step_start(
                instance_id, step_def.name, db_attempt, input_data
            )

            try:
                ctx = ExecutionContext(
                    instance_id=instance_id,
                    workflow_name=workflow.name,
                    step_name=step_def.name,
                    attempt=db_attempt,
                    engine=self,
                )

                output = self._call_step(step_def.func, input_data, ctx)
                self.backend.record_step_complete(
                    instance_id, step_def.name, output
                )
                return output

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                self.backend.record_step_failed(
                    instance_id, step_def.name, error_msg
                )
                last_error = e

                if attempt_num < max_attempts:
                    # Exponential backoff
                    delay = min(1.0 * (2 ** (attempt_num - 1)), 30.0)
                    time.sleep(delay)
                else:
                    raise last_error

        raise last_error  # type: ignore

    def _execute_fanout_step(
        self,
        workflow: Workflow,
        step_def: StepDef,
        step_outputs: dict[str, Any],
        instance_id: str,
    ):
        """Execute a fan-out step: once per item in the dependency's output list."""
        if not step_def.depends_on:
            raise ValueError(f"Fan-out step '{step_def.name}' has no dependencies")

        parent_output = step_outputs.get(step_def.depends_on[0])
        if not isinstance(parent_output, list):
            raise ValueError(
                f"Fan-out step '{step_def.name}' requires list input, got {type(parent_output)}"
            )

        results = []
        for i, item in enumerate(parent_output):
            sub_step_name = f"{step_def.name}[{i}]"
            # Check if this sub-step already completed
            existing = self.backend.get_step(instance_id, sub_step_name)
            if existing and existing.status == "completed":
                results.append(existing.output_data)
                continue

            attempt = (existing.attempt + 1) if existing else 1
            self.backend.record_step_start(
                instance_id, sub_step_name, attempt, item
            )

            try:
                ctx = ExecutionContext(
                    instance_id=instance_id,
                    workflow_name=workflow.name,
                    step_name=sub_step_name,
                    attempt=attempt,
                    engine=self,
                )
                output = self._call_step(step_def.func, item, ctx)
                self.backend.record_step_complete(instance_id, sub_step_name, output)
                results.append(output)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                self.backend.record_step_failed(
                    instance_id, sub_step_name, error_msg
                )
                raise

        step_outputs[step_def.name] = results

    def _resolve_inputs(
        self, step_def: StepDef, step_outputs: dict[str, Any]
    ) -> Any:
        """Resolve step inputs from dependency outputs."""
        if not step_def.depends_on:
            return step_outputs.get("__input__", {})

        if len(step_def.depends_on) == 1:
            return step_outputs.get(step_def.depends_on[0])

        # Multiple dependencies → return dict
        return {dep: step_outputs.get(dep) for dep in step_def.depends_on}

    def _call_step(
        self, func: Callable, input_data: Any, ctx: ExecutionContext
    ) -> Any:
        """Call a step function with smart argument matching."""
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Build kwargs from input_data
        kwargs: dict[str, Any] = {}

        if isinstance(input_data, dict):
            kwargs = {k: v for k, v in input_data.items() if k in params}
            # If no kwargs matched but function takes a single arg (not ctx),
            # pass the entire dict as that arg
            if not kwargs and len(params) == 1 and params[0] != "ctx":
                kwargs[params[0]] = input_data
            elif not kwargs and len(params) == 2 and "ctx" in params:
                other = [p for p in params if p != "ctx"][0]
                kwargs[other] = input_data
            elif not kwargs:
                # Multi-dep: try positional matching by values in order
                non_ctx_params = [p for p in params if p != "ctx"]
                values = list(input_data.values())
                if len(non_ctx_params) == len(values):
                    for p, v in zip(non_ctx_params, values):
                        kwargs[p] = v
        elif input_data is not None:
            # Single non-dict input → pass to first non-ctx param
            for p in params:
                if p != "ctx":
                    kwargs[p] = input_data
                    break

        # Add ctx if the function accepts it
        if "ctx" in params:
            kwargs["ctx"] = ctx

        # Call with matching args
        if kwargs:
            return func(**kwargs)
        else:
            # No args needed
            return func()

    def status(self, instance_id: str) -> dict:
        """Get the status of a workflow instance."""
        instance = self.backend.get_instance(instance_id)
        if not instance:
            return {"error": f"Instance '{instance_id}' not found"}

        steps = self.backend.get_all_steps(instance_id)
        return {
            "instance_id": instance["id"],
            "workflow": instance["workflow_name"],
            "status": instance["status"],
            "created_at": instance["created_at"],
            "started_at": instance["started_at"],
            "completed_at": instance["completed_at"],
            "steps": [
                {
                    "name": s.step_name,
                    "status": s.status,
                    "attempt": s.attempt,
                    "started_at": s.started_at,
                    "completed_at": s.completed_at,
                }
                for s in steps
            ],
        }

    def step(self, **kwargs):
        """
        Convenience decorator. Same as @step from decorators module,
        but scoped to this engine.
        """
        from .decorators import step as _step

        return _step(**kwargs)

    def close(self):
        self.backend.close()
