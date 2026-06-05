"""
Workflow DAG definition — composable graph of durable steps.
"""

from typing import Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class StepDef:
    """Definition of a single step in a workflow."""

    name: str
    func: Callable
    depends_on: list[str] = field(default_factory=list)
    fan_out: bool = False  # If True, run once per item in input list
    retries: int = 0
    timeout_seconds: Optional[float] = None
    description: str = ""


class Workflow:
    """A directed acyclic graph of durable steps."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: dict[str, StepDef] = {}
        self._order: list[str] = []

    def add_step(
        self,
        func: Callable,
        depends_on: Optional[list[str]] = None,
        fan_out: bool = False,
        retries: int = 0,
        timeout_seconds: Optional[float] = None,
        description: str = "",
    ) -> "Workflow":
        """Add a step to the workflow. Returns self for chaining."""
        name = func.__name__
        if name in self.steps:
            raise ValueError(f"Step '{name}' already exists in workflow")

        self.steps[name] = StepDef(
            name=name,
            func=func,
            depends_on=depends_on or [],
            fan_out=fan_out,
            retries=retries,
            timeout_seconds=timeout_seconds,
            description=description or (func.__doc__ or "").strip(),
        )
        self._order.append(name)
        return self

    def topological_order(self) -> list[str]:
        """Return steps in dependency-respecting topological order."""
        UNVISITED, VISITING, VISITED = 0, 1, 2
        state: dict[str, int] = {name: UNVISITED for name in self.steps}
        result: list[str] = []

        def visit(name: str):
            if state[name] == VISITED:
                return
            if state[name] == VISITING:
                raise ValueError(f"Workflow contains a cycle involving step '{name}'")
            state[name] = VISITING
            for dep in self.steps[name].depends_on:
                if dep not in self.steps:
                    raise ValueError(
                        f"Step '{name}' depends on unknown step '{dep}'"
                    )
                visit(dep)
            state[name] = VISITED
            result.append(name)

        for name in self._order:
            visit(name)

        return result

    def to_dict(self) -> dict:
        """Serialize workflow definition for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": {
                name: {
                    "name": s.name,
                    "depends_on": s.depends_on,
                    "fan_out": s.fan_out,
                    "retries": s.retries,
                    "timeout_seconds": s.timeout_seconds,
                    "description": s.description,
                }
                for name, s in self.steps.items()
            },
            "order": self._order,
        }

    def validate(self):
        """Validate the workflow DAG."""
        # Check all deps exist
        for name, step in self.steps.items():
            for dep in step.depends_on:
                if dep not in self.steps:
                    raise ValueError(
                        f"Step '{name}' depends on unknown step '{dep}'"
                    )

        # Check no cycles via topological sort
        self.topological_order()

        # Fan-out steps must depend on exactly one step
        for name, step in self.steps.items():
            if step.fan_out and len(step.depends_on) != 1:
                raise ValueError(
                    f"Fan-out step '{name}' must depend on exactly one step"
                )

        return True
