"""
Durable Mini — A minimal durable execution engine in Python.

Inspired by Microsoft's pg_durable (open-sourced June 2026).
Workflows survive crashes by checkpointing every step to SQLite.
"""

from .engine import DurableEngine
from .decorators import step
from .workflow import Workflow
from .retry import RetryPolicy, FAST_RETRY, PERSISTENT_RETRY, NO_RETRY

__version__ = "0.2.0"
__all__ = [
    "DurableEngine", "step", "Workflow",
    "RetryPolicy", "FAST_RETRY", "PERSISTENT_RETRY", "NO_RETRY",
]
