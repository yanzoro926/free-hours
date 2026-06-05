"""
@step decorator — wraps Python functions as durable workflow steps.

The decorator adds metadata but doesn't change function behavior.
The engine uses this metadata to track step identity and configuration.
"""

from functools import wraps
from typing import Callable, Optional, Any


def step(
    retries: int = 0,
    timeout_seconds: Optional[float] = None,
    description: str = "",
):
    """
    Decorator to mark a function as a durable step.

    Args:
        retries: Number of retry attempts on failure.
        timeout_seconds: Max execution time before timeout.
        description: Human-readable description of the step.

    Example:
        @step(retries=3)
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Attach durable metadata
        wrapper._durable_step = True
        wrapper._step_retries = retries
        wrapper._step_timeout = timeout_seconds
        wrapper._step_description = description or (func.__doc__ or "").strip()
        wrapper._step_name = func.__name__

        return wrapper

    return decorator


def is_durable_step(func: Callable) -> bool:
    """Check if a function is marked as a durable step."""
    return getattr(func, "_durable_step", False)
