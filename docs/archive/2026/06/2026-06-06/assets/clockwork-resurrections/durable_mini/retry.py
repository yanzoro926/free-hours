"""
Retry policies for durable steps.

Provides configurable retry behavior with exponential backoff.
"""

import time
import random
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class RetryPolicy:
    """Configuration for step retry behavior."""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if retry should be attempted."""
        if attempt > self.max_retries:
            return False
        return isinstance(exception, self.retryable_exceptions)

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = self.initial_delay_seconds * (self.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay_seconds)
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay


# Pre-built policies

DEFAULT_RETRY = RetryPolicy(max_retries=0)  # No retry by default

FAST_RETRY = RetryPolicy(
    max_retries=3,
    initial_delay_seconds=0.5,
    max_delay_seconds=10.0,
    backoff_multiplier=2.0,
    jitter=True,
)

PERSISTENT_RETRY = RetryPolicy(
    max_retries=10,
    initial_delay_seconds=2.0,
    max_delay_seconds=120.0,
    backoff_multiplier=2.0,
    jitter=True,
)

NO_RETRY = RetryPolicy(max_retries=0)


def with_retry(
    func: Callable,
    policy: RetryPolicy = FAST_RETRY,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """
    Execute a function with retry logic.

    Args:
        func: The function to execute.
        policy: Retry configuration.
        on_retry: Optional callback invoked before each retry (attempt, exception, delay).

    Returns:
        The function's return value.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None
    for attempt in range(1, policy.max_retries + 2):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if not policy.should_retry(attempt, e):
                raise
            delay = policy.delay_for_attempt(attempt)
            if on_retry:
                on_retry(attempt, e, delay)
            time.sleep(delay)
    raise last_exception  # type: ignore
