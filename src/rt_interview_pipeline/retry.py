from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def with_backoff(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay_seconds: float = 0.15,
    retryable: tuple[type[Exception], ...] = (Exception,),
) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except retryable as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(base_delay_seconds * (2**attempt))
    if last_error is None:
        raise RuntimeError("retry operation failed without an exception")
    raise last_error
