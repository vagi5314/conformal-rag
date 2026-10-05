"""Per-component latency tracing."""

from __future__ import annotations

import time
from collections import defaultdict
from functools import wraps
from typing import Callable, Optional


class LatencyTracer:
    """Collects and computes latency statistics."""

    def __init__(self):
        self.traces: dict[str, float] = defaultdict(float)
        self._records: dict[str, list[float]] = defaultdict(list)
        self._start: Optional[float] = None
        self.total_ms: float = 0.0

    def record(self, operation: str, milliseconds: float) -> None:
        """Record a latency measurement."""
        self.traces[operation] = milliseconds
        self._records[operation].append(milliseconds)

    def percentile(self, operation: str, p: float) -> float:
        """Compute percentile for an operation."""
        values = sorted(self._records.get(operation, []))
        if not values:
            return 0.0
        idx = int(len(values) * p / 100)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def summary(self) -> dict[str, dict[str, float]]:
        """Get p50/p95/p99 for all operations."""
        result = {}
        for op in self._records:
            result[op] = {
                "p50": self.percentile(op, 50),
                "p95": self.percentile(op, 95),
                "p99": self.percentile(op, 99),
            }
        return result

    async def __aenter__(self):
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, *args):
        self.total_ms = (time.perf_counter() - self._start) * 1000


def trace_latency(tracer: LatencyTracer, operation: str) -> Callable:
    """Decorator that traces async function latency."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            tracer.record(operation, elapsed_ms)
            return result

        return wrapper

    return decorator
