"""Cost tracking and budget simulation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional


PROVIDER_PRICING = {
    "groq": {"input": 0.05, "output": 0.10},
    "together": {"input": 0.10, "output": 0.10},
    "siliconflow": {"input": 0.01, "output": 0.02},
}


@dataclass
class CostTrace:
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False


class CostTracker:
    def __init__(self):
        self.traces: list[CostTrace] = []
        self.total_cost: float = 0.0
        self.by_provider: dict[str, float] = {}

    def record(
        self,
        provider: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: Optional[float] = None,
        cached: bool = False,
    ) -> None:
        if cost_usd is None:
            pricing = PROVIDER_PRICING.get(provider, {"input": 0.1, "output": 0.1})
            cost_usd = (
                (input_tokens / 1000) * pricing["input"]
                + (output_tokens / 1000) * pricing["output"]
            )

        trace = CostTrace(
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cached=cached,
        )
        self.traces.append(trace)
        self.total_cost += cost_usd
        self.by_provider[provider] = self.by_provider.get(provider, 0.0) + cost_usd

    def avg_cost_per_query(self) -> float:
        if not self.traces:
            return 0.0
        return self.total_cost / len(self.traces)

    def simulate_budget(
        self, avg_latency_ms: float, qps_levels: list[int]
    ) -> dict[int, dict[str, float]]:
        avg_cost = self.avg_cost_per_query()
        results = {}
        for qps in qps_levels:
            queries_per_hour = qps * 3600
            cost_per_hour = avg_cost * queries_per_hour
            results[qps] = {
                "cost_per_hour": cost_per_hour,
                "cost_per_day": cost_per_hour * 24,
                "cost_per_month": cost_per_hour * 24 * 30,
            }
        return results


def track_cost(tracker: CostTracker, provider: str) -> Callable:
    """Decorator that tracks async function cost."""
    def decorator(func: Callable) -> Callable:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"track_cost requires an async function, got {func.__name__}. "
                "Use async def for functions decorated with @track_cost."
            )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            usage = result.get("usage", {}) if isinstance(result, dict) else {}
            tracker.record(
                provider=provider,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
            return result
        return wrapper
    return decorator
