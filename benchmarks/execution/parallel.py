from __future__ import annotations

import asyncio
from typing import Callable, Awaitable, TypeVar, Optional

T = TypeVar("T")


class ParallelExecutor:
    """Controlled parallelism via asyncio.gather + semaphore."""

    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max_concurrency

    async def run_all(
        self,
        items: list,
        fn: Callable[[any], Awaitable[T]],
        on_result: Optional[Callable[[T], None]] = None,
    ) -> list[T | None]:
        """Run fn on all items with bounded concurrency."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _guarded(item):
            async with semaphore:
                return await fn(item)

        tasks = [_guarded(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for result in results:
            if isinstance(result, Exception):
                processed.append(None)
            else:
                if on_result:
                    on_result(result)
                processed.append(result)
        return processed
