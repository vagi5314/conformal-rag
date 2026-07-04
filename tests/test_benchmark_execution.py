import asyncio
import pytest
from benchmarks.execution.parallel import ParallelExecutor


@pytest.mark.asyncio
async def test_parallel_executor_respects_concurrency():
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def slow_task(item):
        nonlocal max_concurrent, current_concurrent
        async with lock:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
        await asyncio.sleep(0.05)
        async with lock:
            current_concurrent -= 1
        return item

    executor = ParallelExecutor(max_concurrency=3)
    results = await executor.run_all(list(range(10)), slow_task)
    assert max_concurrent <= 3
    assert results == list(range(10))


@pytest.mark.asyncio
async def test_parallel_executor_isolates_errors():
    async def failing_task(item):
        if item == 5:
            raise ValueError("item 5 fails")
        return item

    executor = ParallelExecutor(max_concurrency=2)
    results = await executor.run_all(list(range(10)), failing_task)
    assert results[5] is None
    assert results[0] == 0
    assert results[9] == 9
