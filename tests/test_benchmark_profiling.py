import asyncio
import pytest
from benchmarks.profiling.latency import trace_latency, LatencyTracer


@pytest.mark.asyncio
async def test_trace_latency_decorator():
    tracer = LatencyTracer()

    @trace_latency(tracer, "test_op")
    async def slow_operation():
        await asyncio.sleep(0.01)
        return "done"

    result = await slow_operation()
    assert result == "done"
    assert tracer.traces["test_op"] > 0


@pytest.mark.asyncio
async def test_trace_latency_context_manager():
    async with LatencyTracer() as tracer:
        await asyncio.sleep(0.01)

    assert tracer.total_ms > 0


def test_latency_tracer_percentiles():
    tracer = LatencyTracer()
    for i in range(100):
        tracer.record("op", float(i))

    p50 = tracer.percentile("op", 50)
    p95 = tracer.percentile("op", 95)
    assert p50 == pytest.approx(50.0, abs=1.0)
    assert p95 == pytest.approx(95.0, abs=1.0)
