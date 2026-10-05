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


def test_cost_tracker_record():
    from benchmarks.profiling.cost import CostTracker
    tracker = CostTracker()
    tracker.record("groq", input_tokens=100, output_tokens=50, cost_usd=0.001)
    assert tracker.total_cost == pytest.approx(0.001)
    assert tracker.by_provider["groq"] == pytest.approx(0.001)


def test_cost_tracker_multiple_providers():
    from benchmarks.profiling.cost import CostTracker
    tracker = CostTracker()
    tracker.record("groq", input_tokens=100, output_tokens=50, cost_usd=0.001)
    tracker.record("together", input_tokens=200, output_tokens=100, cost_usd=0.005)
    assert tracker.total_cost == pytest.approx(0.006)
    assert len(tracker.traces) == 2


def test_cost_tracker_budget_simulation():
    from benchmarks.profiling.cost import CostTracker
    tracker = CostTracker()
    tracker.record("groq", input_tokens=100, output_tokens=50, cost_usd=0.001)
    sim = tracker.simulate_budget(avg_latency_ms=2000, qps_levels=[1, 10, 100])
    assert 1 in sim
    assert 10 in sim
    assert 100 in sim
    assert sim[1]["cost_per_hour"] > 0


@pytest.mark.asyncio
async def test_track_cost_decorator():
    from benchmarks.profiling.cost import CostTracker, track_cost
    tracker = CostTracker()

    @track_cost(tracker, "groq")
    async def llm_call():
        return {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    result = await llm_call()
    assert result["usage"]["prompt_tokens"] == 100
    assert len(tracker.traces) == 1
