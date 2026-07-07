# Phase 2: Evaluation Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a plugin-based evaluation framework that runs 5 benchmarks, proves FDR ≤ α on real data, profiles latency with per-component traces, and simulates cost at production QPS levels.

**Architecture:** Plugin registry with `@register_benchmark` decorator. Each benchmark is a standalone module implementing `BaseBenchmark` ABC. Runner discovers plugins, orchestrates evaluation, and generates JSON + Markdown reports. Profiling decorators capture per-component latency and cost.

**Tech Stack:** Python 3.10+, MAPIE v1, litellm, DeBERTa-v3, sentence-transformers, HuggingFace datasets, matplotlib, pandas, PyYAML

## Global Constraints

- Python 3.10+, <3.13
- All existing 108 tests must continue passing
- Provider fallback chain: Groq → Together → SiliconFlow
- α ∈ {0.01, 0.05, 0.10}
- Calibration/test split: 80/20
- No secrets in code — use keyring for API keys
- TDD: RED-GREEN-REFACTOR for every task

---

## File Structure

```
benchmarks/
├── __init__.py              # Registry + @register_benchmark
├── base.py                  # BaseBenchmark ABC, Sample, Result, Aggregation
├── runner.py                # Orchestrator + CLI
├── config.py                # Load config.yaml
├── hotpotqa.py              # HotpotQA plugin
├── ragtruth.py              # RAGTruth plugin
├── financebench.py          # FinanceBench plugin
├── halueval.py              # HaluEval plugin
├── multihoprag.py           # MultiHop-RAG plugin
├── profiling/
│   ├── __init__.py
│   ├── latency.py           # @trace_latency
│   └── cost.py              # @track_cost
├── config.yaml              # Configuration
├── results/                 # JSON output (gitignored)
└── reports/                 # Markdown reports

tests/
├── test_benchmark_framework.py
├── test_benchmark_hotpotqa.py
├── test_benchmark_ragtruth.py
├── test_benchmark_financebench.py
├── test_benchmark_halueval.py
├── test_benchmark_multihoprag.py
└── test_benchmark_profiling.py
```

---

## Sub-phase 2.1: Framework

### Task 1: Create `benchmarks/__init__.py` — Registry

**Files:**
- Create: `benchmarks/__init__.py`
- Test: `tests/test_benchmark_framework.py`

**Interfaces:**
- Produces: `@register_benchmark` decorator, `BENCHMARK_REGISTRY` dict, `get_all_benchmarks()` function

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark_framework.py
import pytest
from benchmarks import register_benchmark, get_all_benchmarks, BENCHMARK_REGISTRY


def test_register_benchmark_decorator():
    @register_benchmark
    class DummyBenchmark:
        name = "dummy"
        description = "A dummy benchmark"

    assert "dummy" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["dummy"] is DummyBenchmark


def test_get_all_benchmarks():
    result = get_all_benchmarks()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_registry_prevents_duplicate_names():
    @register_benchmark
    class BenchmarkA:
        name = "unique_name"
        description = "First"

    with pytest.raises(ValueError, match="already registered"):
        @register_benchmark
        class BenchmarkB:
            name = "unique_name"
            description = "Second"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_framework.py -v`
Expected: FAIL with `ImportError: cannot import name 'register_benchmark'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/__init__.py
"""Benchmark plugin registry."""

from __future__ import annotations

from typing import Any, Callable, Type

BENCHMARK_REGISTRY: dict[str, Type] = {}


def register_benchmark(cls: Type) -> Type:
    """Register a benchmark class in the global registry."""
    name = getattr(cls, "name", None)
    if name is None:
        raise ValueError(f"Class {cls.__name__} must have a 'name' attribute")
    if name in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Benchmark '{name}' already registered. "
            f"Existing: {BENCHMARK_REGISTRY[name].__name__}, "
            f"New: {cls.__name__}"
        )
    BENCHMARK_REGISTRY[name] = cls
    return cls


def get_all_benchmarks() -> dict[str, Type]:
    """Return all registered benchmarks."""
    return dict(BENCHMARK_REGISTRY)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_framework.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/__init__.py tests/test_benchmark_framework.py
git commit -m "feat(benchmarks): add plugin registry with @register_benchmark decorator"
```

---

### Task 2: Create `benchmarks/base.py` — Data Models

**Files:**
- Create: `benchmarks/base.py`
- Test: `tests/test_benchmark_framework.py` (append)

**Interfaces:**
- Produces: `Sample`, `Result`, `Aggregation`, `BaseBenchmark` ABC

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_benchmark_framework.py
from benchmarks.base import Sample, Result, Aggregation, BaseBenchmark
import pytest


def test_sample_dataclass():
    sample = Sample(
        id="q1",
        question="What is the capital of France?",
        context="France is a country in Europe.",
        ground_truth="Paris",
    )
    assert sample.id == "q1"
    assert sample.question == "What is the capital of France?"


def test_result_dataclass():
    result = Result(
        sample_id="q1",
        prediction_set=["Paris", "Lyon"],
        abstained=False,
        confidence=0.92,
        latency_trace=None,
        cost_trace=None,
    )
    assert result.sample_id == "q1"
    assert result.abstained is False


def test_aggregation_dataclass():
    agg = Aggregation(
        fdr=0.032,
        coverage=0.94,
        abstention_rate=0.18,
        total_samples=200,
        latency_traces=[],
        cost_traces=[],
    )
    assert agg.fdr == pytest.approx(0.032)


def test_base_benchmark_is_abstract():
    with pytest.raises(TypeError):
        BaseBenchmark()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_framework.py::test_sample_dataclass -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/base.py
"""Base benchmark interface and data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sample:
    """A single benchmark sample."""
    id: str
    question: str
    context: str
    ground_truth: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LatencyTrace:
    """Per-component latency breakdown."""
    retrieval_ms: float = 0.0
    nli_scoring_ms: float = 0.0
    llm_judge_ms: float = 0.0
    confidence_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class CostTrace:
    """Per-query cost breakdown."""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


@dataclass
class Result:
    """Result of running the engine on a single sample."""
    sample_id: str
    prediction_set: list[str]
    abstained: bool
    confidence: float
    latency_trace: Optional[LatencyTrace] = None
    cost_trace: Optional[CostTrace] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Aggregation:
    """Aggregated results across all samples."""
    fdr: float
    coverage: float
    abstention_rate: float
    total_samples: int
    latency_traces: list[LatencyTrace] = field(default_factory=list)
    cost_traces: list[CostTrace] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""

    name: str
    description: str

    @abstractmethod
    async def load_dataset(self) -> list[Sample]:
        """Load and return benchmark samples."""
        ...

    @abstractmethod
    async def run_sample(self, sample: Sample, engine) -> Result:
        """Run the engine on a single sample."""
        ...

    @abstractmethod
    async def aggregate(self, results: list[Result]) -> Aggregation:
        """Aggregate results into final metrics."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_framework.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/base.py
git commit -m "feat(benchmarks): add base data models (Sample, Result, Aggregation, BaseBenchmark)"
```

---

### Task 3: Create `benchmarks/profiling/latency.py` — Latency Tracer

**Files:**
- Create: `benchmarks/profiling/__init__.py`
- Create: `benchmarks/profiling/latency.py`
- Test: `tests/test_benchmark_profiling.py`

**Interfaces:**
- Produces: `@trace_latency` decorator, `LatencyTracer` context manager

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark_profiling.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_profiling.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/profiling/__init__.py
"""Profiling utilities for benchmarks."""
from benchmarks.profiling.latency import trace_latency, LatencyTracer
from benchmarks.profiling.cost import track_cost, CostTracker

__all__ = ["trace_latency", "LatencyTracer", "track_cost", "CostTracker"]
```

```python
# benchmarks/profiling/latency.py
"""Per-component latency tracing."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Callable, Optional


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_profiling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/profiling/
git commit -m "feat(benchmarks): add latency tracer with @trace_latency decorator"
```

---

### Task 4: Create `benchmarks/profiling/cost.py` — Cost Tracker

**Files:**
- Create: `benchmarks/profiling/cost.py`
- Test: `tests/test_benchmark_profiling.py` (append)

**Interfaces:**
- Produces: `@track_cost` decorator, `CostTracker` class

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_benchmark_profiling.py
from benchmarks.profiling.cost import CostTracker, track_cost


def test_cost_tracker_record():
    tracker = CostTracker()
    tracker.record("groq", input_tokens=100, output_tokens=50, cost_usd=0.001)
    assert tracker.total_cost == pytest.approx(0.001)
    assert tracker.by_provider["groq"] == pytest.approx(0.001)


def test_cost_tracker_multiple_providers():
    tracker = CostTracker()
    tracker.record("groq", input_tokens=100, output_tokens=50, cost_usd=0.001)
    tracker.record("together", input_tokens=200, output_tokens=100, cost_usd=0.005)
    assert tracker.total_cost == pytest.approx(0.006)
    assert len(tracker.traces) == 2


def test_cost_tracker_budget_simulation():
    tracker = CostTracker()
    tracker.record("groq", input_tokens=100, output_tokens=50, cost_usd=0.001)
    sim = tracker.simulate_budget(avg_latency_ms=2000, qps_levels=[1, 10, 100])
    assert 1 in sim
    assert 10 in sim
    assert 100 in sim
    assert sim[1]["cost_per_hour"] > 0


@pytest.mark.asyncio
async def test_track_cost_decorator():
    tracker = CostTracker()

    @track_cost(tracker, "groq")
    async def llm_call():
        return {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    result = await llm_call()
    assert result["usage"]["prompt_tokens"] == 100
    assert len(tracker.traces) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_profiling.py::test_cost_tracker_record -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/profiling/cost.py
"""Cost tracking and budget simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional


# Provider pricing (USD per 1K tokens)
PROVIDER_PRICING = {
    "groq": {"input": 0.05, "output": 0.10},
    "together": {"input": 0.10, "output": 0.10},
    "siliconflow": {"input": 0.01, "output": 0.02},
}


@dataclass
class CostTrace:
    """Single cost trace entry."""
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False


class CostTracker:
    """Collects and computes cost statistics."""

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
        """Record a cost trace."""
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
        """Average cost per query."""
        if not self.traces:
            return 0.0
        return self.total_cost / len(self.traces)

    def simulate_budget(
        self, avg_latency_ms: float, qps_levels: list[int]
    ) -> dict[int, dict[str, float]]:
        """Simulate cost at different QPS levels."""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_profiling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/profiling/cost.py
git commit -m "feat(benchmarks): add cost tracker with budget simulation"
```

---

### Task 5: Create `benchmarks/config.py` — Configuration Loader

**Files:**
- Create: `benchmarks/config.py`
- Create: `benchmarks/config.yaml`
- Test: `tests/test_benchmark_framework.py` (append)

**Interfaces:**
- Produces: `BenchmarkConfig` dataclass, `load_config()` function

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_benchmark_framework.py
from benchmarks.config import load_config, BenchmarkConfig


def test_load_config():
    config = load_config()
    assert isinstance(config, BenchmarkConfig)
    assert 0.01 in config.alpha_values
    assert 0.05 in config.alpha_values
    assert 0.10 in config.alpha_values
    assert config.calibration_ratio == 0.8


def test_config_providers():
    config = load_config()
    assert "groq" in config.providers
    assert "together" in config.providers
    assert "siliconflow" in config.providers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_framework.py::test_load_config -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/config.yaml
alpha_values: [0.01, 0.05, 0.10]
calibration_ratio: 0.8
providers:
  - groq
  - together
  - siliconflow
budget_simulation:
  qps_levels: [1, 10, 100]
  duration_seconds: 3600
output:
  results_dir: benchmarks/results
  reports_dir: benchmarks/reports
```

```python
# benchmarks/config.py
"""Benchmark configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class BudgetSimConfig:
    qps_levels: list[int] = field(default_factory=lambda: [1, 10, 100])
    duration_seconds: int = 3600


@dataclass
class OutputConfig:
    results_dir: str = "benchmarks/results"
    reports_dir: str = "benchmarks/reports"


@dataclass
class BenchmarkConfig:
    alpha_values: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    calibration_ratio: float = 0.8
    providers: list[str] = field(default_factory=lambda: ["groq", "together", "siliconflow"])
    budget_simulation: BudgetSimConfig = field(default_factory=BudgetSimConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Optional[Path] = None) -> BenchmarkConfig:
    """Load benchmark configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        return BenchmarkConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    budget = BudgetSimConfig(**data.get("budget_simulation", {}))
    output = OutputConfig(**data.get("output", {}))

    return BenchmarkConfig(
        alpha_values=data.get("alpha_values", [0.01, 0.05, 0.10]),
        calibration_ratio=data.get("calibration_ratio", 0.8),
        providers=data.get("providers", ["groq", "together", "siliconflow"]),
        budget_simulation=budget,
        output=output,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_framework.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/config.py benchmarks/config.yaml
git commit -m "feat(benchmarks): add YAML config loader"
```

---

### Task 6: Create `benchmarks/runner.py` — Orchestrator + CLI

**Files:**
- Create: `benchmarks/runner.py`
- Test: `tests/test_benchmark_framework.py` (append)

**Interfaces:**
- Consumes: `BENCHMARK_REGISTRY`, `load_config`, `BaseBenchmark`
- Produces: `BenchmarkRunner` class, CLI entry point

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_benchmark_framework.py
from benchmarks.runner import BenchmarkRunner


def test_runner_discovers_benchmarks():
    runner = BenchmarkRunner()
    assert len(runner.benchmarks) > 0


def test_runner_config():
    runner = BenchmarkRunner()
    assert runner.config is not None
    assert runner.config.calibration_ratio == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_framework.py::test_runner_discovers_benchmarks -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/runner.py
"""Benchmark runner orchestrator and CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmarks import get_all_benchmarks
from benchmarks.config import load_config, BenchmarkConfig
from benchmarks.base import Aggregation


class BenchmarkRunner:
    """Discovers and runs benchmark plugins."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or load_config()
        self.benchmarks = get_all_benchmarks()

    async def run_one(
        self, name: str, alpha: float = 0.05, dry_run: bool = False
    ) -> dict:
        """Run a single benchmark."""
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not found. Available: {list(self.benchmarks.keys())}")

        benchmark_cls = self.benchmarks[name]
        benchmark = benchmark_cls()

        print(f"Loading {name} dataset...")
        samples = await benchmark.load_dataset()
        print(f"Loaded {len(samples)} samples")

        calibration_size = int(len(samples) * self.config.calibration_ratio)
        calibration_samples = samples[:calibration_size]
        test_samples = samples[calibration_size:]

        if dry_run:
            print(f"Dry run: would calibrate on {len(calibration_size)} samples, test on {len(test_samples)}")
            return {"benchmark": name, "dry_run": True, "samples": len(samples)}

        # Run evaluation
        results = []
        for sample in test_samples:
            result = await benchmark.run_sample(sample, engine=None)
            results.append(result)

        aggregation = await benchmark.aggregate(results)

        # Save results
        output_data = {
            "benchmark": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "alpha": alpha,
                "calibration_size": calibration_size,
                "test_size": len(test_samples),
                "providers": self.config.providers,
            },
            "metrics": {
                "fdr": aggregation.fdr,
                "coverage": aggregation.coverage,
                "abstention_rate": aggregation.abstention_rate,
                "total_samples": aggregation.total_samples,
            },
        }

        # Write JSON
        results_dir = Path(self.config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        json_path = results_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {json_path}")
        return output_data

    async def run_all(self, alpha: float = 0.05, dry_run: bool = False) -> list[dict]:
        """Run all registered benchmarks."""
        results = []
        for name in self.benchmarks:
            result = await self.run_one(name, alpha=alpha, dry_run=dry_run)
            results.append(result)
        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Conformal RAG Benchmark Runner")
    parser.add_argument("--benchmark", "-b", help="Run a specific benchmark")
    parser.add_argument("--alpha", "-a", type=float, default=0.05, help="Significance level")
    parser.add_argument("--dry-run", action="store_true", help="Load only, no evaluation")
    args = parser.parse_args()

    runner = BenchmarkRunner()

    if args.benchmark:
        asyncio.run(runner.run_one(args.benchmark, alpha=args.alpha, dry_run=args.dry_run))
    else:
        asyncio.run(runner.run_all(alpha=args.alpha, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_framework.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/runner.py
git commit -m "feat(benchmarks): add orchestrator runner with CLI"
```

---

### Task 7: Update `pyproject.toml` — Add Benchmark Dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: none
- Produces: `benchmarks` optional dependency group

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_benchmark_framework.py
def test_imports_datasets():
    import datasets
    assert hasattr(datasets, "load_dataset")


def test_imports_pandas():
    import pandas
    assert hasattr(pandas, "DataFrame")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_framework.py::test_imports_datasets -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Modify pyproject.toml**

```toml
# Add after [project.optional-dependencies] dev section
benchmarks = [
    "datasets>=2.18",
    "matplotlib>=3.8",
    "pandas>=2.2",
    "pyyaml>=6.0",
]
```

- [ ] **Step 4: Install and run test**

Run: `uv pip install -e ".[benchmarks]"`
Run: `pytest tests/test_benchmark_framework.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add benchmark dependencies (datasets, matplotlib, pandas, pyyaml)"
```

---

### Task 8: Add `benchmarks/results/` and `benchmarks/reports/` to `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append to .gitignore**

```gitignore
# Benchmark output
benchmarks/results/
benchmarks/reports/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore benchmark output directories"
```

---

### Task 9: Run Full Framework Tests

**Files:**
- None (verification only)

- [ ] **Step 1: Run all framework tests**

Run: `pytest tests/test_benchmark_framework.py tests/test_benchmark_profiling.py -v`
Expected: ALL PASS

- [ ] **Step 2: Run all existing tests**

Run: `pytest tests/ -v`
Expected: ALL PASS (108 existing + new framework tests)

- [ ] **Step 3: Commit if needed**

```bash
git status
# If changes, commit them
```

---

## Sub-phase 2.2: HotpotQA + RAGTruth

### Task 10: Create `benchmarks/hotpotqa.py` — HotpotQA Plugin

**Files:**
- Create: `benchmarks/hotpotqa.py`
- Test: `tests/test_benchmark_hotpotqa.py`

**Interfaces:**
- Consumes: `BaseBenchmark`, `Sample`, `Result`, `Aggregation`, `@register_benchmark`
- Produces: `HotpotQABenchmark` class

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark_hotpotqa.py
import pytest
from benchmarks.hotpotqa import HotpotQABenchmark


@pytest.mark.asyncio
async def test_hotpotqa_load_dataset():
    benchmark = HotpotQABenchmark()
    samples = await benchmark.load_dataset()
    assert len(samples) > 0
    assert samples[0].question is not None
    assert samples[0].ground_truth is not None


@pytest.mark.asyncio
async def test_hotpotqa_sample_format():
    benchmark = HotpotQABenchmark()
    samples = await benchmark.load_dataset()
    sample = samples[0]
    assert hasattr(sample, "id")
    assert hasattr(sample, "question")
    assert hasattr(sample, "context")
    assert hasattr(sample, "ground_truth")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_hotpotqa.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/hotpotqa.py
"""HotpotQA benchmark plugin."""

from __future__ import annotations

import asyncio
from typing import Optional

from benchmarks import register_benchmark
from benchmarks.base import (
    BaseBenchmark,
    Sample,
    Result,
    Aggregation,
    LatencyTrace,
    CostTrace,
)


@register_benchmark
class HotpotQABenchmark(BaseBenchmark):
    """HotpotQA: Multi-hop reasoning benchmark."""

    name = "hotpotqa"
    description = "Multi-hop question answering — tests retrieval + reasoning"

    async def load_dataset(self) -> list[Sample]:
        """Load HotpotQA dataset from HuggingFace."""
        from datasets import load_dataset

        dataset = load_dataset("hotpot_qa", "distractor", split="train[:1000]")
        samples = []
        for i, item in enumerate(dataset):
            context = " ".join(item["context"]["sentences"])
            samples.append(
                Sample(
                    id=f"hotpotqa_{i}",
                    question=item["question"],
                    context=context,
                    ground_truth=item["answer"],
                    metadata={
                        "type": item["type"],
                        "level": item["level"],
                    },
                )
            )
        return samples

    async def run_sample(self, sample: Sample, engine) -> Result:
        """Run the conformal engine on a single sample."""
        # TODO: Integrate with actual engine in Task 12
        return Result(
            sample_id=sample.id,
            prediction_set=[sample.ground_truth],
            abstained=False,
            confidence=0.9,
        )

    async def aggregate(self, results: list[Result]) -> Aggregation:
        """Compute FDR, coverage, abstention rate."""
        total = len(results)
        abstained = sum(1 for r in results if r.abstained)
        # FDR = false positives / (true positives + false positives)
        # For now, placeholder — real computation requires ground truth comparison
        return Aggregation(
            fdr=0.0,
            coverage=1.0 - (abstained / total) if total > 0 else 0.0,
            abstention_rate=abstained / total if total > 0 else 0.0,
            total_samples=total,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_hotpotqa.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/hotpotqa.py tests/test_benchmark_hotpotqa.py
git commit -m "feat(benchmarks): add HotpotQA plugin"
```

---

### Task 11: Create `benchmarks/ragtruth.py` — RAGTruth Plugin

**Files:**
- Create: `benchmarks/ragtruth.py`
- Test: `tests/test_benchmark_ragtruth.py`

**Interfaces:**
- Consumes: `BaseBenchmark`, `Sample`, `Result`, `Aggregation`, `@register_benchmark`
- Produces: `RAGTruthBenchmark` class

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark_ragtruth.py
import pytest
from benchmarks.ragtruth import RAGTruthBenchmark


@pytest.mark.asyncio
async def test_ragtruth_load_dataset():
    benchmark = RAGTruthBenchmark()
    samples = await benchmark.load_dataset()
    assert len(samples) > 0
    assert samples[0].question is not None
    assert samples[0].ground_truth is not None


@pytest.mark.asyncio
async def test_ragtruth_metadata():
    benchmark = RAGTruthBenchmark()
    samples = await benchmark.load_dataset()
    sample = samples[0]
    assert "hallucination_type" in sample.metadata or "type" in sample.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_ragtruth.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/ragtruth.py
"""RAGTruth benchmark plugin."""

from __future__ import annotations

from benchmarks import register_benchmark
from benchmarks.base import (
    BaseBenchmark,
    Sample,
    Result,
    Aggregation,
)


@register_benchmark
class RAGTruthBenchmark(BaseBenchmark):
    """RAGTruth: Real LLM hallucination benchmark."""

    name = "ragtruth"
    description = "Real LLM hallucinations — worst-case for embedding detection"

    async def load_dataset(self) -> list[Sample]:
        """Load RAGTruth dataset from HuggingFace."""
        from datasets import load_dataset

        dataset = load_dataset("liamcripwell/ragtruth", split="train[:1000]")
        samples = []
        for i, item in enumerate(dataset):
            samples.append(
                Sample(
                    id=f"ragtruth_{i}",
                    question=item.get("question", item.get("query", "")),
                    context=item.get("context", item.get("passage", "")),
                    ground_truth=item.get("answer", item.get("label", "")),
                    metadata={
                        "hallucination_type": item.get("hallucination_type", "unknown"),
                        "type": item.get("type", "unknown"),
                    },
                )
            )
        return samples

    async def run_sample(self, sample: Sample, engine) -> Result:
        """Run the conformal engine on a single sample."""
        # TODO: Integrate with actual engine in Task 12
        return Result(
            sample_id=sample.id,
            prediction_set=[sample.ground_truth],
            abstained=False,
            confidence=0.9,
        )

    async def aggregate(self, results: list[Result]) -> Aggregation:
        """Compute FDR, coverage, abstention rate."""
        total = len(results)
        abstained = sum(1 for r in results if r.abstained)
        return Aggregation(
            fdr=0.0,
            coverage=1.0 - (abstained / total) if total > 0 else 0.0,
            abstention_rate=abstained / total if total > 0 else 0.0,
            total_samples=total,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_ragtruth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/ragtruth.py tests/test_benchmark_ragtruth.py
git commit -m "feat(benchmarks): add RAGTruth plugin"
```

---

### Task 12: Integrate Benchmarks with Conformal Engine

**Files:**
- Modify: `benchmarks/hotpotqa.py:run_sample`
- Modify: `benchmarks/ragtruth.py:run_sample`
- Modify: `benchmarks/runner.py`

**Interfaces:**
- Consumes: `ConformalEngine` from `src/conformal_rag/engine.py`
- Produces: Real evaluation results

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_benchmark_hotpotqa.py
@pytest.mark.asyncio
async def test_hotpotqa_run_sample_with_engine():
    from conformal_rag.engine import ConformalEngine
    from conformal_rag.config import ConformalConfig

    benchmark = HotpotQABenchmark()
    samples = await benchmark.load_dataset()

    config = ConformalConfig()
    engine = ConformalEngine(config)

    result = await benchmark.run_sample(samples[0], engine)
    assert result.sample_id is not None
    assert isinstance(result.prediction_set, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_hotpotqa.py::test_hotpotqa_run_sample_with_engine -v`
Expected: FAIL (engine not integrated)

- [ ] **Step 3: Update run_sample in both plugins**

```python
# In hotpotqa.py and ragtruth.py, replace run_sample:
async def run_sample(self, sample: Sample, engine) -> Result:
    """Run the conformal engine on a single sample."""
    prediction = await engine.predict(sample.question, context=sample.context)
    return Result(
        sample_id=sample.id,
        prediction_set=prediction.prediction_set,
        abstained=prediction.abstained,
        confidence=prediction.confidence,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_hotpotqa.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/hotpotqa.py benchmarks/ragtruth.py
git commit -m "feat(benchmarks): integrate HotpotQA + RAGTruth with ConformalEngine"
```

---

### Task 13: Run HotpotQA + RAGTruth Benchmarks

**Files:**
- None (execution only)

- [ ] **Step 1: Run HotpotQA**

Run: `python -m benchmarks.runner --benchmark hotpotqa --alpha 0.05`
Expected: JSON output in `benchmarks/results/`, metrics printed

- [ ] **Step 2: Run RAGTruth**

Run: `python -m benchmarks.runner --benchmark ragtruth --alpha 0.05`
Expected: JSON output in `benchmarks/results/`, metrics printed

- [ ] **Step 3: Verify FDR ≤ α**

Check JSON output: `fdr` should be ≤ 0.05

- [ ] **Step 4: Commit results**

```bash
git add benchmarks/results/
git commit -m "docs: add HotpotQA + RAGTruth benchmark results"
```

---

## Sub-phase 2.3: FinanceBench + HaluEval + MultiHop-RAG

### Task 14: Create `benchmarks/financebench.py`

**Files:**
- Create: `benchmarks/financebench.py`
- Test: `tests/test_benchmark_financebench.py`

- [ ] **Step 1-5:** Same pattern as Task 10 (HotpotQA)

---

### Task 15: Create `benchmarks/halueval.py`

**Files:**
- Create: `benchmarks/halueval.py`
- Test: `tests/test_benchmark_halueval.py`

- [ ] **Step 1-5:** Same pattern as Task 10

---

### Task 16: Create `benchmarks/multihoprag.py`

**Files:**
- Create: `benchmarks/multihoprag.py`
- Test: `tests/test_benchmark_multihoprag.py`

- [ ] **Step 1-5:** Same pattern as Task 10

---

### Task 17: Run Remaining Benchmarks

- [ ] **Step 1:** Run FinanceBench
- [ ] **Step 2:** Run HaluEval
- [ ] **Step 3:** Run MultiHop-RAG
- [ ] **Step 4:** Verify FDR ≤ α on all
- [ ] **Step 5:** Commit results

---

## Sub-phase 2.4: Profiling + Reports

### Task 18: Integrate Latency Tracing into Runner

**Files:**
- Modify: `benchmarks/runner.py`
- Modify: `benchmarks/hotpotqa.py:run_sample` (wrap with tracer)

- [ ] **Step 1-5:** Add `@trace_latency` to each pipeline stage

---

### Task 19: Integrate Cost Tracking into Runner

**Files:**
- Modify: `benchmarks/runner.py`
- Modify: `benchmarks/profiling/cost.py`

- [ ] **Step 1-5:** Add `@track_cost` to LLM calls

---

### Task 20: Generate Markdown Reports

**Files:**
- Create: `benchmarks/reporting.py`
- Test: `tests/test_benchmark_profiling.py` (append)

- [ ] **Step 1-5:** Report generator that reads JSON, produces Markdown

---

### Task 21: Budget Simulation

**Files:**
- Modify: `benchmarks/runner.py`

- [ ] **Step 1-5:** Add `--simulate-budget` flag to runner

---

### Task 22: Aggregate Report Across All Benchmarks

**Files:**
- Modify: `benchmarks/reporting.py`

- [ ] **Step 1-5:** Cross-benchmark comparison table

---

### Task 23: Final Verification

- [ ] **Step 1:** Run all tests: `pytest tests/ -v`
- [ ] **Step 2:** Run all benchmarks: `python -m benchmarks.runner`
- [ ] **Step 3:** Verify all JSON outputs
- [ ] **Step 4:** Verify all Markdown reports
- [ ] **Step 5:** Final commit

---

## Summary

| Sub-phase | Tasks | Deliverable |
|-----------|-------|-------------|
| 2.1 Framework | 1-9 | Plugin registry, base classes, profiling, runner, CLI |
| 2.2 HotpotQA + RAGTruth | 10-13 | Two working benchmarks with FDR verification |
| 2.3 Remaining Benchmarks | 14-17 | All 5 benchmarks running |
| 2.4 Profiling + Reports | 18-23 | Latency traces, cost analysis, Markdown reports |
