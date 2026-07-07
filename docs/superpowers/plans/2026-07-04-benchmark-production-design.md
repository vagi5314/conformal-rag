# Benchmark Production System — Design Document

**Date**: 2026-07-04
**Status**: Draft
**Depends on**: Phase 2 Evaluation Framework (current: 139 tests passing)
**Author**: opencode (mimo-v2.5-free)

---

## 1. Problem Statement

The current benchmark runner (`benchmarks/runner.py:66-69`) processes samples sequentially with no error handling:

```python
results = []
for sample in test_samples:
    result = await benchmark.run_sample(sample, engine)
    results.append(result)
```

This causes 6 production gaps:

| Gap | Impact | Root Cause |
|-----|--------|------------|
| No parallelization | 50 samples = 50x serial latency | Sequential `for` loop |
| No retry/fallback | Single API failure kills entire run | Bare `await` with no try/except |
| No result caching | Re-runs waste $4+ in API credits | No persistence layer |
| No progress reporting | No visibility into 30+ minute runs | Only `print()` statements |
| No incremental saves | Crash = total progress loss | Single JSON dump at end |
| No error isolation | One bad sample kills entire benchmark | No per-sample try/except |

---

## 2. Architecture

### 2.1 New Module Structure

```
benchmarks/
├── __init__.py              # (existing) Registry
├── base.py                  # (existing) Data models
├── config.py                # (existing) Config loader
├── runner.py                # (existing) Orchestrator — refactor internals
├── hotpotqa.py              # (existing) Plugin
├── ragtruth.py              # (existing) Plugin
├── profiling/
│   ├── __init__.py          # (existing)
│   ├── latency.py           # (existing) LatencyTracer
│   └── cost.py              # (existing) CostTracker
├── execution/               # (NEW) Production execution layer
│   ├── __init__.py
│   ├── parallel.py          # asyncio.gather + semaphore
│   ├── retry.py             # Exponential backoff + provider fallback
│   ├── cache.py             # SQLite result cache
│   ├── progress.py          # tqdm progress bars
│   ├── checkpoint.py        # Incremental saves
│   └── error隔离.py         # Per-sample error isolation
└── config.yaml              # (existing) Shared config
```

### 2.2 Data Flow — Before vs After

**Before (current):**
```
Runner → for sample in samples: → engine.predict() → append to list → dump JSON
```

**After (production):**
```
Runner → load cache → create semaphore → tqdm progress bar
         │
         ▼
    asyncio.gather(*[run_sample_with_retry(s) for s in samples])
         │
         ├── retry.py: try → engine.predict() → cache result
         │              except → fallback provider → retry
         │
         ├── checkpoint.py: save every N samples to SQLite
         │
         └── error隔离.py: try/except per sample → log error, continue
         │
         ▼
    aggregate results → final JSON + markdown report
```

---

## 3. Component Design

### 3.1 Parallel Execution — `execution/parallel.py`

**Purpose**: Process samples concurrently with controlled parallelism.

**Design**:
```python
import asyncio
from typing import Callable, Awaitable, TypeVar

T = TypeVar("T")

class ParallelExecutor:
    """Controlled parallelism via asyncio.gather + semaphore."""

    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run_all(
        self,
        items: list,
        fn: Callable[[any], Awaitable[T]],
        on_result: Callable[[T], None] | None = None,
    ) -> list[T]:
        """Run fn on all items with bounded concurrency."""
        async def _guarded(item):
            async with self._semaphore:
                return await fn(item)

        tasks = [_guarded(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Error already logged by error isolation wrapper
                processed.append(None)
            else:
                if on_result:
                    on_result(result)
                processed.append(result)
        return processed
```

**Key decisions**:
- Default `max_concurrency=5` — conservative for free-tier LLM APIs (rate limits ~10 QPS)
- `return_exceptions=True` — prevents `gather` from cancelling other tasks on error
- `on_result` callback — allows checkpoint/progress hooks without coupling

**Configuration** (`config.yaml`):
```yaml
execution:
  max_concurrency: 5        # Parallel samples
  checkpoint_interval: 25   # Save every N samples
  cache_enabled: true       # SQLite result cache
```

### 3.2 Retry with Fallback — `execution/retry.py`

**Purpose**: Survive transient API failures with exponential backoff and provider fallback.

**Design**:
```python
import asyncio
import random
from typing import Callable, Awaitable

class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0        # seconds
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True

class RetryWithFallback:
    """Exponential backoff retry with provider fallback chain."""

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    async def execute(
        self,
        primary_fn: Callable[[], Awaitable],
        fallback_fns: list[Callable[[], Awaitable]] | None = None,
    ) -> any:
        """Try primary, retry with backoff, then fallback providers."""
        # Try primary with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return await primary_fn()
            except Exception as e:
                last_error = e
                delay = self._compute_delay(attempt)
                await asyncio.sleep(delay)

        # Primary exhausted — try fallbacks
        if fallback_fns:
            for fallback in fallback_fns:
                try:
                    return await fallback()
                except Exception:
                    continue

        raise last_error  # All providers failed

    def _compute_delay(self, attempt: int) -> float:
        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
        delay = min(delay, self.config.max_delay)
        if self.config.jitter:
            delay *= random.uniform(0.5, 1.0)
        return delay
```

**Provider fallback chain** (from `config.yaml`):
```yaml
providers:
  - groq          # Primary — fastest, cheapest
  - together      # Fallback 1 — reliable
  - siliconflow   # Fallback 2 — cheapest
```

**Integration with runner**:
```python
retry = RetryWithFallback(config)
result = await retry.execute(
    primary_fn=lambda: engine.predict(q, ctx, gt),
    fallback_fns=[
        lambda: engine_with_provider("together").predict(q, ctx, gt),
        lambda: engine_with_provider("siliconflow").predict(q, ctx, gt),
    ],
)
```

### 3.3 Result Caching — `execution/cache.py`

**Purpose**: Avoid re-running expensive LLM API calls on re-runs.

**Design**:
```python
import sqlite3
import json
from pathlib import Path
from benchmarks.base import Result, LatencyTrace, CostTrace

class ResultCache:
    """SQLite-based result cache."""

    def __init__(self, db_path: str = "benchmarks/cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                sample_id TEXT PRIMARY KEY,
                benchmark TEXT NOT NULL,
                alpha REAL NOT NULL,
                result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def get(self, sample_id: str, benchmark: str, alpha: float) -> Result | None:
        """Retrieve cached result."""
        row = self._conn.execute(
            "SELECT result_json FROM results WHERE sample_id=? AND benchmark=? AND alpha=?",
            (sample_id, benchmark, alpha),
        ).fetchone()
        if row:
            data = json.loads(row[0])
            return self._deserialize_result(data)
        return None

    def put(self, result: Result, benchmark: str, alpha: float):
        """Store result in cache."""
        data = self._serialize_result(result)
        self._conn.execute(
            "INSERT OR REPLACE INTO results (sample_id, benchmark, alpha, result_json) VALUES (?, ?, ?, ?)",
            (result.sample_id, benchmark, alpha, json.dumps(data)),
        )
        self._conn.commit()

    def has(self, sample_id: str, benchmark: str, alpha: float) -> bool:
        """Check if result exists in cache."""
        row = self._conn.execute(
            "SELECT 1 FROM results WHERE sample_id=? AND benchmark=? AND alpha=?",
            (sample_id, benchmark, alpha),
        ).fetchone()
        return row is not None

    def stats(self, benchmark: str, alpha: float) -> dict:
        """Cache hit/miss stats."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE benchmark=? AND alpha=?",
            (benchmark, alpha),
        ).fetchone()[0]
        return {"cached_samples": total}

    def clear(self, benchmark: str | None = None):
        """Clear cache (all or per-benchmark)."""
        if benchmark:
            self._conn.execute("DELETE FROM results WHERE benchmark=?", (benchmark,))
        else:
            self._conn.execute("DELETE FROM results")
        self._conn.commit()

    def _serialize_result(self, result: Result) -> dict:
        return {
            "sample_id": result.sample_id,
            "prediction_set": result.prediction_set,
            "abstained": result.abstained,
            "confidence": result.confidence,
            "metadata": result.metadata,
        }

    def _deserialize_result(self, data: dict) -> Result:
        return Result(
            sample_id=data["sample_id"],
            prediction_set=data["prediction_set"],
            abstained=data["abstained"],
            confidence=data["confidence"],
            metadata=data.get("metadata", {}),
        )
```

**Why SQLite over HuggingFace datasets**:
- Zero external dependency (sqlite3 is stdlib)
- Fast point lookups by `(sample_id, benchmark, alpha)`
- Atomic writes — no corruption on crash
- Easy to inspect/debug with `sqlite3` CLI

### 3.4 Progress Reporting — `execution/progress.py`

**Purpose**: Real-time visibility into long-running benchmarks.

**Design**:
```python
from tqdm import tqdm
from benchmarks.base import Result

class BenchmarkProgress:
    """tqdm-based progress tracking."""

    def __init__(self, total: int, benchmark_name: str, alpha: float):
        self.pbar = tqdm(
            total=total,
            desc=f"[{benchmark_name}] α={alpha}",
            unit="sample",
            ncols=100,
        )
        self.errors = 0
        self.cached = 0

    def update(self, result: Result | None, from_cache: bool = False):
        """Update progress bar."""
        if result is None:
            self.errors += 1
        if from_cache:
            self.cached += 1
        self.pbar.update(1)
        self.pbar.set_postfix({
            "errors": self.errors,
            "cached": self.cached,
        })

    def close(self):
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

**Output example**:
```
[hotpotqa] α=0.05:  45%|████████▌            | 90/200 [01:23<01:42, 1.08sample/s, errors=2, cached=15]
```

### 3.5 Checkpoint Saves — `execution/checkpoint.py`

**Purpose**: Persist progress incrementally so crashes don't lose everything.

**Design**:
```python
import json
from pathlib import Path
from benchmarks.base import Result

class CheckpointManager:
    """Incremental saves to SQLite + periodic JSON snapshots."""

    def __init__(self, checkpoint_dir: str, interval: int = 25):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self._buffer: list[Result] = []
        self._count = 0

    def add(self, result: Result):
        """Buffer result, checkpoint at interval."""
        self._buffer.append(result)
        self._count += 1
        if self._count % self.interval == 0:
            self._flush()

    def _flush(self):
        """Write buffered results to checkpoint file."""
        checkpoint_path = self.dir / f"checkpoint_{self._count}.json"
        data = [self._result_to_dict(r) for r in self._buffer]
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        self._buffer.clear()

    def finalize(self) -> list[Result]:
        """Flush remaining, return all results from latest checkpoint."""
        if self._buffer:
            self._flush()
        # Load latest checkpoint
        checkpoints = sorted(self.dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return []
        with open(checkpoints[-1]) as f:
            data = json.load(f)
        return [self._result_from_dict(d) for d in data]

    def _result_to_dict(self, r: Result) -> dict:
        return {
            "sample_id": r.sample_id,
            "prediction_set": r.prediction_set,
            "abstained": r.abstained,
            "confidence": r.confidence,
            "metadata": r.metadata,
        }

    def _result_from_dict(self, d: dict) -> Result:
        return Result(**d)
```

**Checkpoint lifecycle**:
1. Every 25 samples → write `checkpoint_25.json`, `checkpoint_50.json`, etc.
2. On crash → load latest `checkpoint_N.json`, resume from sample N+1
3. On completion → load final checkpoint, aggregate results

### 3.6 Error Isolation — `execution/error隔离.py`

**Purpose**: One bad sample must never kill the entire benchmark.

**Design**:
```python
import logging
import traceback
from typing import Callable, Awaitable
from benchmarks.base import Result

logger = logging.getLogger(__name__)

class ErrorIsolation:
    """Per-sample error isolation with structured logging."""

    @staticmethod
    async def run_sample(
        fn: Callable[[], Awaitable[Result]],
        sample_id: str,
    ) -> Result | None:
        """Run sample function with full error isolation."""
        try:
            return await fn()
        except Exception as e:
            logger.error(
                "Sample %s failed: %s\n%s",
                sample_id,
                str(e),
                traceback.format_exc(),
            )
            return None
```

**Integration with parallel executor**:
```python
async def _guarded(item):
    async with self._semaphore:
        return await ErrorIsolation.run_sample(
            fn=lambda: benchmark.run_sample(item, engine),
            sample_id=item.id,
        )
```

**Error handling strategy**:
- Log full traceback to `benchmarks/logs/{benchmark}_{timestamp}.log`
- Return `None` for failed samples
- Count errors in progress bar
- Skip `None` results during aggregation
- Report error rate in final output

---

## 4. Refactored Runner

### 4.1 Updated `runner.py`

The existing `BenchmarkRunner` gets new internal methods that compose the execution layer:

```python
class BenchmarkRunner:
    def __init__(self, config, alpha):
        self.config = config
        self.alpha = alpha
        self.cache = ResultCache() if config.execution.cache_enabled else None
        self.executor = ParallelExecutor(config.execution.max_concurrency)
        self.retry = RetryWithFallback()

    async def run_one(self, name, dry_run=False):
        benchmark = self.benchmarks[name]()
        samples = await benchmark.load_dataset()

        # Split
        cal_size = int(len(samples) * self.config.calibration_ratio)
        cal_samples = samples[:cal_size]
        test_samples = samples[cal_size:]

        # Calibrate
        engine = ConformalRAGEngine(alpha=self.alpha, scoring=ScoringPipeline(use_llm_judge=False))
        engine.calibrate(...)

        # Evaluate with production execution
        results = await self._evaluate_with_execution(benchmark, test_samples, engine, name)

        # Aggregate and save
        aggregation = await benchmark.aggregate([r for r in results if r is not None])
        self._save_output(name, aggregation, results)
        return aggregation

    async def _evaluate_with_execution(self, benchmark, samples, engine, name):
        """Production-grade evaluation with all 6 fixes."""
        checkpoint = CheckpointManager(self.config.output.checkpoint_dir)
        progress = BenchmarkProgress(len(samples), name, self.alpha)

        # Check cache for existing results
        uncached_samples = []
        cached_results = []
        if self.cache:
            for s in samples:
                cached = self.cache.get(s.id, name, self.alpha)
                if cached:
                    cached_results.append(cached)
                    progress.update(cached, from_cache=True)
                else:
                    uncached_samples.append(s)
        else:
            uncached_samples = samples

        # Run uncached samples with parallelism + retry + error isolation
        async def _run_sample(sample):
            result = await self.retry.execute(
                primary_fn=lambda: benchmark.run_sample(sample, engine),
            )
            if result and self.cache:
                self.cache.put(result, name, self.alpha)
            checkpoint.add(result)
            progress.update(result)
            return result

        new_results = await self.executor.run_all(uncached_samples, _run_sample)
        progress.close()

        return cached_results + [r for r in new_results if r is not None]
```

### 4.2 CLI Additions

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b")
    parser.add_argument("--alpha", "-a", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", "-c", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
```

---

## 5. Configuration

### 5.1 Updated `config.yaml`

```yaml
alpha_values: [0.01, 0.05, 0.10]
calibration_ratio: 0.8
providers:
  - groq
  - together
  - siliconflow

execution:
  max_concurrency: 5
  checkpoint_interval: 25
  cache_enabled: true
  cache_path: benchmarks/cache.db
  checkpoint_dir: benchmarks/checkpoints

retry:
  max_retries: 3
  base_delay: 1.0
  max_delay: 30.0
  backoff_factor: 2.0
  jitter: true

budget_simulation:
  qps_levels: [1, 10, 100]
  duration_seconds: 3600

output:
  results_dir: benchmarks/results
  reports_dir: benchmarks/reports
  logs_dir: benchmarks/logs
```

### 5.2 Updated `BenchmarkConfig`

```python
@dataclass
class ExecutionConfig:
    max_concurrency: int = 5
    checkpoint_interval: int = 25
    cache_enabled: bool = True
    cache_path: str = "benchmarks/cache.db"
    checkpoint_dir: str = "benchmarks/checkpoints"

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True

@dataclass
class BenchmarkConfig:
    alpha_values: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    calibration_ratio: float = 0.8
    providers: list[str] = field(default_factory=lambda: ["groq", "together", "siliconflow"])
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    budget_simulation: BudgetSimConfig = field(default_factory=BudgetSimConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
```

---

## 6. Test Strategy

### 6.1 Unit Tests — `tests/test_benchmark_execution.py`

| Test | What it verifies |
|------|------------------|
| `test_parallel_executor_respects_concurrency` | Semaphore limits concurrent tasks to N |
| `test_parallel_executor_gathers_all_results` | All items processed, none dropped |
| `test_parallel_executor_isolates_errors` | One failure doesn't cancel others |
| `test_retry_succeeds_on_first_attempt` | No retry when primary works |
| `test_retry_retries_on_failure` | Retries N times with backoff |
| `test_retry_uses_fallback` | Falls back to next provider after primary exhaustion |
| `test_retry_raises_after_all_exhausted` | Raises last error when all providers fail |
| `test_cache_put_get_roundtrip` | Serialize → store → retrieve → deserialize |
| `test_cache_hit_avoids_rerun` | Cached result returned, engine not called |
| `test_cache_clear_per_benchmark` | Clear only affects specified benchmark |
| `test_checkpoint_saves_at_interval` | Checkpoint written every N samples |
| `test_checkpoint_resume_loads_latest` | Resume loads latest checkpoint file |
| `test_error_isolation_returns_none` | Failed sample returns None, doesn't crash |
| `test_error_isolation_logs_traceback` | Error logged with full traceback |
| `test_progress_bar_updates` | tqdm bar increments correctly |
| `test_cache_stats_report` | Stats show correct cached count |

### 6.2 Integration Tests — `tests/test_benchmark_production.py`

| Test | What it verifies |
|------|------------------|
| `test_run_one_with_caching` | Full run, cache populated, re-run hits cache |
| `test_run_one_with_concurrency` | Parallel execution completes correctly |
| `test_run_one_survives_api_failure` | Partial failures don't kill benchmark |
| `test_checkpoint_resume_after_crash` | Simulate crash, resume from checkpoint |
| `test_incremental_save_during_run` | Checkpoint files exist mid-run |
| `test_error_rate_in_output` | Failed samples counted in metrics |

### 6.3 Test Patterns

```python
# Parallel executor concurrency test
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
        await asyncio.sleep(0.1)
        async with lock:
            current_concurrent -= 1
        return item

    executor = ParallelExecutor(max_concurrency=3)
    await executor.run_all(list(range(10)), slow_task)
    assert max_concurrent <= 3

# Cache hit test
@pytest.mark.asyncio
async def test_cache_hit_avoids_rerun():
    cache = ResultCache(":memory:")
    result = Result(sample_id="q1", prediction_set=["A"], abstained=False, confidence=0.9)
    cache.put(result, "test", 0.05)

    cached = cache.get("q1", "test", 0.05)
    assert cached is not None
    assert cached.sample_id == "q1"

# Retry fallback test
@pytest.mark.asyncio
async def test_retry_uses_fallback():
    call_count = 0

    async def failing_primary():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("API down")

    async def working_fallback():
        return "fallback_result"

    retry = RetryWithFallback(RetryConfig(max_retries=1))
    result = await retry.execute(failing_primary, [working_fallback])
    assert result == "fallback_result"
    assert call_count == 1  # Tried primary once, then fallback
```

### 6.4 Verification Commands

```bash
# Run all benchmark tests
python -m pytest tests/test_benchmark_*.py -v

# Run production execution tests only
python -m pytest tests/test_benchmark_execution.py tests/test_benchmark_production.py -v

# Run with coverage
python -m pytest tests/test_benchmark_*.py --cov=benchmarks/execution --cov-report=term-missing

# Quick smoke test (dry run)
python -m benchmarks.runner --dry-run --concurrency 1
```

---

## 7. Migration Path

### 7.1 Backward Compatibility

The refactored runner is **backward compatible**:
- Existing `BenchmarkRunner` API unchanged
- New config fields have defaults (no config file change required)
- Cache is opt-in via `config.yaml` or `--no-cache` flag
- Old JSON output format preserved

### 7.2 Rollout Order

1. **Phase A**: Add `benchmarks/execution/` module with all 6 components
2. **Phase B**: Add unit tests for each component
3. **Phase C**: Refactor `runner.py` to use execution layer (internal only)
4. **Phase D**: Add integration tests
5. **Phase E**: Update CLI with new flags

### 7.3 Dependencies

No new external dependencies required:
- `asyncio` — stdlib
- `sqlite3` — stdlib
- `tqdm` — already in `pyproject.toml`
- `json`, `logging`, `traceback` — stdlib

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Semaphore too aggressive for rate limits | High | Default 5, configurable; monitor 429 errors |
| SQLite lock contention under concurrency | Medium | One writer, WAL mode, per-benchmark tables |
| Checkpoint files accumulate on disk | Low | Auto-cleanup: keep last 3 checkpoints |
| Cache grows unbounded | Low | `--clear-cache` flag, TTL-based expiry (future) |
| tqdm conflicts with logging | Low | Use `tqdm.write()` for log output |

---

## 9. Success Criteria

- [ ] 6 production gaps resolved
- [ ] All existing 139 tests still pass
- [ ] New execution layer adds ≥15 unit tests
- [ ] New integration tests ≥6
- [ ] Parallel execution measurable speedup (5x for 5 concurrency)
- [ ] Cache hit rate >80% on re-run
- [ ] Checkpoint resume works after simulated crash
- [ ] Error samples logged with full traceback
- [ ] Progress bar visible during execution
- [ ] No new external dependencies (stdlib + tqdm only)

---

## 10. File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `benchmarks/execution/__init__.py` | **NEW** | Package init, exports |
| `benchmarks/execution/parallel.py` | **NEW** | ParallelExecutor class |
| `benchmarks/execution/retry.py` | **NEW** | RetryWithFallback class |
| `benchmarks/execution/cache.py` | **NEW** | ResultCache class |
| `benchmarks/execution/progress.py` | **NEW** | BenchmarkProgress class |
| `benchmarks/execution/checkpoint.py` | **NEW** | CheckpointManager class |
| `benchmarks/execution/error隔离.py` | **NEW** | ErrorIsolation class |
| `benchmarks/runner.py` | **MODIFY** | Refactor internals to use execution layer |
| `benchmarks/config.py` | **MODIFY** | Add ExecutionConfig, RetryConfig dataclasses |
| `benchmarks/config.yaml` | **MODIFY** | Add execution/retry sections |
| `tests/test_benchmark_execution.py` | **NEW** | Unit tests for execution layer |
| `tests/test_benchmark_production.py` | **NEW** | Integration tests |
