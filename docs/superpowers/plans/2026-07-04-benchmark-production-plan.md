# Benchmark Production System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-grade execution layer to benchmark system — parallel processing, retry/fallback, caching, progress tracking, checkpoint saves, and error isolation.

**Architecture:** New `benchmarks/execution/` module with 6 components. Refactor `runner.py` to use execution layer internally. Backward compatible — new features opt-in via config.

**Tech Stack:** Python 3.10+, asyncio, sqlite3 (stdlib), tqdm

## Global Constraints

- Python 3.10+, <3.13
- All existing 139 tests must continue passing
- No secrets in code — use keyring for API keys
- No new external dependencies (stdlib + tqdm only)
- TDD: RED-GREEN-REFACTOR for every task

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `benchmarks/execution/__init__.py` | **NEW** | Package init, exports |
| `benchmarks/execution/parallel.py` | **NEW** | ParallelExecutor — asyncio.gather + semaphore |
| `benchmarks/execution/retry.py` | **NEW** | RetryWithFallback — exponential backoff + provider chain |
| `benchmarks/execution/cache.py` | **NEW** | ResultCache — SQLite result caching |
| `benchmarks/execution/progress.py` | **NEW** | BenchmarkProgress — tqdm progress bars |
| `benchmarks/execution/checkpoint.py` | **NEW** | CheckpointManager — incremental saves |
| `benchmarks/execution/errors.py` | **NEW** | ErrorIsolation — per-sample error handling |
| `benchmarks/runner.py` | **MODIFY** | Refactor internals to use execution layer |
| `benchmarks/config.py` | **MODIFY** | Add ExecutionConfig, RetryConfig dataclasses |
| `benchmarks/config.yaml` | **MODIFY** | Add execution/retry sections |
| `tests/test_benchmark_execution.py` | **NEW** | Unit tests for execution layer |

---

### Task 1: Create execution package + ParallelExecutor

**Files:**
- Create: `benchmarks/execution/__init__.py`
- Create: `benchmarks/execution/parallel.py`
- Create: `tests/test_benchmark_execution.py`

**Interfaces:**
- Produces: `ParallelExecutor(max_concurrency=5).run_all(items, fn) -> list[T]`

- [ ] **Step 1: Write failing test**

```python
# tests/test_benchmark_execution.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmark_execution.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Create execution package**

```python
# benchmarks/execution/__init__.py
from benchmarks.execution.parallel import ParallelExecutor
from benchmarks.execution.retry import RetryWithFallback, RetryConfig
from benchmarks.execution.cache import ResultCache
from benchmarks.execution.progress import BenchmarkProgress
from benchmarks.execution.checkpoint import CheckpointManager
from benchmarks.execution.errors import ErrorIsolation

__all__ = [
    "ParallelExecutor",
    "RetryWithFallback",
    "RetryConfig",
    "ResultCache",
    "BenchmarkProgress",
    "CheckpointManager",
    "ErrorIsolation",
]
```

- [ ] **Step 4: Implement ParallelExecutor**

```python
# benchmarks/execution/parallel.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmark_execution.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add benchmarks/execution/ tests/test_benchmark_execution.py
git commit -m "feat(benchmarks): add ParallelExecutor with bounded concurrency"
```

---

### Task 2: RetryWithFallback

**Files:**
- Create: `benchmarks/execution/retry.py`
- Modify: `tests/test_benchmark_execution.py`

**Interfaces:**
- Produces: `RetryWithFallback(config).execute(primary_fn, fallback_fns) -> T`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_benchmark_execution.py
from benchmarks.execution.retry import RetryWithFallback, RetryConfig

@pytest.mark.asyncio
async def test_retry_succeeds_on_first_attempt():
    async def primary():
        return "success"

    retry = RetryWithFallback(RetryConfig(max_retries=3))
    result = await retry.execute(primary)
    assert result == "success"

@pytest.mark.asyncio
async def test_retry_retries_on_failure():
    call_count = 0

    async def failing_primary():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("API down")
        return "recovered"

    retry = RetryWithFallback(RetryConfig(max_retries=3, base_delay=0.01))
    result = await retry.execute(failing_primary)
    assert result == "recovered"
    assert call_count == 3

@pytest.mark.asyncio
async def test_retry_uses_fallback():
    call_count = 0

    async def failing_primary():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("API down")

    async def working_fallback():
        return "fallback_result"

    retry = RetryWithFallback(RetryConfig(max_retries=1, base_delay=0.01))
    result = await retry.execute(failing_primary, [working_fallback])
    assert result == "fallback_result"

@pytest.mark.asyncio
async def test_retry_raises_after_all_exhausted():
    async def always_fails():
        raise ConnectionError("permanent failure")

    retry = RetryWithFallback(RetryConfig(max_retries=2, base_delay=0.01))
    with pytest.raises(ConnectionError):
        await retry.execute(always_fails)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmark_execution.py -k retry -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement RetryWithFallback**

```python
# benchmarks/execution/retry.py
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True


class RetryWithFallback:
    """Exponential backoff retry with provider fallback chain."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    async def execute(
        self,
        primary_fn: Callable[[], Awaitable],
        fallback_fns: Optional[list[Callable[[], Awaitable]]] = None,
    ) -> any:
        """Try primary, retry with backoff, then fallback providers."""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return await primary_fn()
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self._compute_delay(attempt)
                    await asyncio.sleep(delay)

        if fallback_fns:
            for fallback in fallback_fns:
                try:
                    return await fallback()
                except Exception:
                    continue

        raise last_error

    def _compute_delay(self, attempt: int) -> float:
        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)
        delay = min(delay, self.config.max_delay)
        if self.config.jitter:
            delay *= random.uniform(0.5, 1.0)
        return delay
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmark_execution.py -k retry -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/execution/retry.py tests/test_benchmark_execution.py
git commit -m "feat(benchmarks): add RetryWithFallback with exponential backoff"
```

---

### Task 3: ResultCache

**Files:**
- Create: `benchmarks/execution/cache.py`
- Modify: `tests/test_benchmark_execution.py`

**Interfaces:**
- Produces: `ResultCache(db_path).get(sample_id, benchmark, alpha) -> Result | None`
- Produces: `ResultCache(db_path).put(result, benchmark, alpha) -> None`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_benchmark_execution.py
from benchmarks.execution.cache import ResultCache
from benchmarks.base import Result

@pytest.fixture
def cache():
    return ResultCache(":memory:")

def test_cache_put_get_roundtrip(cache):
    result = Result(
        sample_id="q1",
        prediction_set=["answer"],
        abstained=False,
        confidence=0.9,
        metadata={"type": "bridge"},
    )
    cache.put(result, "test_benchmark", 0.05)
    cached = cache.get("q1", "test_benchmark", 0.05)
    assert cached is not None
    assert cached.sample_id == "q1"
    assert cached.prediction_set == ["answer"]
    assert cached.abstained is False

def test_cache_miss_returns_none(cache):
    cached = cache.get("nonexistent", "test", 0.05)
    assert cached is None

def test_cache_stats(cache):
    result = Result(sample_id="q1", prediction_set=[], abstained=True, confidence=0.5)
    cache.put(result, "test", 0.05)
    stats = cache.stats("test", 0.05)
    assert stats["cached_samples"] == 1

def test_cache_clear_per_benchmark(cache):
    r1 = Result(sample_id="q1", prediction_set=[], abstained=True, confidence=0.5)
    r2 = Result(sample_id="q2", prediction_set=[], abstained=False, confidence=0.9)
    cache.put(r1, "bench_a", 0.05)
    cache.put(r2, "bench_b", 0.05)
    cache.clear("bench_a")
    assert cache.get("q1", "bench_a", 0.05) is None
    assert cache.get("q2", "bench_b", 0.05) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmark_execution.py -k cache -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement ResultCache**

```python
# benchmarks/execution/cache.py
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from benchmarks.base import Result


class ResultCache:
    """SQLite-based result cache."""

    def __init__(self, db_path: str = "benchmarks/cache.db"):
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
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

    def get(self, sample_id: str, benchmark: str, alpha: float) -> Optional[Result]:
        row = self._conn.execute(
            "SELECT result_json FROM results WHERE sample_id=? AND benchmark=? AND alpha=?",
            (sample_id, benchmark, alpha),
        ).fetchone()
        if row:
            data = json.loads(row[0])
            return Result(**data)
        return None

    def put(self, result: Result, benchmark: str, alpha: float):
        data = {
            "sample_id": result.sample_id,
            "prediction_set": result.prediction_set,
            "abstained": result.abstained,
            "confidence": result.confidence,
            "metadata": result.metadata,
        }
        self._conn.execute(
            "INSERT OR REPLACE INTO results (sample_id, benchmark, alpha, result_json) VALUES (?, ?, ?, ?)",
            (result.sample_id, benchmark, alpha, json.dumps(data)),
        )
        self._conn.commit()

    def has(self, sample_id: str, benchmark: str, alpha: float) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM results WHERE sample_id=? AND benchmark=? AND alpha=?",
            (sample_id, benchmark, alpha),
        ).fetchone()
        return row is not None

    def stats(self, benchmark: str, alpha: float) -> dict:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE benchmark=? AND alpha=?",
            (benchmark, alpha),
        ).fetchone()[0]
        return {"cached_samples": total}

    def clear(self, benchmark: Optional[str] = None):
        if benchmark:
            self._conn.execute("DELETE FROM results WHERE benchmark=?", (benchmark,))
        else:
            self._conn.execute("DELETE FROM results")
        self._conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmark_execution.py -k cache -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/execution/cache.py tests/test_benchmark_execution.py
git commit -m "feat(benchmarks): add ResultCache with SQLite persistence"
```

---

### Task 4: ErrorIsolation

**Files:**
- Create: `benchmarks/execution/errors.py`
- Modify: `tests/test_benchmark_execution.py`

**Interfaces:**
- Produces: `ErrorIsolation.run_sample(fn, sample_id) -> Result | None`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_benchmark_execution.py
from benchmarks.execution.errors import ErrorIsolation
from benchmarks.base import Result

@pytest.mark.asyncio
async def test_error_isolation_returns_none():
    async def failing_fn():
        raise ValueError("sample processing failed")

    result = await ErrorIsolation.run_sample(failing_fn, "q1")
    assert result is None

@pytest.mark.asyncio
async def test_error_isolation_returns_result():
    async def success_fn():
        return Result(sample_id="q1", prediction_set=["a"], abstained=False, confidence=0.9)

    result = await ErrorIsolation.run_sample(success_fn, "q1")
    assert result is not None
    assert result.sample_id == "q1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmark_execution.py -k error -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement ErrorIsolation**

```python
# benchmarks/execution/errors.py
from __future__ import annotations

import logging
import traceback
from typing import Callable, Awaitable, Optional

from benchmarks.base import Result

logger = logging.getLogger(__name__)


class ErrorIsolation:
    """Per-sample error isolation with structured logging."""

    @staticmethod
    async def run_sample(
        fn: Callable[[], Awaitable[Optional[Result]]],
        sample_id: str,
    ) -> Optional[Result]:
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

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmark_execution.py -k error -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/execution/errors.py tests/test_benchmark_execution.py
git commit -m "feat(benchmarks): add ErrorIsolation for per-sample error handling"
```

---

### Task 5: CheckpointManager

**Files:**
- Create: `benchmarks/execution/checkpoint.py`
- Modify: `tests/test_benchmark_execution.py`

**Interfaces:**
- Produces: `CheckpointManager(dir, interval).add(result) -> None`
- Produces: `CheckpointManager(dir).finalize() -> list[Result]`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_benchmark_execution.py
import tempfile
from benchmarks.execution.checkpoint import CheckpointManager

def test_checkpoint_saves_at_interval():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir, interval=5)
        for i in range(12):
            result = Result(
                sample_id=f"q{i}",
                prediction_set=[],
                abstained=False,
                confidence=0.9,
            )
            mgr.add(result)
        checkpoints = list(Path(tmpdir).glob("checkpoint_*.json"))
        assert len(checkpoints) >= 2

def test_checkpoint_finalize_loads():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir, interval=3)
        for i in range(7):
            result = Result(
                sample_id=f"q{i}",
                prediction_set=[],
                abstained=False,
                confidence=0.9,
            )
            mgr.add(result)
        loaded = mgr.finalize()
        assert len(loaded) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmark_execution.py -k checkpoint -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement CheckpointManager**

```python
# benchmarks/execution/checkpoint.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from benchmarks.base import Result


class CheckpointManager:
    """Incremental saves to periodic JSON snapshots."""

    def __init__(self, checkpoint_dir: str, interval: int = 25):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self._buffer: List[Result] = []
        self._count = 0

    def add(self, result: Result):
        self._buffer.append(result)
        self._count += 1
        if self._count % self.interval == 0:
            self._flush()

    def _flush(self):
        checkpoint_path = self.dir / f"checkpoint_{self._count}.json"
        data = [self._result_to_dict(r) for r in self._buffer]
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        self._buffer.clear()

    def finalize(self) -> List[Result]:
        if self._buffer:
            self._flush()
        checkpoints = sorted(self.dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return []
        with open(checkpoints[-1]) as f:
            data = json.load(f)
        return [Result(**d) for d in data]

    def _result_to_dict(self, r: Result) -> dict:
        return {
            "sample_id": r.sample_id,
            "prediction_set": r.prediction_set,
            "abstained": r.abstained,
            "confidence": r.confidence,
            "metadata": r.metadata,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmark_execution.py -k checkpoint -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/execution/checkpoint.py tests/test_benchmark_execution.py
git commit -m "feat(benchmarks): add CheckpointManager for incremental saves"
```

---

### Task 6: BenchmarkProgress

**Files:**
- Create: `benchmarks/execution/progress.py`
- Modify: `tests/test_benchmark_execution.py`

**Interfaces:**
- Produces: `BenchmarkProgress(total, name, alpha).update(result, from_cache) -> None`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_benchmark_execution.py
from benchmarks.execution.progress import BenchmarkProgress

def test_progress_bar_updates():
    progress = BenchmarkProgress(total=10, benchmark_name="test", alpha=0.05)
    for i in range(10):
        result = Result(sample_id=f"q{i}", prediction_set=[], abstained=False, confidence=0.9)
        progress.update(result)
    assert progress.pbar.n == 10
    progress.close()

def test_progress_bar_counts_errors():
    progress = BenchmarkProgress(total=5, benchmark_name="test", alpha=0.05)
    progress.update(None)
    progress.update(None)
    progress.update(Result(sample_id="q1", prediction_set=[], abstained=False, confidence=0.9))
    assert progress.errors == 2
    progress.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmark_execution.py -k progress -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement BenchmarkProgress**

```python
# benchmarks/execution/progress.py
from __future__ import annotations

from typing import Optional
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

    def update(self, result: Optional[Result], from_cache: bool = False):
        if result is None:
            self.errors += 1
        if from_cache:
            self.cached += 1
        self.pbar.update(1)
        self.pbar.set_postfix({"errors": self.errors, "cached": self.cached})

    def close(self):
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmark_execution.py -k progress -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/execution/progress.py tests/test_benchmark_execution.py
git commit -m "feat(benchmarks): add BenchmarkProgress with tqdm tracking"
```

---

### Task 7: Refactor Runner to Use Execution Layer

**Files:**
- Modify: `benchmarks/runner.py`
- Modify: `benchmarks/config.py`
- Modify: `benchmarks/config.yaml`
- Modify: `tests/test_benchmark_framework.py`

**Interfaces:**
- Consumes: All execution layer components
- Produces: Refactored `BenchmarkRunner` with production execution

- [ ] **Step 1: Update config with execution settings**

```python
# Append to benchmarks/config.py
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
```

- [ ] **Step 2: Update config.yaml**

```yaml
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
```

- [ ] **Step 3: Refactor runner.py**

```python
# Replace runner.py with refactored version
# Key changes:
# 1. Import execution layer components
# 2. Add _evaluate_with_execution method
# 3. Use ParallelExecutor, RetryWithFallback, ResultCache, etc.
# 4. Keep backward-compatible API
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_benchmark_*.py -v`
Expected: All tests pass (existing + new)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/runner.py benchmarks/config.py benchmarks/config.yaml
git commit -m "refactor(benchmarks): integrate execution layer into runner"
```

---

### Task 8: Verify Production Execution

**Files:**
- None (execution only)

- [ ] **Step 1: Run dry-run**

Run: `python -m benchmarks.runner --dry-run --concurrency 2`
Expected: Loads datasets, prints sample counts, no API calls

- [ ] **Step 2: Run HotpotQA with 5 samples**

Run: `python -m benchmarks.runner --benchmark hotpotqa --alpha 0.05 --concurrency 2`
Expected: Progress bar, JSON results in `benchmarks/results/`

- [ ] **Step 3: Verify cache populated**

Run: `ls -la benchmarks/cache.db`
Expected: File exists

- [ ] **Step 4: Re-run HotpotQA (cache hit)**

Run: `python -m benchmarks.runner --benchmark hotpotqa --alpha 0.05`
Expected: Much faster, progress bar shows cached=45

- [ ] **Step 5: Commit results**

```bash
git add benchmarks/results/
git commit -m "docs: add HotpotQA benchmark results with production execution"
```
