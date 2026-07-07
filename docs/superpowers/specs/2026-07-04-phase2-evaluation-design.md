# Phase 2: Evaluation Framework — Design Spec

**Date**: 2026-07-04
**Status**: Approved
**Depends on**: Phase 1 (Core) — 108 tests passing, real NLI/LLM scoring, hybrid retrieval

---

## 1. Goal

Build a plugin-based evaluation framework that runs 5 benchmarks, proves FDR ≤ α on real data, profiles latency with per-component traces, and simulates cost at production QPS levels.

---

## 2. Architecture

```
benchmarks/
├── __init__.py              # Registry + decorators
├── base.py                  # BaseBenchmark ABC + dataclasses
├── runner.py                # Orchestrator + CLI entry point
├── hotpotqa.py              # HotpotQA benchmark
├── ragtruth.py              # RAGTruth benchmark
├── financebench.py          # FinanceBench benchmark
├── halueval.py              # HaluEval benchmark
├── multihoprag.py           # MultiHop-RAG benchmark
├── profiling/
│   ├── __init__.py
│   ├── latency.py           # Per-component trace
│   └── cost.py              # Budget simulation
├── config.yaml              # Shared configuration
├── results/                 # JSON output (gitignored)
└── reports/                 # Markdown reports
```

### 2.1 Plugin Architecture

**BaseBenchmark ABC:**
```python
from abc import ABC, abstractmethod

class BaseBenchmark(ABC):
    name: str
    description: str

    @abstractmethod
    async def load_dataset(self) -> list[Sample]: ...

    @abstractmethod
    async def run_sample(self, sample: Sample, engine: ConformalEngine) -> Result: ...

    @abstractmethod
    async def aggregate(self, results: list[Result]) -> Aggregation: ...
```

**Registry:**
```python
@register_benchmark
class HotpotQABenchmark(BaseBenchmark):
    ...
```

Runner discovers all registered benchmarks via `__init__.py` imports.

### 2.2 Data Flow

```
Benchmark Plugin → Runner → Profiler
     │              │          │
     ▼              ▼          ▼
  load_dataset  calibrate   trace_latency
  (80/20 split) evaluate    track_cost
                 │          │
                 ▼          ▼
            ConformalEngine  Results JSON + Markdown
```

**Pipeline per benchmark:**
1. `load_dataset()` — fetch + split 80/20 (calibration/test)
2. `calibrate()` — run engine on calibration set, store threshold
3. `evaluate()` — run engine on test set, collect verdicts
4. `profile()` — wrap each step with latency tracer + cost tracker
5. `aggregate()` — compute FDR, coverage, abstention rate, latency percentiles
6. `report()` — write JSON + Markdown

---

## 3. Benchmarks

| Benchmark | Dataset | Tests | Source |
|-----------|---------|-------|--------|
| HotpotQA | `hotpot_qa` | Multi-hop reasoning FDR | HuggingFace |
| RAGTruth | `ragtruth` | Real LLM hallucination FDR | HuggingFace |
| FinanceBench | `PatronusAI/financebench` | Domain-specific FDR | HuggingFace |
| HaluEval | `liamcripwell/halueval` | RLHF hallucination FDR | HuggingFace |
| MultiHop-RAG | `MqaDQ/MultiHop-RAG` | Complex retrieval FDR | HuggingFace |

**Per-benchmark config:**
- Fetch dataset from HuggingFace (streaming for large datasets)
- Split 80/20 (calibration/test)
- Run conformal engine with α ∈ {0.01, 0.05, 0.10}
- Record all traces + costs
- Generate independent JSON + Markdown

**Shared across all:**
- Same `ConformalEngine` instance
- Same provider fallback chain (Groq → Together → SiliconFlow)
- Same latency tracer + cost tracker
- Same report template

---

## 4. Latency Profiling

### 4.1 Per-Component Trace

```python
@dataclass
class LatencyTrace:
    retrieval_ms: float      # Vector search + BM25 + RRF fusion
    nli_scoring_ms: float    # DeBERTa inference
    llm_judge_ms: float      # LLM API call (if triggered)
    confidence_ms: float     # Conformal threshold computation
    total_ms: float          # End-to-end
```

- Wraps `engine.predict()` with timing hooks at each stage
- Stores traces per sample, aggregates to p50/p95/p99
- Waterfall: Markdown table showing component breakdown

### 4.2 Waterfall Visualization

```markdown
| Component        | p50    | p95    | p99    | % of Total |
|------------------|--------|--------|--------|------------|
| Retrieval        | 45ms   | 120ms  | 180ms  | 2.1%       |
| NLI Scoring      | 1100ms | 1400ms | 1600ms | 52.4%      |
| LLM Judge        | 800ms  | 1200ms | 1500ms | 38.1%      |
| Conformal        | 15ms   | 25ms   | 35ms   | 0.7%       |
| **Total**        | **2100ms** | **3200ms** | **4100ms** | **100%** |
```

---

## 5. Cost Analysis

### 5.1 Per-Query Cost Tracking

```python
@dataclass
class CostTrace:
    provider: str            # groq / together / siliconflow
    input_tokens: int
    output_tokens: int
    cost_usd: float          # computed from provider pricing
    cached: bool             # was this a cache hit?
```

### 5.2 Budget Simulation

Simulate cost at different QPS levels using **actual recorded latency distributions** from benchmark runs. Example output (derived from real measurements):

| QPS | Cost/Hour | Cost/Day | Cost/Month |
|-----|-----------|----------|------------|
| 1   | $77.76    | $1,866   | $55,987    |
| 10  | $777.60   | $18,662  | $559,872   |
| 100 | $7,776    | $186,624 | $5,598,720 |

Note: Actual numbers will be computed from per-query costs and latency distributions recorded during benchmark execution.

---

## 6. Output Format

### 6.1 JSON Structure

`benchmarks/results/{benchmark}_{timestamp}.json`:
```json
{
  "benchmark": "hotpotqa",
  "timestamp": "2026-07-04T12:00:00Z",
  "config": {
    "alpha": 0.05,
    "calibration_size": 800,
    "test_size": 200,
    "providers": ["groq", "together", "siliconflow"]
  },
  "metrics": {
    "fdr": 0.032,
    "coverage": 0.94,
    "abstention_rate": 0.18,
    "total_samples": 200
  },
  "latency": {
    "retrieval_p50_ms": 45,
    "retrieval_p95_ms": 120,
    "retrieval_p99_ms": 180,
    "nli_p50_ms": 1100,
    "nli_p95_ms": 1400,
    "llm_judge_p50_ms": 800,
    "total_p50_ms": 2100,
    "total_p95_ms": 3200,
    "total_p99_ms": 4100
  },
  "cost": {
    "total_usd": 4.32,
    "avg_per_query_usd": 0.0216,
    "by_provider": { "groq": 1.2, "together": 2.1, "siliconflow": 1.02 }
  },
  "budget_simulation": {
    "1_qps": { "cost_per_hour": 77.76, "cost_per_day": 1866.24 },
    "10_qps": { "cost_per_hour": 777.60, "cost_per_day": 18662.40 },
    "100_qps": { "cost_per_hour": 7776.00, "cost_per_day": 186624.00 }
  },
  "samples": [ ... ]
}
```

### 6.2 Markdown Report

`benchmarks/reports/{benchmark}_{timestamp}.md`:
- Executive summary (FDR, coverage, key findings)
- Latency breakdown table + waterfall
- Cost analysis + budget projections
- Per-sample detail (abstained vs certified)
- Comparison across α values (0.01, 0.05, 0.10)

---

## 7. Implementation Phases

### Sub-phase 2.1: Framework
- `base.py` — BaseBenchmark ABC, Sample/Result/Aggregation dataclasses
- `__init__.py` — Registry + `@register_benchmark` decorator
- `profiling/latency.py` — `@trace_latency` decorator
- `profiling/cost.py` — `@track_cost` decorator
- `runner.py` — Orchestrator + CLI entry point
- Tests for framework itself

### Sub-phase 2.2: HotpotQA + RAGTruth
- `hotpotqa.py` — Plugin implementation
- `ragtruth.py` — Plugin implementation
- Run both, verify FDR ≤ α
- First Markdown report

### Sub-phase 2.3: FinanceBench + HaluEval + MultiHop-RAG
- `financebench.py`, `halueval.py`, `multihoprag.py`
- Run all three, verify FDR ≤ α

### Sub-phase 2.4: Profiling + Reports
- Budget simulation (1/10/100 QPS)
- Waterfall visualization
- Aggregate report across all 5 benchmarks
- Final comparison table

---

## 8. Dependencies

```toml
[project.optional-dependencies]
benchmarks = [
    "datasets>=2.18",      # HuggingFace datasets
    "matplotlib>=3.8",     # Plots for reports
    "pandas>=2.2",         # Data aggregation
]
```

**Config** (`benchmarks/config.yaml`):
```yaml
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

**CLI:**
```bash
python -m benchmarks.runner                        # Run all
python -m benchmarks.runner --benchmark hotpotqa   # Run one
python -m benchmarks.runner --alpha 0.05           # Override alpha
python -m benchmarks.runner --dry-run              # Load only, no eval
```

---

## 9. Testing Strategy

**Framework tests:**
- Registry discovers `@register_benchmark` plugins
- Latency tracer captures per-component timing
- Cost tracker computes correct USD from token counts
- Runner discovers all plugins, runs them, collects results

**Benchmark tests:**
- Each plugin loads dataset correctly
- 80/20 split produces valid calibration/test sets
- Engine runs on samples without crashing
- FDR ≤ α is verified (assertion in test)
- Results JSON is valid against schema

**Profiling tests:**
- Budget simulation produces reasonable projections
- Markdown report contains required sections
- JSON output matches schema

**Integration test:**
- Full pipeline: load → calibrate → evaluate → profile → report
- Verifies end-to-end on a tiny subset (50 samples)

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| HuggingFace datasets too large to download | High | Stream loading, use `split='train[:1000]'` for calibration |
| LLM API rate limits during benchmark runs | High | Exponential backoff, per-provider quotas, cache results |
| FDR exceeds α on some benchmarks | Medium | Log it, analyze why, adjust scoring — this is a finding, not a failure |
| Latency tracer overhead skews results | Medium | Use `time.perf_counter_ns()`, negligible overhead |
| Budget simulation unrealistic | Low | Use actual recorded latency distributions, not synthetic |
| Plugin registry fails to discover benchmarks | Low | Explicit `__all__` in `__init__.py`, test discovery separately |

---

## 11. Success Criteria

- [ ] All 5 benchmarks run end-to-end
- [ ] FDR ≤ α verified on each benchmark
- [ ] Latency p50/p95/p99 reported per component
- [ ] Cost tracked per provider, per query
- [ ] Budget simulation at 1/10/100 QPS
- [ ] JSON output for each benchmark
- [ ] Markdown report for each benchmark
- [ ] Aggregate comparison table across all benchmarks
- [ ] All tests pass
