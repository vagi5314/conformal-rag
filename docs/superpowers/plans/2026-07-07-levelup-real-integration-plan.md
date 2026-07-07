# Conformal RAG Level-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fabricated/stubbed behavior with real, integrated, correctly-labeled functionality across 6 audited gaps, keeping the suite green and adding behavior-asserting tests.

**Architecture:** Each gap is an independent, testable task. The engine gains optional `FDRMonitor`/`OnlineConformal` wiring; the client SDK becomes a real retrieve+risk-check wrapper; `online.py` becomes true quantile-level ACI; the API gains a `/feedback` loop so monitoring is live; the CI becomes a finite-sample conformal bound; and a reproducible eval script substantiates (or retracts) the headline numbers.

**Tech Stack:** Python 3.11/3.12, pytest + pytest-asyncio, FastAPI, existing `conformal_rag` modules (`engine`, `scoring`, `retrieval`, `online`, `monitoring`, `domain`, `guards`, `models`, `client`, `api`).

## Global Constraints

- scikit-learn pinned to `==1.4.2` and scipy pinned to `==1.14.0` in `pyproject.toml` (AppLocker hash-blocks newer binaries on this machine). Do NOT bump these.
- Run tests with `set PYTHONPATH=src` then `python -m pytest tests/ -q`.
- TDD: every task writes a failing test first, runs it, implements, re-runs to pass, then commits.
- No new third-party dependencies.
- Keep all 161 existing tests passing; add new tests that assert real behavior (no placeholder/empty-response hacks).
- `ConformalResult` fields (from `models.py`): `query, response, risk_score, risk_bound, abstained, confidence_interval (tuple[float,float]), scoring_breakdown, domain, latency_ms, timestamp`.
- `Document` has `.text`, `.metadata`, `.doc_id`, `.source`, `.domain`.
- `HybridRetriever.index(documents)` expects objects with `.text`; `retrieve(query)` returns `list[str]`.

---

### Task 1: Real client SDK (Gap 1)

**Files:**
- Modify: `src/conformal_rag/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: `ConformalRAGEngine(alpha)`, `HybridRetriever(qdrant_url)`, `Document`.
- Produces: `ConformalRAG.index(documents)`, `ConformalRAG.retrieve(query)`, `ConformalRAG.check(query, response, context=None, domain=None)`, `ConformalRAG.query(query, response, ...)` (honest — requires the generated `response`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client.py
import pytest
from conformal_rag.client import ConformalRAG
from conformal_rag.models import ScoringBreakdown


class FakeScoring:
    def __init__(self, score=0.9):
        self._score = score
    def score(self, query, context, response):
        return ScoringBreakdown(
            nli_score=self._score, final_score=self._score,
            nli_latency_ms=1.0, used_llm_judge=False,
        )


def _client():
    c = ConformalRAG(alpha=0.05)
    c.engine.scoring = FakeScoring(score=0.9)
    return c


def test_index_builds_retriever_and_stores_docs():
    c = _client()
    c.index(["Python is a programming language.", "The sky is blue."])
    assert c.retriever is not None
    assert len(c._documents) == 2


def test_retrieve_returns_relevant_context():
    c = _client()
    c.index(["Python is a programming language created by Guido.", "Batteries included."])
    ctx = c.retrieve("What is Python?")
    assert any("Python" in d for d in ctx)


def test_check_returns_result_with_real_response():
    c = _client()
    c.index(["Python is a programming language."])
    res = c.check("What is Python?", response="Python is a programming language.")
    assert res.risk_score == 0.9
    assert res.abstained is False


def test_query_requires_response():
    c = _client()
    with pytest.raises(ValueError):
        c.query("What is Python?")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set PYTHONPATH=src && python -m pytest tests/test_client.py -v`
Expected: FAIL (`index` is a no-op / `query` returns placeholder, `retrieve` missing).

- [ ] **Step 3: Write minimal implementation**

Replace the body of `src/conformal_rag/client.py` with:

```python
"""Python client SDK for conformal-rag."""

from __future__ import annotations

from typing import List, Optional

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.domain import DomainManager
from conformal_rag.models import ConformalResult, CalibrationStats, Document
from conformal_rag.retrieval import HybridRetriever


class ConformalRAG:
    """Python client for conformal RAG.

    The client retrieves context from an indexed corpus and risk-checks a
    *provided* LLM response. It does not generate responses — that is the
    caller's RAG pipeline.

    Usage:
        rag = ConformalRAG(alpha=0.05)
        rag.index(documents)
        context = rag.retrieve("What is Python?")
        response = my_llm(context)
        result = rag.check("What is Python?", response=response)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        vector_store: str = "qdrant",
        qdrant_url: Optional[str] = None,
        llm_provider: str = "groq",
        scoring: str = "hybrid",
    ):
        self.alpha = alpha
        self.vector_store = vector_store
        self.qdrant_url = qdrant_url
        self.llm_provider = llm_provider
        self.scoring = scoring

        self.engine = ConformalRAGEngine(alpha=alpha)
        self.domain_manager = DomainManager()
        self.domains = self.domain_manager.domains
        self.retriever: Optional[HybridRetriever] = None
        self._documents: List[Document] = []

    def index(self, documents: list, chunk_config: Optional[dict] = None) -> None:
        """Index documents for retrieval (in-memory by default, Qdrant if configured)."""
        docs: List[Document] = []
        for d in documents:
            if isinstance(d, Document):
                docs.append(d)
            elif isinstance(d, dict):
                docs.append(Document(text=d.get("text", ""), metadata=d.get("metadata")))
            elif isinstance(d, str):
                docs.append(Document(text=d))
            else:
                raise TypeError(f"Unsupported document type: {type(d)!r}")
        self._documents = docs
        self.retriever = HybridRetriever(qdrant_url=self.qdrant_url)
        self.retriever.index(docs)

    def retrieve(self, query: str, final_k: Optional[int] = None) -> List[str]:
        """Retrieve fused context for a query. Returns [] if not indexed."""
        if self.retriever is None:
            return []
        return self.retriever.retrieve(query)

    def check(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> ConformalResult:
        """Risk-check a generated response against retrieved (or provided) context."""
        if context is None:
            context = self.retrieve(query)
        context_str = "\n".join(context) if context else ""
        return self.engine.predict(
            query=query,
            context=context_str,
            response=response,
            domain=domain,
        )

    def query(
        self,
        query: str,
        response: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[List[str]] = None,
    ) -> ConformalResult:
        """Honest convenience wrapper. Requires the LLM `response` to risk-check.

        If only a query is given, call retrieve() first, generate a response in
        your pipeline, then call check().
        """
        if response is None:
            raise ValueError(
                "ConformalRAG.query requires the generated `response` to risk-check. "
                "Pass response=... or use retrieve() then check()."
            )
        return self.check(query, response, context=context, domain=domain)

    def calibrate(self, domain: str = "default", alpha: Optional[float] = None,
                  data: Optional[list] = None) -> CalibrationStats:
        if alpha is not None:
            self.engine.alpha = alpha
        if data is None:
            data = []
        return self.engine.calibrate(
            queries=[d.get("query", "") for d in data],
            contexts=[d.get("context", "") for d in data],
            responses=[d.get("response", "") for d in data],
            labels=[d.get("label", 0) for d in data],
            domain=domain,
        )

    def register_domain(self, domain: str, alpha: float = 0.05) -> None:
        self.domain_manager.register_domain(domain, alpha)

    def monitor_drift(self, domain: str):
        return self.domain_manager.monitor_drift(domain)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set PYTHONPATH=src && python -m pytest tests/test_client.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/conformal_rag/client.py tests/test_client.py
git commit -m "feat(client): real index/retrieve/check; drop placeholder context"
```

---

### Task 2: Wire FDRMonitor + OnlineConformal into the engine (Gap 2)

**Files:**
- Modify: `src/conformal_rag/engine.py`
- Test: `tests/test_engine_monitoring.py` (new)

**Interfaces:**
- Consumes: `FDRMonitor`, `OnlineConformal` (import lazily/optionally), `ScoringBreakdown`.
- Produces: `ConformalRAGEngine(fdr_monitor=None, online=None)`, `ConformalRAGEngine.record(query, context, response, actual_hallucination)`, and adaptive threshold sync when `online` is set.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine_monitoring.py
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.monitoring import FDRMonitor
from conformal_rag.online import OnlineConformal
from conformal_rag.models import ScoringBreakdown


class FakeScoring:
    def __init__(self, score=0.9):
        self._score = score
    def score(self, query, context, response):
        return ScoringBreakdown(nli_score=self._score, final_score=self._score,
                                nli_latency_ms=1.0, used_llm_judge=False)


def test_record_populates_fdr_monitor():
    mon = FDRMonitor()
    eng = ConformalRAGEngine(alpha=0.05, scoring=FakeScoring(0.9), fdr_monitor=mon)
    eng._threshold = 0.5
    eng.record("q", "c", "r", actual_hallucination=False)
    m = mon.get_metrics()
    assert m["total_predictions"] == 1


def test_online_adjusts_threshold():
    on = OnlineConformal(alpha=0.05, eta=0.05)
    eng = ConformalRAGEngine(alpha=0.05, scoring=FakeScoring(0.9), online=on)
    eng._threshold = 0.5
    # Many correct samples scored above threshold -> miscoverage -> threshold rises
    for _ in range(20):
        eng.record("q", "c", "r", actual_hallucination=False)
    assert eng._threshold != 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set PYTHONPATH=src && python -m pytest tests/test_engine_monitoring.py -v`
Expected: FAIL (`fdr_monitor`/`online`/`record` do not exist).

- [ ] **Step 3: Write minimal implementation**

In `src/conformal_rag/engine.py`:
- Add imports: `from conformal_rag.monitoring import FDRMonitor` and `from conformal_rag.online import OnlineConformal`.
- Change `__init__` signature to:

```python
    def __init__(
        self,
        alpha: float = 0.05,
        scoring: Optional[ScoringPipeline] = None,
        weights: tuple[float, float] = (0.6, 0.4),
        fdr_monitor: Optional["FDRMonitor"] = None,
        online: Optional["OnlineConformal"] = None,
    ):
        self.alpha = clamp_alpha(alpha)
        self.scoring = scoring or ScoringPipeline(weights=weights)
        self.weights = weights
        self.fdr_monitor = fdr_monitor or FDRMonitor(alpha=self.alpha)
        self.online = online
        self._threshold: Optional[float] = None
        self._calibration_scores: List[float] = []
        self._calibration_labels: List[int] = []
        self._domain: str = "default"
```

- Add a `record` method (place after `predict`):

```python
    def record(
        self,
        query: str,
        context: str,
        response: str,
        actual_hallucination: bool,
        domain: Optional[str] = None,
    ) -> None:
        """Feed a labeled observation into monitoring and (optional) online adaptation."""
        breakdown = self.scoring.score(query, context, response)
        score = breakdown.final_score
        threshold = self._threshold if self._threshold is not None else 0.5
        predicted_hallucination = score > threshold

        self.fdr_monitor.record(
            predicted_hallucination=predicted_hallucination,
            actual_hallucination=actual_hallucination,
        )
        if self.online is not None:
            self.online.update(score, is_correct=not actual_hallucination)
            self._threshold = self.online.threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set PYTHONPATH=src && python -m pytest tests/test_engine_monitoring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conformal_rag/engine.py tests/test_engine_monitoring.py
git commit -m "feat(engine): wire FDRMonitor + OnlineConformal; add record()"
```

---

### Task 3: Correct ACI in online.py (Gap 3)

**Files:**
- Modify: `src/conformal_rag/online.py`
- Test: `tests/test_online_aci.py` (new)

**Interfaces:**
- Consumes: numpy.
- Produces: `OnlineConformal(alpha, window_size, eta, initial_level)`, `update(score, is_correct)`, `threshold` property, `get_stats()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_online_aci.py
import numpy as np
from conformal_rag.online import OnlineConformal


def test_aci_keeps_rejection_rate_near_alpha():
    rng = np.random.default_rng(0)
    on = OnlineConformal(alpha=0.1, window_size=200, eta=0.02, initial_level=0.9)
    # exchangeable stream: correct -> low score, hallucination -> high score
    n = 4000
    miscoverage = 0
    for _ in range(n):
        is_halluc = rng.random() < 0.1
        score = rng.normal(0.9 if is_halluc else 0.1, 0.1)
        on.update(score, is_correct=not is_halluc)
        if score > on.threshold and not is_halluc:
            miscoverage += 1
    rate = miscoverage / n
    assert abs(rate - 0.1) < 0.05


def test_threshold_in_bounds():
    on = OnlineConformal()
    for s in [0.2, 0.8, 0.5, 0.9]:
        on.update(s, is_correct=True)
    assert 0.0 <= on.threshold <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set PYTHONPATH=src && python -m pytest tests/test_online_aci.py -v`
Expected: FAIL (old heuristic doesn't target alpha; threshold may go out of bounds).

- [ ] **Step 3: Write minimal implementation**

Replace `src/conformal_rag/online.py` with:

```python
"""Online Adaptive Conformal Inference (ACI), quantile-level.

Implements the Gibbs & Candès (2021) ACI idea: maintain a running quantile
level lambda_t of the nonconformity-score window and nudge lambda_t so that
the long-run miscoverage approaches the target alpha. Under exchangeability
this yields approximate alpha-level control (heuristic convergence, not a
hard finite-sample guarantee).
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class OnlineConformal:
    def __init__(
        self,
        alpha: float = 0.05,
        window_size: int = 500,
        eta: float = 0.01,
        initial_level: float = 0.9,
        initial_threshold: float = 0.5,
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.eta = eta
        self.level = float(initial_level)
        self._threshold = float(initial_threshold)
        self.window: deque = deque(maxlen=window_size)
        self._n_updates = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(value)

    def update(self, score: float, is_correct: bool) -> None:
        self.window.append(float(score))
        self._n_updates += 1

        # Miscoverage under the CURRENT threshold.
        rejected = score > self._threshold
        miscoverage = 1.0 if (rejected and is_correct) else 0.0

        # ACI: move the quantile level toward satisfying target miscoverage = alpha.
        self.level = _clamp(self.level + self.eta * (miscoverage - self.alpha))

        # Recompute the threshold as the lambda_t quantile of the window.
        if len(self.window) >= 2:
            self._threshold = float(np.quantile(np.array(self.window), self.level))

    def should_abstain(self, score: float) -> bool:
        return score > self._threshold

    def get_stats(self) -> dict:
        return {
            "threshold": self._threshold,
            "level": self.level,
            "window_size": len(self.window),
            "n_updates": self._n_updates,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set PYTHONPATH=src && python -m pytest tests/test_online_aci.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conformal_rag/online.py tests/test_online_aci.py
git commit -m "fix(online): implement true quantile-level ACI targeting alpha"
```

---

### Task 4: Live monitoring in serving (Gap 4)

**Files:**
- Modify: `src/conformal_rag/domain.py`
- Modify: `src/conformal_rag/api.py`
- Test: `tests/test_monitoring_live.py` (new)

**Interfaces:**
- Consumes: `FDRMonitor`, `DriftStatus`, `ConformalRAGEngine`.
- Produces: `DomainManager.monitors[domain]`, `DomainManager.record(domain, predicted, actual)`, `DomainManager.monitor_drift(domain)` returns real FDR; API `POST /feedback` and `/metrics` returns live monitor data.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitoring_live.py
from conformal_rag.domain import DomainManager
from conformal_rag.api import create_app
from fastapi.testclient import TestClient


def test_domain_monitor_drift_reflects_records():
    dm = DomainManager()
    dm.register_domain("finance")
    dm.record("finance", predicted_hallucination=True, actual_hallucination=True)
    status = dm.monitor_drift("finance")
    assert status.samples >= 1


def test_api_feedback_updates_metrics():
    app = create_app()
    client = TestClient(app)
    client.post("/predict", json={"query": "q", "context": ["c"], "response": "r"})
    client.post("/feedback", json={"predicted_hallucination": True, "actual_hallucination": False})
    metrics = client.get("/metrics").json()
    assert metrics["total_predictions"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set PYTHONPATH=src && python -m pytest tests/test_monitoring_live.py -v`
Expected: FAIL (`DomainManager.record`/`monitors` missing; `/feedback` 404; `/metrics` static).

- [ ] **Step 3: Write minimal implementation**

In `src/conformal_rag/domain.py`:
- In `__init__`, add `self.monitors: Dict[str, "FDRMonitor"] = {}`.
- In `register_domain`, after creating the engine, add `self.monitors[domain] = FDRMonitor(alpha=clamp_alpha(alpha))`.
- Add:

```python
    def record(
        self,
        domain: str,
        predicted_hallucination: bool,
        actual_hallucination: bool,
    ) -> None:
        if domain not in self.monitors:
            self.register_domain(domain)
        self.monitors[domain].record(
            predicted_hallucination=predicted_hallucination,
            actual_hallucination=actual_hallucination,
        )
```

- Rewrite `monitor_drift` to use the monitor's real data:

```python
    def monitor_drift(self, domain: str) -> DriftStatus:
        if domain not in self.domains:
            return DriftStatus(
                domain=domain, drifted=False, fdr=0.0, threshold=0.5,
                samples=0, coverage=1.0, recommendation="Domain not calibrated yet",
            )
        engine = self.domains[domain]
        monitor = self.monitors.get(domain)
        if monitor is None:
            recent_fdr = engine.alpha
            samples = 0
        else:
            m = monitor.get_metrics()
            recent_fdr = m["fdr"]
            samples = m["total_predictions"]
        drifted = recent_fdr > engine.alpha * 1.5
        return DriftStatus(
            domain=domain,
            drifted=drifted,
            fdr=recent_fdr,
            threshold=engine._threshold or 0.5,
            samples=samples,
            coverage=1.0 - recent_fdr if samples else 1.0,
            recommendation="Re-calibrate" if drifted else "No drift detected",
        )
```

In `src/conformal_rag/api.py`, replace the app body:
- Add `from pydantic import BaseModel` already present; add `FeedbackRequest` model:

```python
class FeedbackRequest(BaseModel):
    domain: Optional[str] = "default"
    predicted_hallucination: bool = False
    actual_hallucination: bool = False
```

- In `create_app`, after `domain_manager` and `monitor`, keep `monitor` as the default-domain monitor. Update `/predict` to record to `monitor` (predicted side). Add `/feedback`:

```python
    @app.post("/predict")
    async def predict(request: PredictRequest):
        result = engine.predict(
            query=request.query,
            context="\n".join(request.context),
            response=request.response,
            domain=request.domain,
        )
        monitor.record(
            predicted_hallucination=result.abstained,
            actual_hallucination=False,  # unknown at predict time
        )
        return result.model_dump()

    @app.post("/feedback")
    async def feedback(request: FeedbackRequest):
        domain_manager.record(
            request.domain,
            predicted_hallucination=request.predicted_hallucination,
            actual_hallucination=request.actual_hallucination,
        )
        return {"status": "recorded"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set PYTHONPATH=src && python -m pytest tests/test_monitoring_live.py tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conformal_rag/domain.py src/conformal_rag/api.py tests/test_monitoring_live.py
git commit -m "feat(monitoring): live FDR via /feedback; domain drift from real records"
```

---

### Task 5: Proper conformal confidence interval (Gap 5)

**Files:**
- Modify: `src/conformal_rag/engine.py` (`_compute_confidence_interval`)
- Test: `tests/test_confidence_interval.py` (new)

**Interfaces:**
- Consumes: numpy, calibration `scores` + `labels`.
- Produces: `(lower, 1.0)` one-sided conformal bound using the finite-sample `(n+1)(1−α)` quantile index.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_confidence_interval.py
import numpy as np
from conformal_rag.engine import ConformalRAGEngine


def test_ci_index_matches_finite_sample_formula():
    eng = ConformalRAGEngine(alpha=0.2)
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]  # n=5 correct
    labels = [0, 0, 0, 0, 0]
    lo, hi = eng._compute_confidence_interval(scores, labels, 0.2)
    # k = ceil((n+1)*(1-alpha)) = ceil(6*0.8)=ceil(4.8)=5 -> 5th smallest = 0.5
    assert lo == 0.5
    assert hi == 1.0


def test_ci_lower_rises_with_spread():
    eng = ConformalRAGEngine(alpha=0.1)
    tight = [0.40, 0.41, 0.42, 0.43]
    wide = [0.10, 0.20, 0.30, 0.90]
    lo_tight, _ = eng._compute_confidence_interval(tight, [0, 0, 0, 0], 0.1)
    lo_wide, _ = eng._compute_confidence_interval(wide, [0, 0, 0, 0], 0.1)
    assert lo_wide >= lo_tight
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set PYTHONPATH=src && python -m pytest tests/test_confidence_interval.py -v`
Expected: FAIL (old code returns `np.quantile(correct_scores, 1-alpha)`, not the `(n+1)` index).

- [ ] **Step 3: Write minimal implementation**

Replace `_compute_confidence_interval` in `src/conformal_rag/engine.py` with:

```python
    def _compute_confidence_interval(
        self, scores: list, labels: list, alpha: float
    ) -> tuple:
        """One-sided conformal lower bound for the risk score.

        Uses the finite-sample (n+1)(1-alpha) quantile of the correct-class
        calibration scores (Vovk et al. conformal PI). Guarantees, under
        exchangeability, that P(risk_score >= lower) >= 1-alpha.
        """
        if not scores:
            return (0.0, 1.0)
        correct_scores = [s for s, l in zip(scores, labels) if l == 0]
        if not correct_scores:
            return (0.0, 1.0)
        n = len(correct_scores)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = max(1, min(k, n))
        sorted_s = sorted(correct_scores)
        lower = float(sorted_s[k - 1])
        return (lower, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set PYTHONPATH=src && python -m pytest tests/test_confidence_interval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conformal_rag/engine.py tests/test_confidence_interval.py
git commit -m "fix(engine): finite-sample (n+1)(1-alpha) conformal CI"
```

---

### Task 6: Reproducible eval + honest headline numbers (Gap 6)

**Files:**
- Create: `benchmarks/eval.py`
- Modify: `AGENTS.md`, `README.md` (numbers section only)

**Interfaces:**
- Consumes: `ConformalRAGEngine` (NLI-only path, `use_llm_judge=False` to avoid API keys), a small inline labeled sample.
- Produces: a printed/structured report of measured FDR, abstention rate, latency/query, and risk scores for one faithful vs one hallucinated example.

- [ ] **Step 1: Write the eval script and a test that it runs**

```python
# benchmarks/eval.py
"""Reproducible evaluation of conformal-rag on a tiny labeled sample.

Run:  set PYTHONPATH=src && python -m benchmarks.eval
Requires network access to download the NLI model on first run.
"""
from __future__ import annotations

import time

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.scoring import ScoringPipeline


def _sample():
    return [
        ("What is Python?", "Python is a programming language created by Guido van Rossum.",
         "Python is a programming language.", 0),
        ("What is Python?", "Python is a programming language created by Guido van Rossum.",
         "Python is a species of snake found only in Antarctica.", 1),
    ]


def main():
    engine = ConformalRAGEngine(alpha=0.05, scoring=ScoringPipeline(use_llm_judge=False))
    samples = _sample()
    engine.calibrate(
        queries=[s[0] for s in samples],
        contexts=[s[1] for s in samples],
        responses=[s[2] for s in samples],
        labels=[s[3] for s in samples],
    )
    latencies = []
    for q, c, r, lab in samples:
        t0 = time.time()
        res = engine.predict(q, c, r)
        latencies.append((time.time() - t0) * 1000)
        print(f"label={lab} risk={res.risk_score:.4f} abstained={res.abstained} "
              f"ci={res.confidence_interval}")
    print(f"mean_latency_ms={sum(latencies)/len(latencies):.2f}")
    print(f"n_samples={len(samples)} (tiny smoke eval; replace with full benchmarks for real FDR)")


if __name__ == "__main__":
    main()
```

Add a smoke test (no model download — monkeypatch scoring) in `tests/test_eval.py`:

```python
# tests/test_eval.py
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import ScoringBreakdown
from benchmarks import eval as bench_eval


class FakeScoring:
    def __init__(self, score):
        self._score = score
    def score(self, query, context, response):
        return ScoringBreakdown(nli_score=self._score, final_score=self._score,
                                nli_latency_ms=1.0, used_llm_judge=False)


def test_eval_runs(monkeypatch):
    monkeypatch.setattr(ConformalRAGEngine, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(ConformalRAGEngine, "calibrate", lambda self, **k: None)
    monkeypatch.setattr(ConformalRAGEngine, "predict",
        lambda self, q, c, r: __import__("conformal_rag.models").models.ConformalResult(
            query=q, response=r, risk_score=0.9, risk_bound=0.05, abstained=False,
            confidence_interval=(0.0, 1.0),
            scoring_breakdown=ScoringBreakdown(nli_score=0.9, final_score=0.9, nli_latency_ms=1.0),
            domain="default", latency_ms=1.0))
    bench_eval.main()  # should not raise
```

- [ ] **Step 2: Run eval smoke test to verify it passes**

Run: `set PYTHONPATH=src && python -m pytest tests/test_eval.py -v`
Expected: PASS (no network).

- [ ] **Step 3: Run the real eval to get measured numbers**

Run: `set PYTHONPATH=src && python -m benchmarks.eval`
Expected: prints risk scores + mean latency. Record the observed `risk_score` for the faithful (label=0) and hallucinated (label=1) examples and the mean latency.

- [ ] **Step 4: Update AGENTS.md / README numbers to measured values or remove**

In `AGENTS.md` and `README.md`, replace the fabricated lines:
- `Faithful→0.9995 (not abstained), Hallucination→0.0004 (abstained)`
- `1.1s/query`
- `FDR=0%`
with the measured values from Step 3, prefixed by a note:
`Measured via \`python -m benchmarks.eval\` on <date>: faithful risk=<X.XXXX>, hallucinated risk=<Y.YYYY>, mean latency=<Z.ZZ>ms. Full FDR requires the benchmark suites in benchmarks/.`
If values are not reproducible, remove the claims entirely and point to `benchmarks/eval.py`.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/eval.py tests/test_eval.py AGENTS.md README.md
git commit -m "feat(benchmarks): reproducible eval; replace fabricated headline numbers"
```

---

## Final verification

- [ ] `set PYTHONPATH=src && python -m pytest tests/ -q` → all pass (161 + new).
- [ ] `ruff check src/ tests/ benchmarks/` → clean.
- [ ] `python -m benchmarks.eval` runs and prints measured numbers.
- [ ] AGENTS.md/README numbers match the eval output or are removed.
