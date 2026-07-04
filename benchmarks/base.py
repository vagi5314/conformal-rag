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
