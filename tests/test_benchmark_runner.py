"""Integration tests for the benchmark runner pipeline (offline)."""

import types
from pathlib import Path
from unittest.mock import patch

from benchmarks import get_all_benchmarks
from benchmarks.config import BenchmarkConfig, OutputConfig
from benchmarks.runner import BenchmarkRunner
from benchmarks.base import Sample
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import ConformalResult, ScoringBreakdown


def _fake_predict_factory(abstain_on_marker: bool):
    def fake_predict(self, query, context, response, domain=None):
        abstain = abstain_on_marker and "[HALL]" in response
        score = 0.1 if abstain else 0.9
        return ConformalResult(
            query=query, response=response,
            risk_score=score, risk_bound=0.05, abstained=abstain,
            confidence_interval=(0.0, 1.0),
            scoring_breakdown=ScoringBreakdown(
                nli_score=score, final_score=score, nli_latency_ms=5.0
            ),
            latency_ms=7.0,
        )

    return fake_predict


async def _fake_dataset(self):
    samples = []
    for i in range(5):
        samples.append(Sample(id=f"t_f{i}", question="Q?", context="C",
                              ground_truth=f"Paris [OK]{i}",
                              metadata={"has_hallucination": False}))
        samples.append(Sample(id=f"t_h{i}", question="Q?", context="C",
                              ground_truth=f"London [HALL]{i}",
                              metadata={"has_hallucination": True}))
    return samples


def _runner(tmp_path: Path) -> BenchmarkRunner:
    cfg = BenchmarkConfig(output=OutputConfig(
        results_dir=str(tmp_path / "results"), reports_dir=str(tmp_path / "reports")
    ))
    return BenchmarkRunner(config=cfg, alpha=0.05)


def test_runner_discovers_all_benchmarks():
    names = set(get_all_benchmarks().keys())
    assert {"hotpotqa", "ragtruth", "halueval", "financebench", "multihoprag"} <= names


async def test_runner_dry_run_no_engine(tmp_path):
    runner = _runner(tmp_path)
    with patch("benchmarks.hotpotqa.HotpotQABenchmark.load_dataset", _fake_dataset):
        out = await runner.run_one("hotpotqa", dry_run=True)
    assert out["dry_run"] is True


async def test_runner_end_to_end_fdr_pass(tmp_path):
    runner = _runner(tmp_path)
    with patch("benchmarks.hotpotqa.HotpotQABenchmark.load_dataset", _fake_dataset):
        with patch.object(ConformalRAGEngine, "predict", _fake_predict_factory(True)):
            with patch.object(
                ConformalRAGEngine, "calibrate",
                lambda self, *a, **k: types.SimpleNamespace(threshold=0.5, estimated_fdr=0.0),
            ):
                out = await runner.run_one("hotpotqa", dry_run=False)
    assert out["metrics"]["fdr"] == 0.0
    assert out["metrics"]["coverage"] == 0.5
    assert list((tmp_path / "results").glob("hotpotqa_*.json"))
    assert list((tmp_path / "reports").glob("hotpotqa_*.md"))


async def test_runner_end_to_end_fdr_fail_when_accepting_hallucinations(tmp_path):
    runner = _runner(tmp_path)
    with patch("benchmarks.hotpotqa.HotpotQABenchmark.load_dataset", _fake_dataset):
        with patch.object(ConformalRAGEngine, "predict", _fake_predict_factory(False)):
            with patch.object(
                ConformalRAGEngine, "calibrate",
                lambda self, *a, **k: types.SimpleNamespace(threshold=0.5, estimated_fdr=0.0),
            ):
                out = await runner.run_one("hotpotqa", dry_run=False)
    # both accepted; one hallucination -> fdr = 0.5 > alpha
    assert out["metrics"]["fdr"] == 0.5
