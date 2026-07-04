import pytest
from benchmarks import register_benchmark, get_all_benchmarks, BENCHMARK_REGISTRY
from benchmarks.base import Sample, Result, Aggregation, BaseBenchmark
from benchmarks.config import load_config, BenchmarkConfig


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


def test_sample_dataclass():
    sample = Sample(
        id="q1",
        question="What is the capital of France?",
        context="France is a country in Europe.",
        ground_truth="Paris",
    )
    assert sample.id == "q1"
    assert sample.question == "What is the capital of France?"
    assert sample.context == "France is a country in Europe."
    assert sample.ground_truth == "Paris"
    assert sample.metadata == {}


def test_sample_default_metadata():
    sample = Sample(id="q1", question="Q", context="C", ground_truth="A")
    assert sample.metadata == {}


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
    assert result.prediction_set == ["Paris", "Lyon"]
    assert result.confidence == pytest.approx(0.92)
    assert result.metadata == {}


def test_result_with_metadata():
    result = Result(
        sample_id="q1",
        prediction_set=["A"],
        abstained=True,
        confidence=0.5,
        metadata={"domain": "science"},
    )
    assert result.metadata["domain"] == "science"


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
    assert agg.coverage == pytest.approx(0.94)
    assert agg.abstention_rate == pytest.approx(0.18)
    assert agg.total_samples == 200


def test_aggregation_default_lists():
    agg = Aggregation(fdr=0.0, coverage=1.0, abstention_rate=0.0, total_samples=0)
    assert agg.latency_traces == []
    assert agg.cost_traces == []
    assert agg.metadata == {}


def test_base_benchmark_is_abstract():
    with pytest.raises(TypeError):
        BaseBenchmark()


def test_base_benchmark_subclass():
    class ConcreteBenchmark(BaseBenchmark):
        name = "test"
        description = "A test benchmark"

        async def load_dataset(self):
            return []

        async def run_sample(self, sample, engine):
            return Result(sample_id=sample.id, prediction_set=[], abstained=False, confidence=0.0)

        async def aggregate(self, results):
            return Aggregation(fdr=0.0, coverage=1.0, abstention_rate=0.0, total_samples=len(results))

    b = ConcreteBenchmark()
    assert b.name == "test"


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
