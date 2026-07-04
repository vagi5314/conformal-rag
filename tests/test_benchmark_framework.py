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
