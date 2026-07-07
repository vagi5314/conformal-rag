"""Benchmark plugin registry."""

from __future__ import annotations

from typing import Type

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


def _import_benchmarks() -> None:
    """Import all benchmark modules to trigger registration."""
    import importlib
    import pathlib

    pkg_dir = pathlib.Path(__file__).parent
    for py_file in pkg_dir.glob("*.py"):
        if py_file.name.startswith("_") or py_file.name in ("base.py", "config.py", "runner.py"):
            continue
        module_name = f"benchmarks.{py_file.stem}"
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass


_import_benchmarks()


def get_all_benchmarks() -> dict[str, Type]:
    """Return all registered benchmarks."""
    return dict(BENCHMARK_REGISTRY)
