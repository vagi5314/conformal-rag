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
