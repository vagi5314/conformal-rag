"""Benchmark runner orchestrator and CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmarks import get_all_benchmarks
from benchmarks.config import load_config, BenchmarkConfig
from benchmarks.base import Aggregation
from conformal_rag.engine import ConformalRAGEngine


class BenchmarkRunner:
    """Discovers and runs benchmark plugins."""

    def __init__(self, config: Optional[BenchmarkConfig] = None, alpha: float = 0.05):
        self.config = config or load_config()
        self.benchmarks = get_all_benchmarks()
        self.alpha = alpha

    async def run_one(
        self, name: str, alpha: float = 0.05, dry_run: bool = False
    ) -> dict:
        """Run a single benchmark."""
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not found. Available: {list(self.benchmarks.keys())}")

        benchmark_cls = self.benchmarks[name]
        benchmark = benchmark_cls()

        print(f"Loading {name} dataset...")
        samples = await benchmark.load_dataset()
        print(f"Loaded {len(samples)} samples")

        calibration_size = int(len(samples) * self.config.calibration_ratio)
        calibration_samples = samples[:calibration_size]
        test_samples = samples[calibration_size:]

        if dry_run:
            print(f"Dry run: would calibrate on {calibration_size} samples, test on {len(test_samples)}")
            return {"benchmark": name, "dry_run": True, "samples": len(samples)}

        engine = ConformalRAGEngine(alpha=alpha)

        results = []
        for sample in test_samples:
            result = await benchmark.run_sample(sample, engine)
            results.append(result)

        aggregation = await benchmark.aggregate(results)

        output_data = {
            "benchmark": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "alpha": alpha,
                "calibration_size": calibration_size,
                "test_size": len(test_samples),
                "providers": self.config.providers,
            },
            "metrics": {
                "fdr": aggregation.fdr,
                "coverage": aggregation.coverage,
                "abstention_rate": aggregation.abstention_rate,
                "total_samples": aggregation.total_samples,
            },
        }

        results_dir = Path(self.config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        json_path = results_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {json_path}")
        return output_data

    async def run_all(self, alpha: float = 0.05, dry_run: bool = False) -> list[dict]:
        """Run all registered benchmarks."""
        results = []
        for name in self.benchmarks:
            result = await self.run_one(name, alpha=alpha, dry_run=dry_run)
            results.append(result)
        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Conformal RAG Benchmark Runner")
    parser.add_argument("--benchmark", "-b", help="Run a specific benchmark")
    parser.add_argument("--alpha", "-a", type=float, default=0.05, help="Significance level")
    parser.add_argument("--dry-run", action="store_true", help="Load only, no evaluation")
    args = parser.parse_args()

    runner = BenchmarkRunner()

    if args.benchmark:
        asyncio.run(runner.run_one(args.benchmark, alpha=args.alpha, dry_run=args.dry_run))
    else:
        asyncio.run(runner.run_all(alpha=args.alpha, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
