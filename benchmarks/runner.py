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
from benchmarks._common import aggregate_results
from benchmarks.profiling.cost import CostTracker
from benchmarks.profiling.latency import LatencyTracer
from benchmarks.reporting import (
    latency_percentiles,
    render_aggregate_report,
    render_run_report,
)
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.scoring import ScoringPipeline


class BenchmarkRunner:
    """Discovers and runs benchmark plugins."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        alpha: float = 0.05,
        use_llm_judge: bool = False,
    ):
        self.config = config or load_config()
        self.benchmarks = get_all_benchmarks()
        self.alpha = alpha
        self.use_llm_judge = use_llm_judge

    def _build_engine(self, cost_tracker: CostTracker) -> ConformalRAGEngine:
        scoring = ScoringPipeline(use_llm_judge=self.use_llm_judge)
        engine = ConformalRAGEngine(alpha=self.alpha, scoring=scoring)
        if self.use_llm_judge:
            # Wire real token-level cost capture into the LLM judge client.
            engine.scoring.llm_judge.client.config.usage_callback = (
                lambda provider, inp, out: cost_tracker.record(
                    provider.split("/")[0], inp, out
                )
            )
        return engine

    async def run_one(
        self,
        name: str,
        alpha: Optional[float] = None,
        dry_run: bool = False,
        simulate_budget: bool = False,
        limit: Optional[int] = None,
    ) -> dict:
        """Run a single benchmark and return its result payload."""
        alpha = alpha if alpha is not None else self.alpha
        if name not in self.benchmarks:
            raise ValueError(
                f"Benchmark '{name}' not found. Available: {list(self.benchmarks.keys())}"
            )

        benchmark_cls = self.benchmarks[name]
        benchmark = benchmark_cls()

        print(f"Loading {name} dataset...")
        if limit is not None:
            samples = await benchmark.load_dataset(limit=limit)
        else:
            samples = await benchmark.load_dataset()
        print(f"Loaded {len(samples)} samples")

        calibration_size = int(len(samples) * self.config.calibration_ratio)
        calibration_samples = samples[:calibration_size]
        test_samples = samples[calibration_size:]

        if dry_run:
            print(
                f"Dry run: calibrate on {len(calibration_samples)} samples, "
                f"test on {len(test_samples)}"
            )
            return {"benchmark": name, "dry_run": True, "samples": len(samples)}

        cost_tracker = CostTracker()
        tracer = LatencyTracer()
        engine = self._build_engine(cost_tracker)

        cal_queries = [s.question for s in calibration_samples]
        cal_contexts = [s.context for s in calibration_samples]
        cal_responses = [s.ground_truth for s in calibration_samples]
        cal_labels = [
            1 if s.metadata.get("has_hallucination", False) else 0
            for s in calibration_samples
        ]
        stats = engine.calibrate(cal_queries, cal_contexts, cal_responses, cal_labels)
        print(
            f"Calibration: threshold={stats.threshold:.4f}, "
            f"est_fdr={stats.estimated_fdr:.4f}"
        )

        results = []
        for sample in test_samples:
            results.append(
                await benchmark.run_sample(
                    sample, engine, tracer=tracer, cost_tracker=cost_tracker
                )
            )

        aggregation: Aggregation = await benchmark.aggregate(results)

        budget = None
        if simulate_budget:
            avg_latency = (
                tracer.percentile("total_ms", 50) or 0.0
            )
            budget = cost_tracker.simulate_budget(
                avg_latency_ms=avg_latency,
                qps_levels=self.config.budget_simulation.qps_levels,
            )

        latency_summary = latency_percentiles(aggregation.latency_traces)

        output_data = {
            "benchmark": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "alpha": alpha,
                "use_llm_judge": self.use_llm_judge,
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
            "latency_summary": latency_summary,
            "cost_summary": {
                "total_cost": cost_tracker.total_cost,
                "avg_per_query": cost_tracker.avg_cost_per_query(),
                "by_provider": cost_tracker.by_provider,
            },
        }
        if budget:
            output_data["budget"] = budget

        self._write_outputs(name, output_data, aggregation, cost_tracker, budget)
        return output_data

    def _write_outputs(
        self,
        name: str,
        output_data: dict,
        aggregation: Aggregation,
        cost_tracker: CostTracker,
        budget: Optional[dict],
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.output.results_dir)
        reports_dir = Path(self.config.output.reports_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        json_path = results_dir / f"{name}_{stamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        md_path = reports_dir / f"{name}_{stamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(
                render_run_report(output_data, cost_tracker=cost_tracker, budget=budget)
            )
        print(f"Results: {json_path}\nReport:  {md_path}")

    async def run_all(
        self,
        alpha: Optional[float] = None,
        dry_run: bool = False,
        simulate_budget: bool = False,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """Run all registered benchmarks and write an aggregate report."""
        alpha = alpha if alpha is not None else self.alpha
        rows = []
        for name in self.benchmarks:
            rows.append(
                await self.run_one(
                    name, alpha=alpha, dry_run=dry_run, simulate_budget=simulate_budget, limit=limit
                )
            )

        if not dry_run:
            reports_dir = Path(self.config.output.reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            agg_path = reports_dir / "aggregate_report.md"
            with open(agg_path, "w", encoding="utf-8") as f:
                f.write(render_aggregate_report(rows))
            print(f"Aggregate report: {agg_path}")
        return rows


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Conformal RAG Benchmark Runner")
    parser.add_argument("--benchmark", "-b", help="Run a specific benchmark")
    parser.add_argument("--alpha", "-a", type=float, default=0.05, help="Significance level")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM judge (costs API credits)")
    parser.add_argument("--dry-run", action="store_true", help="Load only, no evaluation")
    parser.add_argument("--simulate-budget", action="store_true", help="Simulate cost at production QPS")
    parser.add_argument("--limit", type=int, default=None, help="Cap dataset size per benchmark")
    args = parser.parse_args()

    runner = BenchmarkRunner(alpha=args.alpha, use_llm_judge=args.llm_judge)

    if args.benchmark:
        asyncio.run(
            runner.run_one(
                args.benchmark,
                alpha=args.alpha,
                dry_run=args.dry_run,
                simulate_budget=args.simulate_budget,
                limit=args.limit,
            )
        )
    else:
        asyncio.run(
            runner.run_all(
                alpha=args.alpha,
                dry_run=args.dry_run,
                simulate_budget=args.simulate_budget,
                limit=args.limit,
            )
        )


if __name__ == "__main__":
    main()
