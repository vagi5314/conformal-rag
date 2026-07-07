"""Markdown report generation for benchmark runs."""

from __future__ import annotations

from typing import Optional

from benchmarks.base import LatencyTrace
from benchmarks.profiling.cost import CostTracker


def latency_percentiles(traces: list[LatencyTrace]) -> dict:
    """Compute p50/p95/p99 (ms) for each latency component."""
    fields = ("nli_scoring_ms", "llm_judge_ms", "total_ms")

    def pct(values: list[float], p: float) -> float:
        values = sorted(v for v in values if v is not None)
        if not values:
            return 0.0
        idx = min(len(values) - 1, int(len(values) * p / 100))
        return round(values[idx], 2)

    result = {}
    for field in fields:
        series = [getattr(t, field) for t in traces]
        result[field] = {
            "p50": pct(series, 50),
            "p95": pct(series, 95),
            "p99": pct(series, 99),
            "mean": round(sum(series) / len(series), 2) if series else 0.0,
        }
    return result


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_run_report(
    result_dict: dict,
    cost_tracker: Optional[CostTracker] = None,
    budget: Optional[dict] = None,
) -> str:
    """Render a single benchmark run as Markdown."""
    cfg = result_dict.get("config", {})
    metrics = result_dict.get("metrics", {})
    latency = result_dict.get("latency_summary", {})
    name = result_dict.get("benchmark", "unknown")
    alpha = cfg.get("alpha", 0.05)
    fdr = metrics.get("fdr", 0.0)

    lines = [
        f"# Benchmark Report: `{name}`",
        "",
        f"- **Timestamp:** {result_dict.get('timestamp', 'n/a')}",
        f"- **α (target FDR):** {alpha}",
        f"- **Calibration samples:** {cfg.get('calibration_size', 'n/a')}",
        f"- **Test samples:** {cfg.get('test_size', 'n/a')}",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value | Target |",
        "|--------|-------|--------|",
        f"| FDR (empirical) | {_fmt_pct(fdr)} | ≤ {_fmt_pct(alpha)} |",
        f"| Coverage | {_fmt_pct(metrics.get('coverage', 0.0))} | higher is better |",
        f"| Abstention rate | {_fmt_pct(metrics.get('abstention_rate', 0.0))} | — |",
        f"| Total samples | {metrics.get('total_samples', 0)} | — |",
        "",
    ]

    fdr_ok = fdr <= alpha + 1e-9
    verdict = (
        "PASS — empirical FDR within the α guarantee."
        if fdr_ok
        else "FAIL — empirical FDR exceeds α."
    )
    lines += ["**FDR guarantee:** " + verdict, ""]

    if latency:
        lines += [
            "## Latency (ms)",
            "",
            "| Component | p50 | p95 | p99 | mean |",
            "|-----------|-----|-----|-----|------|",
        ]
        for comp, stats in latency.items():
            lines.append(
                f"| {comp} | {stats['p50']} | {stats['p95']} | {stats['p99']} | {stats['mean']} |"
            )
        lines.append("")

    if cost_tracker is not None and (cost_tracker.traces or cost_tracker.total_cost):
        lines += [
            "## Cost",
            "",
            f"- **Total cost:** ${cost_tracker.total_cost:.4f}",
            f"- **Avg/query:** ${cost_tracker.avg_cost_per_query():.6f}",
            f"- **By provider:** "
            + ", ".join(f"{k}=${v:.4f}" for k, v in cost_tracker.by_provider.items()),
            "",
        ]

    if budget:
        lines += ["## Budget Simulation (per QPS level)", ""]
        lines.append("| QPS | $/hour | $/day | $/month |")
        lines.append("|-----|--------|-------|---------|")
        for qps, sim in sorted(budget.items()):
            lines.append(
                f"| {qps} | ${sim['cost_per_hour']:.2f} | "
                f"${sim['cost_per_day']:.2f} | ${sim['cost_per_month']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def render_aggregate_report(rows: list[dict]) -> str:
    """Render a cross-benchmark comparison table."""
    lines = [
        "# Cross-Benchmark Aggregate Report",
        "",
        "| Benchmark | α | FDR | Coverage | Abstention | Samples | FDR ≤ α |",
        "|-----------|---|-----|----------|------------|---------|--------|",
    ]
    for r in rows:
        cfg = r.get("config", {})
        m = r.get("metrics", {})
        alpha = cfg.get("alpha", 0.05)
        fdr = m.get("fdr", 0.0)
        ok = "✅" if fdr <= alpha + 1e-9 else "❌"
        lines.append(
            f"| {r.get('benchmark', '?')} | {alpha} | {_fmt_pct(fdr)} | "
            f"{_fmt_pct(m.get('coverage', 0.0))} | {_fmt_pct(m.get('abstention_rate', 0.0))} | "
            f"{m.get('total_samples', 0)} | {ok} |"
        )
    lines.append("")
    return "\n".join(lines)
