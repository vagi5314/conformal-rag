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
