"""Panel demo for Conformal RAG — verifies the guarantee in aggregate.

The finite-sample CRC bound is a statement about a RATE over many answers, not
about any single answer:

    E[ P(approved AND wrong) ] <= alpha

So this demo does what the theorem actually claims: it calibrates on one set of
labelled examples, then measures the empirical rate of "approved but wrong" on a
held-out set the threshold never saw. That is the number worth defending.

Run:
    .venv\\Scripts\\python.exe demo.py

Options:
    --dataset tech|med    which fact set to calibrate on (default: tech)
    --alpha 0.05          error budget
    --small               use the weak NLI preset; usually reports VIOLATED,
                          which is a useful contrast against the default

First run scores every unique (context, response) pair on CPU and takes a few
minutes. Scores are cached under .demo_cache/ keyed by model and input list, so
later runs take ~15s. Only the NLI forward pass is cached — calibration, guard
checks, and threshold construction recompute every time. Delete .demo_cache/ to
force a full rescore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import Document, ScoringBreakdown
from conformal_rag.retrieval import HybridRetriever
from conformal_rag.scoring import ScoringPipeline

from demo_data import DATASETS, balance, build_dataset, make_corpus, split_dataset

W = 76
CACHE_DIR = Path(__file__).parent / ".demo_cache"
_CACHE_VERSION = 3


def rule(char: str = "-") -> None:
    print(char * W)


def head(title: str) -> None:
    print()
    rule("=")
    print(title)
    rule("=")


class ReplayScoring(ScoringPipeline):
    """Serves precomputed NLI scores.

    Substitutes only the model forward pass, which is a pure function of
    (context, response). Engine calibration, guard checks, threshold
    construction, and the decision rule all run for real.
    """

    def __init__(self, scores: dict[tuple[str, str], float], latency_ms: float = 0.0):
        super().__init__(use_llm_judge=False, nli_model="small")
        self._scores = scores
        self._latency_ms = latency_ms

    def _lookup(self, context: str, response: str) -> float:
        key = (context, response)
        if key not in self._scores:
            raise KeyError(f"uncached input: {context[:50]!r} / {response[:50]!r}")
        return self._scores[key]

    def score(self, query: str, context: str, response: str) -> ScoringBreakdown:
        value = self._lookup(context, response)
        return ScoringBreakdown(
            nli_score=value,
            llm_judge_score=None,
            final_score=value,
            nli_latency_ms=self._latency_ms,
            llm_latency_ms=None,
            used_llm_judge=False,
        )

    def score_batch(self, queries, contexts, responses, batch_size: int = 8):
        return [self.score(q, c, r) for q, c, r in zip(queries, contexts, responses)]


def score_dataset(rows, preset: str) -> tuple[dict[tuple[str, str], float], float]:
    """NLI-score every (context, response) pair, with an on-disk cache."""
    pairs = sorted({(r[1], r[2]) for r in rows})
    key = hashlib.sha256(
        json.dumps([_CACHE_VERSION, preset, pairs], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    path = CACHE_DIR / f"{preset}-{key}.json"

    if path.is_file():
        blob = json.loads(path.read_text(encoding="utf-8"))
        scores = {(ctx, resp): float(v) for ctx, resp, v in blob["scores"]}
        print(f"      {len(scores)} NLI scores from cache "
              f"(delete .demo_cache/ to rescore)")
        return scores, float(blob["latency_ms"])

    print(f"      scoring {len(pairs)} pairs with the {preset} NLI model "
          f"(first run only)...")
    scorer = ScoringPipeline(use_llm_judge=False, nli_model=preset)
    t0 = time.time()
    values = scorer.nli.score_batch(pairs, batch_size=8)
    per_item_ms = (time.time() - t0) * 1000 / max(len(pairs), 1)

    scores = {pair: value for pair, value in zip(pairs, values)}
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "version": _CACHE_VERSION,
        "latency_ms": per_item_ms,
        "scores": [[ctx, resp, value] for (ctx, resp), value in scores.items()],
    }), encoding="utf-8")
    print(f"      done in {time.time() - t0:.0f}s ({per_item_ms:.0f}ms/pair) - cached")
    return scores, per_item_ms


def unpack(rows):
    return ([r[0] for r in rows], [r[1] for r in rows],
            [r[2] for r in rows], [r[3] for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="tech")
    ap.add_argument("--small", action="store_true",
                    help="weak NLI preset; usually reports VIOLATED")
    args = ap.parse_args()

    alpha = args.alpha
    preset = "small" if args.small else "large"
    t_start = time.time()

    facts = DATASETS[args.dataset]
    rows = balance(build_dataset(facts))
    calibration, held_out = split_dataset(rows)

    head("CONFORMAL RAG")
    print("  A mathematical bound on how often approved AI answers are wrong.")

    # ---- 1. what is being claimed -----------------------------------------
    rule()
    print("THE CLAIM")
    rule()
    print(f"""
  E[ P(approved AND wrong) ]  <=  alpha = {alpha:.0%}

  This is a statement about a RATE across many answers, not about any single
  answer. An individual answer can still be approved and be wrong — any system
  with a non-zero error rate will do that. What is bounded is how OFTEN.

  Two things are deliberately NOT claimed:
    - no bound on coverage (the system may abstain on most inputs)
    - no per-answer guarantee (P(wrong | approved) <= alpha is not achievable
      for CRC and is not implied by the marginal bound)

  The bound holds only if calibration data is drawn from the same distribution
  as live traffic. That assumption is called exchangeability, and no code can
  check it for you. It is the weakest link in the whole method.
""")

    # ---- 2. where the number comes from -----------------------------------
    print(rule())
    print("STEP 1  Calibrate: find the line")
    print(rule())
    q, c, r, y = unpack(calibration)
    n_cal = len(calibration)
    n_hall = sum(y)
    print(f"\n  Dataset '{args.dataset}': {len(facts)} facts -> "
          f"{n_cal} calibration / {len(held_out)} held-out examples")
    print(f"  Hallucination rate in calibration: {n_hall}/{n_cal} "
          f"= {n_hall / n_cal:.0%}\n")

    scores, latency_ms = score_dataset(calibration + held_out, preset)
    engine = ConformalRAGEngine(alpha=alpha, scoring=ReplayScoring(scores, latency_ms))
    stats = engine.calibrate(q, c, r, y)

    c_max = int(np.floor((n_cal + 1) * alpha)) - 1
    halluc = sorted((s for s, lbl in zip(engine._calibration_scores, y) if lbl == 1),
                    reverse=True)
    faithful_mean = np.mean([s for s, lbl in zip(engine._calibration_scores, y) if lbl == 0])

    print(f"  n                          {n_cal}")
    print(f"  c_max = floor((n+1)*a) - 1  {c_max}")
    print(f"  threshold                  {stats.threshold:.4f}")
    print(f"    = the {c_max + 1}-th largest hallucination score")
    if len(halluc) > c_max:
        print(f"    that is                  {halluc[c_max]:.4f}")
    print(f"  mean score, faithful       {faithful_mean:.4f}")
    print(f"  mean score, hallucinated   "
          f"{np.mean([s for s, lbl in zip(engine._calibration_scores, y) if lbl == 1]):.4f}")
    for w in stats.warnings:
        print(f"\n  WARNING  {w}")
    print(f"""
  Why floor((n+1)*a) and not floor(n*a)?

  The +1 is the conformal correction. Counting as if there were one more
  calibration point than you actually have is what makes the bound provable on
  an unseen answer rather than merely observed on your own sample. It is also
  why you need at least ceil(1/a) - 1 = {int(np.ceil(1 / alpha)) - 1} examples
  before the threshold means anything at all. Below that, the system abstains
  on everything rather than claim a guarantee it cannot support.
""")

    # ---- 3. held-out verification -----------------------------------------
    print(rule())
    print("STEP 2  Test: measure the rate on answers it has never seen")
    print(rule())
    approved = approved_wrong = 0
    results = []
    for query, ctx, resp, label in held_out:
        res = engine.predict(query=query, context=ctx, response=resp)
        results.append((res.risk_score, label, res.abstained, resp))
        if not res.abstained:
            approved += 1
            if label == 1:
                approved_wrong += 1

    right_abstained = [x for x in results if x[2] is True and x[1] == 0]
    fdr = approved_wrong / approved if approved else 0.0
    coverage = approved / len(held_out)
    violation = approved_wrong > alpha * len(held_out)

    print(f"\n  {'score':>7}  {'truth':>6}  {'verdict':>8}")
    print(f"  {'-' * 7}  {'-' * 6}  {'-' * 8}")
    for score, label, abstained, resp in results:
        print(f"  {score:7.4f}  {'wrong' if label else 'ok':>6}  "
              f"{'ABSTAIN' if abstained else 'approve':>8}   {resp[:44]}")

    print(f"""
  --------------------------------------------
  held-out answers                {len(held_out)}
  genuinely wrong                 {sum(x[1] for x in results)}
  approved                        {approved}
  approved AND wrong              {approved_wrong}
  --------------------------------------------
  coverage  (answered)            {coverage:.0%}
  wrong among approved (FDR)      {fdr:.1%}
  budget alpha                    {alpha:.0%}
  --------------------------------------------
  BOUND                           {'HOLDS' if not violation else 'VIOLATED'}""")

    if violation:
        print(f"""
  {approved_wrong} wrong out of {len(held_out)} exceeds the {alpha:.0%} budget.
  Two possible causes, and this script cannot tell them apart:
    1. the scorer is genuinely weak  (try --small to see this directly)
    2. calibration and test data are not exchangeable
  At this sample size one extra violation is already worth {1 / len(held_out):.0%}.""")
    else:
        print(f"""
  {approved_wrong} approved-but-wrong out of {len(held_out)}, against a budget of
  {alpha:.0%}. The system answered {coverage:.0%} of the questions rather than
  refusing everything, which is what makes the bound meaningful instead of
  trivially satisfied.""")
        if right_abstained:
            print(f"  Cost of that: it abstained on {len(right_abstained)} correct "
                  f"answers. Coverage is the price paid for the bound.")

    # ---- 4. retrieval + adaptive control, briefly -------------------------
    print(rule())
    print("STEP 3  The rest of the system")
    print(rule())
    print("""
  Retrieval     dense embeddings (intfloat/e5-base-v2) + BM25 keyword search,
                fused by reciprocal rank. Both are needed: embeddings miss exact
                identifiers, BM25 misses paraphrase. Verified on this corpus:

    Adaptive      quantile-level ACI (Gibbs & Candes 2021) nudges the threshold
                  up when it observes approved errors. eta = 0.01, window 500.
                  This is a HEURISTIC with long-run average convergence, not a
                  finite-sample certificate — and it REPLACES the offline
                  threshold rather than composing with it. The guarantee above
                  comes from the calibrated line, not the controller.
""")
    retriever = HybridRetriever()
    retriever.index([Document(text=t) for t in make_corpus(facts)])
    for q_text in (f[0].replace("-", " ") for f in facts[:3]):
        hits = retriever.retrieve(q_text)
        print(f"    '{q_text}' -> {(hits[0][:66] if hits else '(nothing)')}...")

    print(f"\n  Scoring latency: {latency_ms:.0f}ms per response on CPU.")
    print(f"  Total runtime:   {time.time() - t_start:.1f}s")
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)