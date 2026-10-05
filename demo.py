"""Panel demo for Conformal RAG — verifies the guarantee in aggregate.

The finite-sample CRC bound is a statement about a RATE over many answers, not
about any single answer:

    E[ P(approved AND wrong) ] <= alpha

So this demo does what the theorem actually claims: it calibrates on one set of
labelled examples, then measures the empirical rate of "approved but wrong" on a
held-out set the threshold never saw. That is the number a panel should care
about, and it is what the paper reports.

A single answer can still be approved and wrong. That is not a bug — any system
with a non-zero error rate will do that. The guarantee is that the RATE stays
under alpha.

Run:
    .venv\\Scripts\\python.exe demo.py

Uses the production NLI model (DeBERTa-v3-large) by default. The first run
scores 63 pairs on CPU and takes a few minutes; scores are cached afterwards:

    .venv\\Scripts\\python.exe demo.py

Contrast with the weak preset, which is fast but likely violates the bound:

    .venv\\Scripts\\python.exe demo.py --small

Scoring is the slow part (~150s for the large model on CPU), so scores are
cached under .demo_cache/ keyed by model + exact input list. Repeat runs are
instant. The engine's calibration and threshold logic is never cached — that
recomputes every time. Delete .demo_cache/ to force a full rescore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import Document, ScoringBreakdown
from conformal_rag.retrieval import HybridRetriever
from conformal_rag.scoring import ScoringPipeline

W = 78
CACHE_DIR = Path(__file__).parent / ".demo_cache"


def rule(char: str = "-") -> None:
    print(char * W)


def head(title: str) -> None:
    print()
    rule("=")
    print(title)
    rule("=")


class ReplayScoring(ScoringPipeline):
    """Scoring pipeline that serves precomputed NLI scores.

    Used only to skip repeated NLI forward passes. Engine calibration, guard
    checks, threshold construction, and the decision rule all run for real —
    this replaces nothing except the model forward pass, which is a pure
    function of (context, response).
    """

    def __init__(self, scores: dict[tuple[str, str], float],
                 latency_ms: float = 0.0):
        super().__init__(use_llm_judge=False, nli_model="small")
        self._scores = scores
        self._latency_ms = latency_ms

    def _lookup(self, context: str, response: str) -> float:
        key = (context, response)
        if key not in self._scores:
            raise KeyError(
                f"uncached input — {context[:60]!r} / {response[:60]!r}. "
                "Delete .demo_cache/ to rescore."
            )
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


_CACHE_VERSION = 2


def score_dataset(rows, preset: str) -> tuple[dict[tuple[str, str], float], float]:
    """NLI-score every (context, response) pair, with an on-disk cache.

    Returns the score map keyed by (context, response) and the mean per-item
    latency measured while scoring. Cache layout is a JSON object:
        {"version": 2, "latency_ms": 169.0, "scores": [[ctx, resp, value], ...]}
    """
    pairs = sorted({(r[1], r[2]) for r in rows})
    key = hashlib.sha256(
        json.dumps([_CACHE_VERSION, preset, pairs], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    path = CACHE_DIR / f"{preset}-{key}.json"

    if path.is_file():
        blob = json.loads(path.read_text(encoding="utf-8"))
        scores = {(ctx, resp): float(v) for ctx, resp, v in blob["scores"]}
        print(f"      {len(scores)} NLI scores loaded from cache "
              f"(delete .demo_cache/ to recompute)")
        return scores, float(blob["latency_ms"])

    print(f"      scoring {len(pairs)} unique (context, response) pairs "
          f"with the {preset} NLI model...")
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
    print(f"      done in {time.time() - t0:.1f}s ({per_item_ms:.0f}ms/item) — cached")
    return scores, per_item_ms


def build_replay_engine(alpha: float, preset: str, rows) -> tuple[ConformalRAGEngine, float]:
    scores, latency_ms = score_dataset(rows, preset)
    return ConformalRAGEngine(alpha=alpha, scoring=ReplayScoring(scores, latency_ms)), latency_ms


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

CORPUS = [
    "Albert Einstein published the special theory of relativity in 1905 and the "
    "general theory in 1915. He received the Nobel Prize in Physics in 1921 for "
    "his explanation of the photoelectric effect.",
    "The French Revolution began in 1789 with the storming of the Bastille and "
    "ended in the late 1790s with the rise of Napoleon Bonaparte.",
    "Python was conceived in the late 1980s by Guido van Rossum at CWI in the "
    "Netherlands as a successor to the ABC programming language.",
    "Conformal prediction constructs prediction sets with guaranteed coverage "
    "without assuming any distribution for the data.",
    "Qdrant is a vector database written in Rust that supports dense, sparse and "
    "hybrid vector search with filtering.",
    "The mitochondrion is the organelle responsible for producing ATP through "
    "oxidative phosphorylation in eukaryotic cells.",
]

# (query, context, response, label)  label 1 = contains a hallucination
# Each topic appears at three faithfulness levels: supported, partially wrong
# (one detail corrupted), and fabricated. Real hallucination data is dominated
# by the middle category — a real system must calibrate on that spread, not on
# the trivially-zero fabricated end.
CALIBRATION = [
    ("Nobel Prize", CORPUS[0], "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect.", 0),
    ("Nobel Prize", CORPUS[0], "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect.", 0),
    ("relativity", CORPUS[0], "Einstein published the special theory of relativity in 1905.", 0),
    ("relativity", CORPUS[0], "He published the general theory of relativity in 1915.", 0),
    ("Nobel Prize", CORPUS[0], "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect.", 0),
    ("relativity", CORPUS[0], "Einstein's relativity work concerned the relationship between mass and energy.", 0),
    ("Nobel Prize", CORPUS[0], "Einstein received the Nobel Prize in Physics in 1921 for the discovery of helium.", 1),
    ("Nobel Prize", CORPUS[0], "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect but the award was actually given in 1922.", 1),
    ("relativity", CORPUS[0], "Einstein published the special theory of relativity in 1906.", 1),
    ("relativity", CORPUS[0], "The general theory of relativity was published in 1915 and won Einstein the Nobel Prize in 1915.", 1),
    ("relativity", CORPUS[0], "Einstein developed quantum theory alongside Bohr and Planck.", 1),

    ("French Revolution", CORPUS[1], "The French Revolution began in 1789 with the storming of the Bastille.", 0),
    ("French Revolution", CORPUS[1], "It began in 1789 and ended in the late 1790s with Napoleon's rise.", 0),
    ("Bastille", CORPUS[1], "The Bastille was stormed by the people of Paris.", 0),
    ("Napoleon", CORPUS[1], "The Revolution ended in the late 1790s with the rise of Napoleon Bonaparte.", 0),
    ("French Revolution", CORPUS[1], "The French Revolution began in 1689 with the storming of the Bastille.", 1),
    ("French Revolution", CORPUS[1], "The Revolution ended with Napoleon Bonaparte taking power in 1799 after the storming of the Bastille in 1789.", 1),
    ("Bastille", CORPUS[1], "The Bastille was stormed in 1789 during the American Revolutionary War.", 1),
    ("Napoleon", CORPUS[1], "The Revolution ended with the restoration of Louis XVI in the 1790s.", 1),

    ("Python", CORPUS[2], "Python was created by Guido van Rossum in the late 1980s.", 0),
    ("Python", CORPUS[2], "Python was conceived at CWI in the Netherlands.", 0),
    ("Python", CORPUS[2], "Python is a successor to the ABC language.", 0),
    ("Guido", CORPUS[2], "Guido van Rossum designed Python.", 0),
    ("Python", CORPUS[2], "Python was created by Guido van Rossum in the late 1980s at CWI.", 0),
    ("Python", CORPUS[2], "Python was created by Guido van Rossum in the late 1980s at Bell Labs.", 1),
    ("Python", CORPUS[2], "Python was created by Guido van Rossum in the late 1980s at CWI as a successor to ABC, and it was released in 1991.", 1),
    ("Guido", CORPUS[2], "Guido van Rossum created Python in 1995 while working at Microsoft.", 1),
    ("Python", CORPUS[2], "Python is a compiled language written in Rust.", 1),

    ("conformal prediction", CORPUS[3], "Conformal prediction gives guaranteed coverage without distributional assumptions.", 0),
    ("conformal prediction", CORPUS[3], "It constructs prediction sets with guaranteed coverage.", 0),
    ("coverage", CORPUS[3], "The guarantee is a coverage guarantee.", 0),
    ("conformal prediction", CORPUS[3], "Conformal prediction requires the data to be normally distributed.", 1),
    ("coverage", CORPUS[3], "Conformal prediction provides a guarantee that the prediction set is never empty.", 1),

    ("Qdrant", CORPUS[4], "Qdrant is a vector database written in Rust.", 0),
    ("Qdrant", CORPUS[4], "Qdrant supports hybrid vector search with filtering.", 0),
    ("Qdrant", CORPUS[4], "Qdrant supports dense and sparse vector search.", 0),
    ("Qdrant", CORPUS[4], "Qdrant is a vector database written in Go.", 1),
    ("Qdrant", CORPUS[4], "Qdrant is written in Rust and is licensed under the MIT licence.", 1),

    ("mitochondria", CORPUS[5], "The mitochondrion produces ATP through oxidative phosphorylation.", 0),
    ("mitochondria", CORPUS[5], "Mitochondria are found in eukaryotic cells.", 0),
    ("mitochondria", CORPUS[5], "The mitochondrion is the powerhouse of the eukaryotic cell.", 0),
    ("mitochondria", CORPUS[5], "The mitochondrion produces ATP through photosynthesis.", 1),
    ("mitochondria", CORPUS[5], "Mitochondria are the organelles responsible for protein synthesis in prokaryotic cells.", 1),
]

# Held-out set. Deliberately contains more hallucinations than a typical
# benchmark so the empirical rate is actually measurable rather than 0/N.
HELD_OUT = [
    ("Nobel Prize", CORPUS[0], "Einstein won the Nobel Prize in Physics in 1921 for the photoelectric effect.", 0),
    ("relativity", CORPUS[0], "Einstein published special relativity in 1905 and general relativity in 1915.", 0),
    ("Nobel Prize", CORPUS[0], "Einstein won the Nobel Prize in Physics in 1921 for the photoelectric effect.", 0),
    ("relativity", CORPUS[0], "Einstein won the Nobel Prize for general relativity in 1915.", 1),
    ("Nobel Prize", CORPUS[0], "Einstein won the Nobel Prize in Chemistry in 1921 for radioactivity.", 1),
    ("relativity", CORPUS[0], "Special relativity was published in 1905 and general relativity in 1916.", 1),
    ("Nobel Prize", CORPUS[0], "Einstein won the Nobel Prize in Physics in 1921 for the discovery of the photon in 1905.", 1),

    ("French Revolution", CORPUS[1], "The Revolution began in 1789 and ended in the late 1790s.", 0),
    ("Bastille", CORPUS[1], "The Bastille was stormed in 1789.", 0),
    ("Napoleon", CORPUS[1], "The Revolution ended with the rise of Napoleon Bonaparte.", 0),
    ("French Revolution", CORPUS[1], "The Revolution began in 1789 and ended with Napoleon taking power in 1815.", 1),
    ("Bastille", CORPUS[1], "The Bastille was stormed in 1793 during the Reign of Terror.", 1),
    ("French Revolution", CORPUS[1], "The French Revolution was caused by the American Declaration of Independence.", 1),

    ("Python", CORPUS[2], "Python was created by Guido van Rossum at CWI in the late 1980s.", 0),
    ("Guido", CORPUS[2], "Python is a successor to the ABC language.", 0),
    ("Python", CORPUS[2], "Python was created by Guido van Rossum at CWI in the late 1980s.", 0),
    ("Python", CORPUS[2], "Python was created by Guido van Rossum at CWI in the early 1990s.", 1),
    ("Guido", CORPUS[2], "Python was written by Bjarne Stroustrup in 1995.", 1),
    ("Python", CORPUS[2], "Python is a statically typed compiled language created at CWI.", 1),

    ("conformal prediction", CORPUS[3], "Conformal prediction provides distribution-free coverage.", 0),
    ("Qdrant", CORPUS[4], "Qdrant supports hybrid search.", 0),
    ("mitochondria", CORPUS[5], "Mitochondria produce ATP.", 0),
    ("conformal prediction", CORPUS[3], "Conformal prediction assumes the data follows a Gaussian distribution.", 1),
    ("Qdrant", CORPUS[4], "Qdrant is a relational SQL database written in Java.", 1),
    ("mitochondria", CORPUS[5], "The mitochondrion is where glycolysis takes place in prokaryotes.", 1),
]


def unpack(rows):
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--small", action="store_true",
                    help="use the small NLI preset (fast, and it will likely "
                         "VIOLATE the guarantee — useful as a contrast)")
    args = ap.parse_args()

    alpha = args.alpha
    t_start = time.time()

    head("CONFORMAL RAG — HOW MANY APPROVED ANSWERS ARE ACTUALLY WRONG?")

    # ---- 1. retrieval -----------------------------------------------------
    print("\n[1/5] Retrieval — hybrid dense (e5) + sparse (BM25), fused by RRF\n")
    retriever = HybridRetriever()
    retriever.index([Document(text=t) for t in CORPUS])
    print(f"      Indexed {len(CORPUS)} documents.")
    print("      BM25 built once; dense embeddings precomputed at index time.\n")

    for q in ["Why did Einstein win the Nobel Prize?",
              "What does Qdrant do?",
              "Who created Python?",
              "mitochondria ATP"]:
        hits = retriever.retrieve(q)
        top = hits[0] if hits else "(nothing)"
        print(f"      Q: {q}")
        print(f"      A: {top[:96]}{'...' if len(top) > 96 else ''}\n")

    # ---- 2. what the guarantee is ----------------------------------------
    rule()
    print("[2/5] The claim being tested")
    rule()
    n_cal = len(CALIBRATION)
    print(f"""
      E[ P(approved AND wrong) ]  <=  alpha = {alpha:.0%}

      Read that carefully. It is a statement about a RATE across many answers,
      not about any one answer. A single answer may still be approved and wrong
      — any system with a non-zero error rate will do that. What is bounded is
      how OFTEN that happens.

      The bound is valid only when the calibration examples are drawn from the
      same distribution as production traffic. That assumption is called
      exchangeability, and nothing in the code can verify it for you.
""")

    # ---- 3. calibration ---------------------------------------------------
    rule()
    print("[3/5] Calibration — find the threshold")
    rule()
    preset = "small" if args.small else "large"
    q, c, r, y = unpack(CALIBRATION)
    print(f"\n      Calibrating on {n_cal} labelled examples with the {preset} NLI model.")
    t0 = time.time()
    engine, replay_latency_ms = build_replay_engine(alpha, preset, CALIBRATION + HELD_OUT)
    stats = engine.calibrate(q, c, r, y)
    print(f"      Calibration (guards + threshold construction) in "
          f"{time.time() - t0:.3f}s\n")

    hallucination_scores = sorted(
        (s for s, lbl in zip(engine._calibration_scores, y) if lbl == 1),
        reverse=True,
    )
    c_max = int(np.floor((n_cal + 1) * alpha)) - 1
    print(f"      n                       = {n_cal}")
    print(f"      c_max = floor((n+1)a)-1 = {c_max}")
    print(f"      threshold               = {stats.threshold:.4f}")
    print(f"        (the {c_max + 1}-th largest hallucination score)")
    print(f"      in-sample coverage      = {stats.coverage:.1%}")
    print(f"      in-sample FDR           = {stats.estimated_fdr:.1%}")
    faithful_mean = np.mean([s for s, lbl in zip(engine._calibration_scores, y) if lbl == 0])
    if hallucination_scores:
        print(f"      hallucination scores    = "
              f"{', '.join(f'{s:.3f}' for s in hallucination_scores[:6])}"
              f"{' ...' if len(hallucination_scores) > 6 else ''}")
    print(f"      faithful score mean     = {faithful_mean:.3f}")
    if warnings_list := getattr(stats, "warnings", None):
        print()
        for w in warnings_list:
            print(f"      WARNING: {w}")

    print(f"""
      Why floor((n+1)*alpha) and not floor(n*alpha)?

      The +1 is the conformal correction. Counting as though there were one more
      calibration point than you actually have is what makes the bound hold on an
      unseen example, not just on the data you measured. It is also why a
      calibration set smaller than ceil(1/alpha)-1 samples cannot certify
      anything at all — for alpha={alpha:.0%} that is {int(np.ceil(1 / alpha)) - 1} samples.
""")

    # ---- 4. held-out verification ----------------------------------------
    rule()
    print("[4/5] Held-out test — the number that matters")
    rule()
    q2, c2, r2, y2 = unpack(HELD_OUT)
    t0 = time.time()
    approved, approved_wrong, true_wrong = 0, 0, sum(y2)
    rows = []
    for query, ctx, resp, label in HELD_OUT:
        res = engine.predict(query=query, context=ctx, response=resp)
        rows.append((res.risk_score, label, res.abstained))
        if not res.abstained:
            approved += 1
            if label == 1:
                approved_wrong += 1
    elapsed = time.time() - t0

    fdr = approved_wrong / approved if approved else 0.0
    coverage = approved / len(HELD_OUT)
    violation = approved_wrong > alpha * len(HELD_OUT)

    print(f"      {'score':>8}  {'true':>6}  {'verdict':>9}   text")
    print(f"      {'-' * 8}  {'-' * 6}  {'-' * 9}   {'-' * 40}")
    for (score, label, abstained), (_, _, resp, _) in zip(rows, HELD_OUT):
        mark = "ABSTAIN" if abstained else "approve"
        truth = "wrong" if label == 1 else "ok"
        print(f"      {score:8.4f}  {truth:>6}  {mark:>9}   {resp[:56]}")

    print(f"""
      ----------------------------------------
      total tested            {len(HELD_OUT)}
      truly wrong             {true_wrong}
      approved                {approved}
      approved AND wrong      {approved_wrong}
      ----------------------------------------
      coverage                {coverage:.1%}
      empirical FDR           {fdr:.1%}
      target alpha            {alpha:.1%}
      ----------------------------------------
      verdict                 {'HOLD' if not violation else 'VIOLATED'}""")
    if not violation:
        print(f"        {approved_wrong} wrong out of {len(HELD_OUT)} is within "
              f"the {alpha:.0%} budget.")
    else:
        print(f"        {approved_wrong} wrong out of {len(HELD_OUT)} EXCEEDS "
              f"the {alpha:.0%} budget.")
        print(f"""
        Read this honestly: the bound was not met on this sample. Two possible
        causes, and this script cannot tell them apart:
          1. The scorer is genuinely weak (try --small to see this directly).
          2. Calibration and test data are not exchangeable, so the theorem's
             assumption was never met to begin with.
        With {len(HELD_OUT)} held-out examples, one extra violation is worth
        {1 / len(HELD_OUT):.1%} on its own — small samples give coarse estimates.""")

    per_item_ms = replay_latency_ms
    print(f"\n      NLI latency per response (measured during scoring): "
          f"{per_item_ms:.0f}ms")
    print(f"      verdict loop (replayed scores): {elapsed:.3f}s")
    print(f"      total runtime: {time.time() - t_start:.1f}s")

    # ---- 5. adaptive control ---------------------------------------------
    print()
    rule("=")
    print("[5/5] What ACI does when reality drifts")
    rule("=")
    print("""
      The offline threshold is frozen. It assumes the world keeps looking like
      the calibration set. Online ACI is a feedback controller that watches the
      observed error and nudges a quantile level:

          lambda[t+1] = clip(lambda[t] + eta * (error[t] - alpha), 0, 1)
          threshold   = quantile(lambda[t], window)

      Too many approved errors -> lambda rises -> threshold rises -> abstain more.
      Too few -> relax.

      eta = 0.01, so correcting a full-alpha error takes roughly 100 observations.

      This is a HEURISTIC, not a certificate. Its guarantee is long-run average
      convergence, not finite-sample validity. Once feedback arrives the deployed
      threshold is the controller's, and the offline certificate no longer
      applies to it. The two are not additive — pick one knowingly.
""")
    print(f"      Full run: {time.time() - t_start:.1f}s total.\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)