"""Seed the demo state: index the corpus, calibrate the main engine and domains.

Run once before the demo (run.cmd does this automatically when state is absent):

    set PYTHONPATH=src
    python scripts/seed_demo.py

Writes CONFORMAL_RAG_STATE_DIR so the API restores everything at startup and
the demo opens already calibrated instead of asking the panel to wait.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conformal_rag.config import Settings  # noqa: E402
from conformal_rag.demo_corpus import (  # noqa: E402
    ALL_DOCS,
    DOMAINS,
    GENERAL_SAMPLES,
    MAIN_ALPHA,
    ROUTING_QUERIES,
)
from conformal_rag.domain import DomainManager  # noqa: E402
from conformal_rag.engine import ConformalRAGEngine  # noqa: E402
from conformal_rag.models import Document  # noqa: E402
from conformal_rag.retrieval import HybridRetriever  # noqa: E402
from conformal_rag.scoring import ScoringPipeline  # noqa: E402

ENGINE_STATE = "engine_state.json"
INDEX_STATE = "index_state.json"
DOMAIN_DIR = "domains"


def _cycle(texts: list, n: int) -> list:
    """Repeat `texts` to length n, so routing sees realistic questions."""
    return [texts[i % len(texts)] for i in range(n)] if texts else ["demo"] * n


def _judge_available() -> bool:
    """Whether any LLM-judge provider key is present in the environment."""
    return any(
        os.environ.get(name)
        for name in (
            "GROQ_API_KEY",
            "OPENROUTER_API_KEY",
            "TOGETHER_API_KEY",
            "TOGETHERAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        )
    )


def main(force: bool = False) -> int:
    settings = Settings.from_env()
    if not settings.state_dir:
        print("ERROR: CONFORMAL_RAG_STATE_DIR is not set; nothing would be persisted.")
        return 2

    state = Path(settings.state_dir)
    if (state / ENGINE_STATE).exists() and not force:
        print(f"State already present in {state}; nothing to do (use --force to rebuild).")
        return 0

    state.mkdir(parents=True, exist_ok=True)
    judge_on = settings.use_llm_judge and _judge_available()
    print(f"Seeding demo state into {state}")
    print(f"  NLI model : {settings.nli_model}")
    print(f"  LLM judge : {'on' if judge_on else 'off (no provider key)'}")
    print(f"  main alpha: {MAIN_ALPHA}")

    engine = ConformalRAGEngine(
        alpha=MAIN_ALPHA,
        scoring=ScoringPipeline(
            nli_model=settings.nli_model,
            use_llm_judge=judge_on,
            fallback_chain=settings.llm_fallback_chain,
        ),
    )
    retriever = HybridRetriever(embedding_model=settings.embedding_model)

    print(f"  indexing {len(ALL_DOCS)} documents ...")
    retriever.index([Document(text=t) for t in ALL_DOCS])

    print(f"  calibrating main engine on {len(GENERAL_SAMPLES)} samples ...")
    stats = engine.calibrate(
        queries=_cycle(ROUTING_QUERIES["general"], len(GENERAL_SAMPLES)),
        contexts=[c for c, _, _ in GENERAL_SAMPLES],
        responses=[r for _, r, _ in GENERAL_SAMPLES],
        labels=[lbl for _, _, lbl in GENERAL_SAMPLES],
        domain="general",
    )
    print(
        f"    threshold={stats.threshold:.4f}  approved={stats.coverage:.0%}  "
        f"risk={stats.approved_wrong_rate:.1%} (bound {MAIN_ALPHA})  "
        f"certifiable={stats.certifiable}"
    )

    manager = DomainManager(
        route_margin=settings.domain_route_margin,
        embedding_model=settings.embedding_model,
        nli_model=settings.nli_model,
        use_llm_judge=judge_on,
        fallback_chain=settings.llm_fallback_chain,
    )

    # A calibrated "general" domain is seeded alongside the topic domains so
    # auto-routing always lands on a calibrated threshold: without it, a
    # general question would route to "general" and hit a fresh, uncalibrated
    # engine (which has no bound to apply).
    seed_domains = {"general": {"alpha": MAIN_ALPHA, "samples": GENERAL_SAMPLES}}
    for name, spec in DOMAINS.items():
        seed_domains[name] = {"alpha": spec["alpha"], "samples": spec["samples"]}

    for name, spec in seed_domains.items():
        samples = spec["samples"]
        manager.register_domain(name, alpha=spec["alpha"])
        dstats = manager.calibrate(
            domain=name,
            queries=_cycle(ROUTING_QUERIES.get(name, []), len(samples)),
            contexts=[c for c, _, _ in samples],
            responses=[r for _, r, _ in samples],
            labels=[lbl for _, _, lbl in samples],
        )
        print(
            f"  domain {name:9} alpha={spec['alpha']:<5} "
            f"threshold={dstats.threshold:.4f}  approved={dstats.coverage:.0%}  "
            f"risk={dstats.approved_wrong_rate:.1%}  certifiable={dstats.certifiable}"
        )

    engine.save_state(state / ENGINE_STATE)
    retriever.save_state(state / INDEX_STATE)
    manager.save_state(state / DOMAIN_DIR)

    print("\nSeeded. Start the API with the same CONFORMAL_RAG_STATE_DIR to restore it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(force="--force" in sys.argv))