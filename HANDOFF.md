---
generated_at: 2026-07-23
goal: "Run benchmarks with LLM judge, deploy demo to HuggingFace Spaces, start Chrome extension."
---

# Handoff — 2026-07-23 (updated)

## Goal of next session

1. Launch Gradio demo locally, verify it works, fix any import issues
2. Run full benchmark suite with `--llm-judge` on a clean machine (no AppLocker DLL blocks)
3. Deploy demo to HuggingFace Spaces for a public shareable URL
4. Start Chrome extension shell if time permits

## State of play

### Done (this session — 6 commits on `dev`)
- **10 critical code fixes** — 105/105 core tests passing, `ruff check src/` clean
- **Client SDK** — `index()`, `retrieve()`, `check()`, `query()` — `src/conformal_rag/client.py`
- **Engine wiring** — `FDRMonitor` + `OnlineConformal` via `__init__` + `record()` — `src/conformal_rag/engine.py:19-42,154-175`
- **True ACI** — Quantile-level Adaptive Conformal Inference (Gibbs & Candès 2021) — `src/conformal_rag/online.py`
- **Live monitoring** — `POST /feedback`, `/predict` records to `FDRMonitor`, domain drift uses real FDR — `src/conformal_rag/api.py`, `src/conformal_rag/domain.py`
- **Proper conformal CI** — `(n+1)(1-α)` finite-sample quantile — `src/conformal_rag/engine.py:182-196`
- **Retrieval fixes** — BM25 built once, Qdrant sparse uses real token weights — `src/conformal_rag/retrieval.py`
- **LLM judge** — Regex-based score parsing — `src/conformal_rag/scoring.py:106-117`
- **NLI scoring** — Verified correct: `probs[0][0]` = ENTAILMENT for `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Gradio demo** — Predict/Calibrate/Monitor tabs — `demo/app.py`
- **Paper citations** — Added 2025-2026 papers (Semantic Illusion, Conformal-RAG, TRAQ, C-RAG, etc.)
- **Research doc** — Updated RESEARCH.md with latest findings and elevation roadmap
- **Handoff + AGENTS.md** — Updated with current state

### Commits (newest first)
```
a0d915b docs: update paper citations and research findings (2025-2026 papers)
b32ba74 chore: remove stray dashboard.html from root (moved to docs/)
aae589c feat(demo): Gradio web UI with prediction, calibration, and monitoring tabs
91bc6cd docs: update AGENTS.md and remaining-work.md with elevation roadmap
3aa87ff test: update client/online tests, add eval script, handoff doc
08e6ce4 fix(retrieval): BM25 built once in index(), Qdrant sparse uses real weights
6e9604d feat(client+api): real client SDK, live monitoring, /feedback endpoint
0095fb3 fix(core): wire FDRMonitor+OnlineConformal, correct ACI, proper conformal CI, robust LLM parsing
```

### Blocked
- 17 benchmark/retrieval tests fail due to `datasets` DLL block (AppLocker). Need different machine.
- `scikit-learn==1.4.2` and `scipy==1.14.0` pinned (AppLocker binary hash block)

## Quick reference

### Running the demo
```bash
cd C:\Users\ELCOT\Desktop\Final_year
set PYTHONPATH=src
pip install gradio
python demo/app.py
# Opens at http://localhost:7860
```

### Client SDK usage
```python
from conformal_rag.client import ConformalRAG

rag = ConformalRAG(alpha=0.05)
rag.index([
    {"text": "Paris is the capital of France.", "metadata": {"domain": "geography"}},
])
context = rag.retrieve("What is the capital of France?")
result = rag.check("What is the capital of France?", response="Paris")
print(result.abstained)      # False
print(result.risk_score)     # 0.02
print(result.risk_bound)     # 0.05
```

### Engine API (lower level)
```python
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.online import OnlineConformal

engine = ConformalRAGEngine(
    alpha=0.05,
    online=OnlineConformal(alpha=0.05),
)
engine.calibrate(queries, contexts, responses, labels)
result = engine.predict(query, context, response)
```

## Open decisions

- **Benchmarks**: Run on different machine or lift AppLocker allowlist. This machine can't import `datasets`.
- **HuggingFace Space**: Deploy demo as public URL for class presentation.
- **Chrome extension**: 15-25h effort. Start after demo is verified.

## Artifacts

- Branch: `dev`
- Demo: `demo/app.py` (Gradio, 3 tabs: Predict, Calibrate, Monitor)
- Paper draft: `docs/paper/draft.md` (v0.2, updated citations)
- Research: `RESEARCH.md` (updated with 2025-2026 findings)
- Remaining work: `docs/remaining-work.md` (elevation roadmap)
- Eval script: `benchmarks/eval.py`
- Handoff: This file
