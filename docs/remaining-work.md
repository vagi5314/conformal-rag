# Remaining Work — Conformal RAG Platform

**Branch:** `dev` (active)  
**Last Session:** 2026-07-23 — 10 critical fixes verified, 105/105 tests passing

---

## 1. Gradio Demo UI (P0 — Demo Centerpiece)

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Build Gradio app with prediction set visualization | Working engine + scoring | 4-8h |
| Deploy to HuggingFace Spaces | Gradio app complete | 2-4h |
| Add live benchmark results to demo | Benchmark run complete | 2h |

**Why:** Every award-winning project has a live demo. Gradio gives `share=True` public URL instantly.

---

## 2. Run Benchmarks with LLM Judge

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Run `python -m benchmarks.runner --llm-judge --limit 50` on clean machine | API keys (Groq, Together, SiliconFlow), no AppLocker DLL block | 5 min |
| Fill paper Section 6 with real FDR/coverage/latency/cost numbers | Above run complete | 30 min |

**Why:** Without this, the paper's results section has gap rows. NLI-only path is conservative — RAGTruth abstains 100%.

---

## 3. Chrome Extension (P1 — "Wow" Factor)

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Build Chrome MV3 extension shell (content script + service worker) | None | 4-6h |
| Wire to FastAPI backend for conformal verdict | Backend deployed | 4h |
| UI polish: verdict badge + coverage guarantee display | Shell complete | 6-8h |
| Testing + Chrome Web Store prep | All above | 3h |

**Why:** No existing fact-checking extension offers statistical coverage guarantees. Unique selling point.

---

## 4. LangChain/LlamaIndex Wrapper (P1 — Ecosystem)

| Task | Prerequisites | Effort |
|------|---------------|--------|
| LangChain `BaseRetriever` wrapper with conformal metadata | Working client SDK | 6h |
| LlamaIndex `QueryEngine` wrapper | Working client SDK | 6h |
| Tests + docs | Wrappers complete | 4h |

**Why:** No existing conformal prediction integration in either framework. Fills genuine ecosystem gap.

---

## 5. Deploy to Production

| Task | Prerequisites | Effort |
|------|---------------|--------|
| `docker compose up -d` (verify all services healthy) | Docker Desktop | 5 min |
| Fly.io deploy (fly.toml ready) | Account + DB provisioned | 25 min |
| HuggingFace Spaces deploy | Gradio app ready | 5 min |

---

## 6. Publication

| Task | Prerequisites | Effort |
|------|---------------|--------|
| **UncertaiNLP @ EMNLP 2026** — deadline Aug 7, 2026 | LLM-judge benchmark results, ablation study | 2-3 weeks |
| COPA 2026 — deadline passed (May 20) | Missed, target next year | — |
| arXiv preprint | Paper draft filled | 2-4h after draft |

**Why:** UncertaiNLP explicitly lists "conformal prediction" as a topic. Workshop paper is easier to accept, still peer-reviewed.

---

## Summary Table

| Priority | Task | Blocked By | Effort |
|----------|------|------------|--------|
| 🔴 P0 | Gradio demo + HuggingFace Space | None | 6-12h |
| 🔴 P0 | Benchmark run with LLM judge | Clean machine + API keys | 5 min + 30 min |
| 🟡 P1 | Chrome extension | Backend deployed | 15-25h |
| 🟡 P1 | LangChain/LlamaIndex wrapper | Working client SDK | 12-20h |
| 🟡 P1 | Fly.io deploy | Account + DB setup | 25 min |
| 🟢 P2 | Publication (UncertaiNLP) | Benchmark results | 2-3 weeks |
| 🟢 P2 | Slack/Discord bot | Backend deployed | 15-20h |

**Total P0+P1: ~37-63h (1-2 weeks focused work)**
