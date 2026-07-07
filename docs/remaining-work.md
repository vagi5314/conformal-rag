# Remaining Work — Conformal RAG Platform

**Branch:** `dev` (active), `main` (releases)  
**Last Commit:** `e34fa0f` — "feat: complete Phases 2-4"  
**Tests:** 161/161 passing

---

## 1. LLM Judge Benchmarking (unblocks RAGTruth coverage)

The NLI-only pipeline is conservative — RAGTruth abstains 100%. The LLM judge resolves the coverage gap.

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Run `python -m benchmarks.runner --llm-judge --limit 50` | API keys in Windows Credential Manager (`keyring`): Groq, Together, SiliconFlow | 5 min |
| Fill paper `docs/paper/draft.md` Section 6 with real FDR/coverage/latency/cost numbers | Above run complete | 30 min |

**Key:** Without this, the paper's results section has gap rows.

---

## 2. Deployment (Docker Compose)

| Task | Prerequisites | Effort |
|------|---------------|--------|
| `docker compose up -d` from project root | Docker Desktop running on host | 2 min |
| Verify all 4 services (Qdrant, Redis, PostgreSQL, app) come up healthy | Docker running | 5 min |
| `http://localhost:8000/health` returns `{"status":"ok"}` | Services healthy | 1 min |

**Gotcha:** `docker-compose.yml` assumes CPU architecture. The NLI model loading needs ~2GB RAM on the app container. If OOM on default Docker Desktop, increase memory in Docker settings.

---

## 3. Deploy to Fly.io

| Task | Prerequisites | Effort |
|------|---------------|--------|
| `flyctl auth login` | Fly.io account, flyctl installed | 5 min |
| `flyctl launch` (uses `fly.toml`) | Auth'd | 2 min |
| Provision Fly Postgres: `flyctl postgres create` | Fly account | 5 min |
| Provision Fly Redis (Upstash): `flyctl redis create` | Fly account | 3 min |
| Set secrets: `flyctl secrets set CONFORMAL_RAG_POSTGRES_URL=... CONFORMAL_RAG_REDIS_URL=...` | DB + Redis provisioned | 2 min |
| `flyctl deploy` | Everything above | 5 min |

**Gotcha:** The DeBERTa model (~1.6GB) downloads on first `/predict` call. Set `TRANSFORMERS_CACHE` to a persistent volume or expect ~2 min cold start.

---

## 4. PyPI Publication

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Create PyPI API token at https://pypi.org/manage/account/ | PyPI account | 5 min |
| Add `PYPI_TOKEN` secret to GitHub repo: `gh secret set PYPI_TOKEN` | GitHub CLI or web UI | 2 min |
| Tag a release: `git tag v0.1.0 && git push origin v0.1.0` | Clean working tree | 1 min |
| Workflow auto-publishes to PyPI via `.github/workflows/pypi.yml` | Tag pushed | ~3 min |

---

## 5. arXiv Submission

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Fill `docs/paper/draft.md` Section 6 with LLM-judge results | LLM-judge run complete | 30 min |
| Convert `draft.md` to LaTeX (`docs/paper/paper.tex`) | Draft filled | 1–2 hr |
| Add author metadata, abstract, references | LaTeX file complete | 30 min |
| Upload to arXiv | arXiv account, .tex + figures | 15 min |

**Gotcha:** arXiv requires LaTeX source, not Markdown. The `draft.md` is designed to be a 1:1 translation guide — section headers map directly to LaTeX `\section{}`.

---

## 6. Open Source Launch

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Make repo public: GitHub → Settings → Danger Zone → Change visibility | arXiv submission done (pre-print) | 1 min |
| Announce on Twitter / LinkedIn | Repo public | 15 min |

---

## Summary Table

| Priority | Task | Blocked By | Effort |
|----------|------|------------|--------|
| 🔴 P0 | LLM-judge benchmark run | API keys | 5 min |
| 🟡 P1 | Docker compose up | Docker Desktop | 5 min |
| 🟡 P1 | Fly.io deploy | Account + DB setup | 25 min |
| 🟡 P1 | arXiv submission | LLM-judge results | 2.5 hr |
| 🟢 P2 | PyPI publish | — | 10 min |
| 🟢 P2 | Open source launch | arXiv done | 15 min |

**Tags:** `P0` = blocks paper, `P1` = blocks deployment, `P2` = nice-to-have
