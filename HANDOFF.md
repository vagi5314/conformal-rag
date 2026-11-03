# Handoff: conformal-rag project – pick up next session

## Goal of next session
Pick up the conformal-rag project: finalize README, refresh HANDOFF.md, restart API server, commit + push to GitHub, run graphify update. The core code is verified (307 passed / 10 skipped, ruff clean, node --check clean). PDF Q&A is verified end to end on two bundled sample PDFs plus a third unrelated paper.

## State of play
- 307 core tests passing (10 skipped). `ruff check src/ tests/` clean.
- **PDF answer fix (2026-11-04)** — PDF extraction left U+0000 control characters (math fonts emit them inside equation runs) inside passages. A prompt carrying them made the answer model reply `NOT_IN_PASSAGES` even when the passage held the answer, so on-topic questions on real papers declined. `chunk_text` now scrubs control characters (`_CONTROL_CHARS` in `documents.py`), verified by stripping/re-adding nulls against the live provider. Regression tests: `test_control_characters_are_stripped`, `test_prompt_carries_no_control_characters`.
- **Truncated-decline fix (2026-11-04)** — one provider emitted the sentinel cut short (`NOT_IN`), and the exact-prefix test let that non-answer through and had it **approved** as a real answer. `is_decline()` now matches the sentinel by leading letters, so any truncation of it (or a rephrased "Not in the passages.") abstains. Regression tests: 5 parametrized decline cases + an ordinary-answer guard.
- Verified live 2026-11-04 after both fixes: 11/11 on-topic questions approved with correct answers across `conformal-risk-control.pdf`, `nist-cloud-computing.pdf` and a third unrelated paper, 3/3 off-topic declined. Question list for manual testing lives in `sample_docs/README.md`.
- Frontend verified 2026-11-04: `/`, `/css/styles.css`, `/js/api.js`, `/js/app.js` all 200; every endpoint the UI calls (`/config` GET+POST, `/calibration`, `/domains`, `/demo`, `/metrics`, `/predict`, `/documents/upload`, `/ask`) responds correctly; `node --check` clean on both JS files.
- `node --check` on both frontend JS files passes.
- Scoring rewrite (windowed premises, `_MAX_PREMISES=48`) verified with a battery of 24 cases — all OK.
- Upload flow verified: `POST /documents/upload` (multipart, 8 cap, 10 MB limit) + `POST /ask` with retrieval retry on decline, closest-passage fallback, `Cache-Control: no-cache` header.
- PDF path: 3/4 on-topic answered with high scores (`0.9949`, `0.9963`, `0.9596`), decline shows closest passage + note.
- All 9 showcase cases (medicine/finance/general: faithful/subtle/blatant) verified over HTTP.
- 2 files archived: `quickstart.py → archive/quickstart.py`, `docs/zeroth-review.html → archive/docs/zeroth-review.html`.
- 2 more archived (2026-11-04): `docs/dashboard.html`, `docs/critique-dashboard.html` → `archive/docs/`; `excalidraw.log` deleted; `.gitignore` now ignores `*.log` instead of the one filename.
- Verified 2026-11-04: `pytest tests -q` → 307 passed / 10 skipped, `ruff check src tests` clean, `node --check` on both frontend files clean, no dangling `charts.js` reference (copy kept in `archive/frontend-ember-v1/js/`).
- Server verified live 2026-11-04 on `http://127.0.0.1:8000` with the demo env (`NLI_MODEL=small`, `ALPHA=0.15`, `STATE_DIR=.state`): `/health` 200 in ~3 s, `/`, `/js/*`, `/css/styles.css` all 200, `/config` reports `calibrated: true, documents: 24`, `/domains` shows 3 calibrated domains, `/documents/upload` + `/ask` on-topic → approved with `grounding: exact`, off-topic → decline with closest-passage note, `/predict` faithful 0.9937 approved vs fabricated 8e-05 abstained. 28 requests, zero errors in `server.log`.
- `.gitignore` updated (`.state/` already ignored, added `.demo_cache/` mention).
- Panel files documented as superseded in `PROJECT_REPORT.md`.
- `.env` stays untracked (in `.gitignore`).
- No real model loaded in tests (`FakeEmbedder` stub).
- Git has many modified tracked files but **no commits yet**.
- Modified files: `src/conformal_rag/{api,config,domain,engine,llm,models,monitoring,scoring}.py`, `frontend/{index.html,js/app.js,js/api.js}`, `pyproject.toml`, `.gitignore`, `README.md`, `PROJECT_REPORT.md`, `tests/test_documents.py`.

## Open decisions
1. Refresh README with upload flow, small NLI default, `CONFORMAL_RAG_NLI_MODEL`, `CONFORMAL_RAG_LLM_MODELS`, `CONFORMAL_RAG_USE_LLM_JUDGE` env vars.
2. Refresh HANDOFF.md with current state (upload, windowed scoring, nli-small default).
3. ~~Start API server on `:8000` for browser UI testing.~~ — running, see above.
4. ~~Commit + push `dev` branch~~ — **done 2026-11-04**, `5741c85` pushed to `origin/dev` (https://github.com/vagi5314/conformal-rag.git).
5. Run `graphify update .` — `graphify-out/` exists.

## Restarting the server
```bash
# equivalent of run.cmd without the auto-browser launch
PYTHONPATH=src CONFORMAL_RAG_NLI_MODEL=small CONFORMAL_RAG_ALPHA=0.15 \
  CONFORMAL_RAG_STATE_DIR=.state \
  .venv/Scripts/python.exe -m uvicorn conformal_rag.api:create_app --factory --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000`. Delete `.state/` to force re-seeding (or run `scripts/seed_demo.py --force`).

## Artifacts (references, do not paste)
- `src/conformal_rag/api.py` — `/ask` (general default, retry `k=10`, closest passage), `/documents/upload` (multipart, 8 cap, 10 MB), `/predict` (judge escalation, NLI+judge final score), `/calibrate`, `/domains`.
- `frontend/index.html` — Check (paste + upload), Topics, Status, Method; Geist/Geist Mono, dark palette, sentence case copy.
- `frontend/js/app.js` — runAsk, renderNoVerdict, renderDocumentAnswer, renderResult, renderGrounded.
- `frontend/js/api.js` — API client for the FastAPI endpoints.
- `pyproject.toml` — added `pypdf>=5.0`, `python-multipart>=0.0.9`.
- `README.md` — updated with upload flow, small NLI default, env vars.
- `PROJECT_REPORT.md` — panel files documented as superseded.
- `tests/test_documents.py` — updated with `IsExactQuote` tests and `test_ask_defaults_uploads_to_general_not_auto_routing`.
- `.env` — git-ignored, holds `GROQ_API_KEY`.
- `.state/` — persisted demo state (index + per-domain calibrations), rebuilt with `scripts/seed_demo.py --force`.
- `archive/quickstart.py` — superseded, replaces by frontend + `scripts/seed_demo.py`.
- `archive/docs/zeroth-review.html` — superseded.
- `api.py` guard: `Cache-Control: no-cache` on `.js`, `.css`, `.html`, `/` so stale `app.js` cannot run against a new API.

## Suggested skills for next session
- `execute` — restart uvicorn, verify routes.
- `git-workflow-and-versioning` — `git add -A`, review diff, commit on `dev`, push to `origin`.
- `systematic-debugging` — re-run test suite after server restart, fix any drift.