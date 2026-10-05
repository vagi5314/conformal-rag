# Panel Demo Runbook — Conformal RAG

Two demo paths. Both verify the same claim; pick based on room setup.
**Run either once before the panel** so model weights and scores are cached.

---

## Demo A — CLI (no browser, no network needed) ⭐ RECOMMENDED

```bash
cd C:\Users\ELCOT\Desktop\Final_year
.venv\Scripts\python.exe demo.py
```

First run: ~3.5 min (downloads + scores 63 pairs on CPU). After that: **~13 seconds**.

**Verified output on this machine:**

```
c_max = floor((n+1)a)-1 = 1
threshold               = 0.0699
hallucination scores    = 0.792, 0.070, 0.026, 0.005, 0.005, 0.003 ...
faithful score mean     = 0.841
in-sample coverage      = 51.2%

total tested            25
truly wrong             13
approved                12
approved AND wrong      0
coverage                48.0%
empirical FDR           = 0.0%
target alpha            5.0%
verdict                 HOLD
total runtime: 12.6s
```

**What to say while it runs:**

| Moment | Line | Point to make |
|---|---|---|
| Step 1 | retrieval | "Two indexes — meaning and keywords — fused by rank. Neither alone is enough." |
| Step 2 | the claim | **"This is a statement about a RATE, not about one answer."** Read it aloud slowly. This is the whole project. |
| Step 3 | `c_max` | "It approves only the top `c_max` hallucination scores. The +1 in `(n+1)α` is the conformal correction — that's what makes it provable on unseen data instead of merely observed." |
| Step 3 | warning | It prints "43 samples, 100 recommended". **Don't hide this.** "It tells you when it can't back its promise." |
| Step 4 | the table | Point at a `0.7912 approve ok` next to `0.0050 abstain wrong`. Clean separation. |
| Step 4 | verdict | "Zero approved-and-wrong out of 25. That's the guarantee holding." |
| Step 5 | ACI | "This part is a heuristic, not a certificate. It trades the proof for adaptivity." |

**If a panel member challenges the result** — the strongest available answer:

> "25 examples is small; one extra violation is worth 4%. The bound is on the rate, and the code refuses to certify anything below ⌈1/α⌉−1 = 19 samples. For a real claim you need the 100+ the warning asks for, on real benchmarks — which is exactly what's blocked on this machine."

**Optional contrast (only if there's time and it strengthens your point):**

```bash
.venv\Scripts\python.exe demo.py --small
```

Uses the fast weak preset → prints **VIOLATED, 21.4% FDR**. Good for showing the guarantee is
not decoration: a weak scorer *does* break it, and the system reports that honestly.

---

## Demo B — Web UI (Ember Console)

```bash
cd C:\Users\ELCOT\Desktop\Final_year
set PYTHONPATH=src
.venv\Scripts\python.exe -m uvicorn conformal_rag.api:create_app --factory --port 8000
```

Open **http://localhost:8000**. First prediction loads the NLI model — allow ~10s and say so
in advance so it doesn't look hung.

### Scripted walkthrough

**1. Check an Answer** (view 01)
- Click **Load example sources** → **Save sources**
- Paste into Question:
  ```
  What is conformal prediction?
  ```
- Paste into AI's answer (deliberately wrong):
  ```
  Conformal prediction assumes the data follows a Gaussian distribution and
  produces a confidence interval around a single prediction.
  ```
- Leave **auto-retrieve** on → **Check the answer**
- Expect: **✗ Not sure**, dial near zero, ember threshold tick at 7.0%
- **The hook:** the amber banner *"Not taught yet — this verdict uses a rough default
  line and carries no statistical guarantee."* The UI never pretends.

**2. Teach It** (view 02)
- Click **Fill in a worked example** → **Teach it**
- Expect: threshold line appears, coverage and FDR reported
- **Say:** "That example is tiny. It warned me. With 100+ examples the line would be
  trustworthy."

**3. Re-check the same answer** → now **✓ Trustworthy**? No — it should now *abstain*, because
the calibrated threshold is stricter than the 0.5 default.
- **This is the best single moment in the demo.** The system got *more* conservative after
  learning, not less.

**4. Live Status** (view 03)
- Report a hallucination via the feedback box → FDR and abstention charts move
- **Say:** "The FDR here is computed only from feedback where a true label arrived. No fake labels."

**5. Topics** (view 04)
- Register `medicine` with tolerance 0.001 → "very strict"
- **Say:** "Per-domain calibration. Different risk tolerance per subject."

---

## Hardening options (only if the panel asks about production readiness)

```bash
set CONFORMAL_RAG_API_KEY=some-secret
set CONFORMAL_RAG_RATE_LIMIT_PER_MINUTE=60
set CONFORMAL_RAG_STATE_DIR=state
set CONFORMAL_RAG_REDIS_URL=redis://localhost:6379/0
set CONFORMAL_RAG_QDRANT_URL=http://localhost:6333
```

```bash
docker compose up -d qdrant redis
```

All opt-in. Unset → zero-config local behaviour, which is what the demo above uses.

---

## Likely questions

**"How do you know it doesn't just abstain on everything?"**
Coverage is reported alongside FDR precisely so this is checkable. Demo A shows 48% coverage
with 0 violations — it's approving half of everything. Abstaining on everything would show
0% coverage, which satisfies the bound trivially and is useless. Both numbers are always shown.

**"Is 0 violations just a small sample?"**
Yes — and I say so first. 0/25 gives an upper 95% bound of ~11%, not 5%. The honest claim is
"consistent with the guarantee", not "proves it". Proving it needs n ≥ 100 on real benchmarks.

**"What if the model gets worse over time?"**
ACI (Demo A step 5) raises the threshold when it observes errors. It is a heuristic — it gives
long-run average control, not a finite-sample certificate. Drift detection per domain is in the
Topics view.

**"What if your calibration data isn't representative?"**
Then the guarantee is void and no code can tell you. That's the exchangeability assumption,
and it's the honest weak point of the whole approach. It's stated in the engine docstring and
in the README.

**"Why not just use a bigger LLM?"**
Size doesn't give you a *bound*. A larger model lowers the error rate; conformal calibration
tells you what rate you're actually getting, and lets you trade it against coverage explicitly.

---

## Pre-panel checklist

- [ ] `demo.py` run once — cache warm, ~13s thereafter
- [ ] Web UI reachable and `uvicorn` started ahead of time
- [ ] Browser zoom ≥ 110% (small mono text)
- [ ] Terminal window sized for 78 columns
- [ ] HF token in env if the network is slow: `set HF_TOKEN=...`
- [ ] Panels/monitors **not** connected (default is charcoal-on-charcoal)

## Honest limitations — state these before you're asked

1. **Benchmarks not run with `--llm-judge`.** Blocked by an AppLocker policy on this machine
   that prevents the `datasets` DLL from loading. NLI-only runs pass trivially because the
   model abstains on 100% of RAGTruth. The paper's results table has no real numbers yet.
2. **Demo data is hand-written** (43 calibration + 25 held-out), not a public benchmark. Enough
   to show the mechanism; not enough to support the α claim statistically.
3. **α is clamped to [0.001, 0.2]** — below 0.001 you need >1000 calibration samples.
4. **The guarantee is marginal.** `E[P(approved ∧ wrong)] ≤ α`, not `P(wrong | approved) ≤ α`.
   The conditional form isn't achievable for CRC; the paper says so explicitly.
5. **ACI is a heuristic** and once active it *replaces* the offline certificate rather than
   composing with it.