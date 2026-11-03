# Archive

Superseded files, kept so nothing is lost and any of them can be restored with
`git mv archive/<file> .`.

| File | Why it was archived | Replaced by |
|---|---|---|
| `quickstart.py` | Short CLI smoke test (index 3 docs, calibrate on 6 samples, score 2 answers). The web console does the same thing with a real interface and the seeded calibration set. | `frontend/` + `scripts/seed_demo.py` |
| `docs/zeroth-review.html` | One-off slide deck from the first project review. It referenced nothing and was never part of the running app. | `frontend/` (the live console) |
| `docs/dashboard.html`, `docs/critique-dashboard.html` | July progress dashboards from a superseded review round; neither is linked from the app or the docs. | `docs/first-review.html`, `PROJECT_REPORT.md` |
| `frontend-ember-v1/` | The first cut of the web console, including `js/charts.js`. The chart code was folded into `frontend/js/app.js`. | `frontend/` |

Everything else in the repository root is live: `demo.py` and `panel_demo.py`
back the viva, `demo_data.py` holds their datasets, and `run.cmd` starts the
app.
