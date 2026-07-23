# Conformal RAG — Gradio Demo

## Quick Start

```bash
# From project root
set PYTHONPATH=src
pip install gradio
python demo/app.py
```

Opens at `http://localhost:7860`. For a public link, set `share=True` in `demo.launch()`.

## What the Demo Shows

### Tab 1: Predict
1. **Init Engine** — Set alpha (FDR target) and enable/disable Online ACI
2. **Index Documents** — Load sample docs or paste your own
3. **Query & Risk Check** — Enter a query, optionally provide context/response, run prediction
4. **Result** — See verdict (ACCEPTED/ABSTAINED), scoring breakdown, online ACI stats, FDR monitor

### Tab 2: Calibrate
- Provide labeled calibration data (query, context, response, label)
- See threshold, estimated FDR, coverage stats

### Tab 3: Monitor
- Record labeled observations for live FDR monitoring
- Track production performance over time

## Architecture

```
User Input → ConformalRAGEngine → ScoringPipeline (NLI + optional LLM Judge)
                                      ↓
                              OnlineConformal (adaptive threshold)
                                      ↓
                              FDRMonitor (live tracking)
                                      ↓
                              ConformalResult (verdict + CI + breakdown)
```

## Key Features Demonstrated
- **FDR Guarantee** — P(hallucination | not abstained) ≤ α
- **Prediction Sets** — Confidence intervals on risk scores
- **Online ACI** — Adaptive threshold that converges to target FDR
- **Per-Domain Calibration** — Different alpha per domain
- **Live Monitoring** — Real-time FDR tracking
