"""Conformal RAG — Gradio Demo

Run:  set PYTHONPATH=src && python demo/app.py
Or:   demo/start.bat
"""

from __future__ import annotations

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import gradio as gr
import numpy as np

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import Document, ScoringBreakdown
from conformal_rag.retrieval import HybridRetriever
from conformal_rag.scoring import ScoringPipeline
from conformal_rag.online import OnlineConformal
from conformal_rag.monitoring import FDRMonitor

_engine: ConformalRAGEngine | None = None
_retriever: HybridRetriever | None = None
_documents: list[Document] = []
_calibrated: bool = False

SAMPLE_DOCS = [
    "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms.",
    "The capital of France is Paris. Paris is the most populous city in France, with a population of over 2 million people.",
    "Conformal prediction is a framework for constructing prediction sets with guaranteed coverage. It provides distribution-free uncertainty quantification.",
    "DeBERTa-v3 is a state-of-the-art transformer model for natural language understanding. It achieves top results on NLU benchmarks.",
    "FastAPI is a modern web framework for building APIs with Python. It provides automatic OpenAPI documentation and async support.",
    "The Eiffel Tower was built in 1889 for the World's Fair. It stands 330 meters tall and was the tallest structure in the world until 1930.",
    "Conformal Risk Control (CRC) extends conformal prediction to control the False Discovery Rate (FDR) in classification and sequence tasks.",
    "Sentence Transformers are used for computing dense embeddings of text. The e5-base-v2 model produces 768-dimensional embeddings.",
]


def init_engine(alpha: float = 0.05, use_online: bool = True) -> str:
    global _engine, _retriever, _documents, _calibrated
    online = OnlineConformal(alpha=alpha) if use_online else None
    _engine = ConformalRAGEngine(alpha=alpha, online=online)
    _retriever = HybridRetriever()
    _documents = []
    _calibrated = False
    return f"alpha={alpha}  |  online={'ON' if use_online else 'OFF'}  |  ready"


def index_documents(text: str) -> str:
    global _documents, _retriever
    if _engine is None:
        init_engine()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return "No documents to index."
    _documents = [Document(text=l) for l in lines]
    _retriever.index(_documents)
    return f"indexed {len(_documents)} documents"


def index_samples() -> str:
    global _documents, _retriever
    if _engine is None:
        init_engine()
    _documents = [Document(text=l) for l in SAMPLE_DOCS]
    _retriever.index(_documents)
    return f"indexed {len(_documents)} sample documents"


def calibrate_engine(queries: str, contexts: str, responses: str, labels: str) -> str:
    global _calibrated
    if _engine is None:
        init_engine()
    q_list = [l.strip() for l in queries.strip().split("\n") if l.strip()]
    c_list = [l.strip() for l in contexts.strip().split("\n") if l.strip()]
    r_list = [l.strip() for l in responses.strip().split("\n") if l.strip()]
    l_list = [int(l.strip()) for l in labels.strip().split("\n") if l.strip()]
    if not (len(q_list) == len(c_list) == len(r_list) == len(l_list)):
        return f"error: field count mismatch — {len(q_list)} queries, {len(c_list)} contexts, {len(r_list)} responses, {len(l_list)} labels"
    stats = _engine.calibrate(q_list, c_list, r_list, l_list)
    _calibrated = True
    return (
        f"samples:       {stats.n_samples}\n"
        f"hallucinations: {stats.n_hallucinations}\n"
        f"threshold:     {stats.threshold:.4f}\n"
        f"estimated fdr: {stats.estimated_fdr:.4f}\n"
        f"coverage:      {stats.coverage:.2%}\n"
        f"alpha:         {stats.alpha}"
    )


def predict(query: str, context: str, response: str, use_retrieval: bool) -> tuple:
    if _engine is None:
        return "engine not initialized", "", ""

    if use_retrieval and _retriever and _documents:
        retrieved = _retriever.retrieve(query)
        context_str = "\n".join(retrieved) if retrieved else context
    else:
        context_str = context

    if not response:
        return "provide a response to risk-check", "", ""

    result = _engine.predict(query, context_str, response)

    if result.abstained:
        verdict = "ABSTAINED"
        verdict_detail = "risk exceeds threshold — response rejected"
    else:
        verdict = "ACCEPTED"
        verdict_detail = "risk within bound — response passes"

    bd = result.scoring_breakdown
    breakdown_text = (
        f"risk score:     {result.risk_score:.4f}\n"
        f"risk bound:     {result.risk_bound:.4f}\n"
        f"abstained:      {result.abstained}\n"
        f"confidence:     ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})\n"
        f"latency:        {result.latency_ms:.1f}ms\n"
        f"domain:         {result.domain}\n"
        f"\n"
        f"nli score:      {bd.nli_score:.4f}\n"
        f"llm judge:      {bd.llm_judge_score:.4f if bd.llm_judge_score else 'n/a'}\n"
        f"final score:    {bd.final_score:.4f}\n"
        f"nli latency:    {bd.nli_latency_ms:.1f}ms\n"
        f"used llm:       {bd.used_llm_judge}"
    )

    online_text = "online aci:     not enabled"
    if _engine.online is not None:
        stats = _engine.online.get_stats()
        online_text = (
            f"online aci:\n"
            f"  threshold:   {stats['threshold']:.4f}\n"
            f"  level:       {stats.get('level', 'n/a')}\n"
            f"  window:      {stats['window_size']}\n"
            f"  updates:     {stats['n_updates']}"
        )

    mon_metrics = _engine.fdr_monitor.get_metrics()
    monitor_text = (
        f"\nfdr monitor:\n"
        f"  predictions:  {mon_metrics['total_predictions']}\n"
        f"  fdr:          {mon_metrics['fdr']:.4f}\n"
        f"  coverage:     {mon_metrics['coverage']:.2%}\n"
        f"  abstention:   {mon_metrics['abstention_rate']:.2%}"
    )

    status = f"**{verdict}** — {verdict_detail}"
    return status, breakdown_text, online_text + "\n" + monitor_text


def record_observation(query: str, context: str, response: str, actual_hallucination: bool) -> str:
    if _engine is None:
        return "engine not initialized"
    _engine.record(query, context, response, actual_hallucination)
    metrics = _engine.fdr_monitor.get_metrics()
    return (
        f"recorded.\n"
        f"total:    {metrics['total_predictions']}\n"
        f"fdr:      {metrics['fdr']:.4f}\n"
        f"coverage: {metrics['coverage']:.2%}"
    )


# ---------------------------------------------------------------------------
# CSS — dark technical theme, no gradients, no decoration
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* -- base -- */
.gradio-container { max-width: 1100px !important; }

/* -- header -- */
.demo-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #27272a;
    margin-bottom: 1.5rem;
}
.demo-header h1 {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #fafafa !important;
    margin: 0 0 0.5rem 0 !important;
}
.demo-header .tagline {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #71717a !important;
    letter-spacing: 0.02em !important;
}
.demo-header .formula {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #34d399 !important;
    margin-top: 0.75rem !important;
    padding: 0.5rem 0.75rem;
    background: #052e16;
    border: 1px solid #14532d;
    display: inline-block;
}

/* -- section labels -- */
.section-label {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em !important;
    color: #52525b !important;
    margin-bottom: 0.5rem !important;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #27272a;
}

/* -- verdict boxes -- */
.verdict-accepted {
    background: #052e16 !important;
    border: 1px solid #14532d !important;
    border-radius: 4px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
}
.verdict-abstained {
    background: #2a0a0a !important;
    border: 1px solid #7f1d1d !important;
    border-radius: 4px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
}

/* -- data panels -- */
.data-panel {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.6 !important;
    background: #09090b !important;
    border: 1px solid #27272a !important;
    border-radius: 4px !important;
    padding: 1rem !important;
    color: #d4d4d8 !important;
}

/* -- buttons -- */
.primary-btn {
    background: #18181b !important;
    border: 1px solid #3f3f46 !important;
    color: #fafafa !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em !important;
    border-radius: 3px !important;
}
.primary-btn:hover {
    background: #27272a !important;
    border-color: #52525b !important;
}

/* -- footer -- */
.demo-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #27272a;
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #52525b !important;
}
"""

# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Conformal RAG",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="zinc",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ),
) as demo:

    # -- Header --
    gr.HTML("""
    <div class="demo-header">
        <h1>conformal rag</h1>
        <div class="tagline">FDR-guaranteed hallucination bounds for RAG systems</div>
        <div class="formula">P(hallucination | not abstained) &le; &alpha;</div>
    </div>
    """)

    with gr.Tabs():

        # ---------------------------------------------------------------
        # Tab 1: Predict
        # ---------------------------------------------------------------
        with gr.Tab("predict", id="predict"):
            gr.HTML('<div class="section-label">engine setup</div>')
            with gr.Row():
                alpha_slider = gr.Slider(
                    0.001, 0.2, value=0.05, step=0.005,
                    label="alpha (FDR target)",
                )
                online_check = gr.Checkbox(value=True, label="online ACI")
                init_btn = gr.Button("init", elem_classes=["primary-btn"], scale=0)
                init_status = gr.Textbox(label="status", interactive=False, elem_classes=["data-panel"])

            gr.HTML('<div class="section-label">index documents</div>')
            with gr.Row():
                sample_btn = gr.Button("load samples", elem_classes=["primary-btn"], scale=0)
                index_btn = gr.Button("index", elem_classes=["primary-btn"], scale=0)
                index_status = gr.Textbox(label="status", interactive=False, elem_classes=["data-panel"])
            doc_input = gr.Textbox(
                label="documents (one per line)",
                lines=5,
                placeholder="paste documents here...",
            )

            gr.HTML('<div class="section-label">query & risk check</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    query_input = gr.Textbox(label="query", placeholder="What is the capital of France?")
                    context_input = gr.Textbox(label="context (leave empty for auto-retrieval)", lines=2)
                    response_input = gr.Textbox(label="response to risk-check", lines=2)
                    use_retrieval = gr.Checkbox(value=True, label="auto-retrieve from indexed docs")
                with gr.Column(scale=1):
                    predict_btn = gr.Button("run prediction", elem_classes=["primary-btn"])
                    verdict_output = gr.Markdown(label="verdict")

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-label">scoring breakdown</div>')
                    breakdown_output = gr.Textbox(lines=12, interactive=False, elem_classes=["data-panel"], show_label=False)
                with gr.Column():
                    gr.HTML('<div class="section-label">online aci + fdr monitor</div>')
                    stats_output = gr.Textbox(lines=12, interactive=False, elem_classes=["data-panel"], show_label=False)

            init_btn.click(fn=init_engine, inputs=[alpha_slider, online_check], outputs=[init_status])
            sample_btn.click(fn=index_samples, outputs=[index_status])
            index_btn.click(fn=index_documents, inputs=[doc_input], outputs=[index_status])
            predict_btn.click(
                fn=predict,
                inputs=[query_input, context_input, response_input, use_retrieval],
                outputs=[verdict_output, breakdown_output, stats_output],
            )

        # ---------------------------------------------------------------
        # Tab 2: Calibrate
        # ---------------------------------------------------------------
        with gr.Tab("calibrate", id="calibrate"):
            gr.HTML('<div class="section-label">calibration data</div>')
            gr.Markdown(
                "Provide labeled calibration data. Labels: `0` = correct, `1` = hallucination."
            )
            with gr.Row():
                cal_queries = gr.Textbox(label="queries", lines=6)
                cal_contexts = gr.Textbox(label="contexts", lines=6)
                cal_responses = gr.Textbox(label="responses", lines=6)
                cal_labels = gr.Textbox(label="labels (0 or 1)", lines=6)
            cal_btn = gr.Button("calibrate", elem_classes=["primary-btn"])
            cal_output = gr.Textbox(label="results", lines=8, interactive=False, elem_classes=["data-panel"])

            cal_btn.click(
                fn=calibrate_engine,
                inputs=[cal_queries, cal_contexts, cal_responses, cal_labels],
                outputs=[cal_output],
            )

        # ---------------------------------------------------------------
        # Tab 3: Monitor
        # ---------------------------------------------------------------
        with gr.Tab("monitor", id="monitor"):
            gr.HTML('<div class="section-label">live fdr monitoring</div>')
            gr.Markdown("Record labeled observations to track production FDR.")
            with gr.Row():
                mon_query = gr.Textbox(label="query")
                mon_context = gr.Textbox(label="context")
                mon_response = gr.Textbox(label="response")
            with gr.Row():
                mon_hallucination = gr.Checkbox(label="actually a hallucination?", value=False)
                mon_btn = gr.Button("record", elem_classes=["primary-btn"])
            mon_output = gr.Textbox(label="status", lines=5, interactive=False, elem_classes=["data-panel"])

            mon_btn.click(
                fn=record_observation,
                inputs=[mon_query, mon_context, mon_response, mon_hallucination],
                outputs=[mon_output],
            )

    # -- Footer --
    gr.HTML("""
    <div class="demo-footer">
        conformal rag  |  deBERTa-v3 nli  |  litellm  |  mapie  |  fastapi
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
