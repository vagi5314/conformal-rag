"""Conformal RAG — Gradio Demo

Run:  set PYTHONPATH=src && python demo/app.py
Or:   gradio demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path when run from project root
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

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_engine: ConformalRAGEngine | None = None
_retriever: HybridRetriever | None = None
_documents: list[Document] = []
_calibrated: bool = False

# Sample documents for quick demo
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
    """Initialize the conformal engine."""
    global _engine, _retriever, _documents, _calibrated

    online = OnlineConformal(alpha=alpha) if use_online else None
    _engine = ConformalRAGEngine(alpha=alpha, online=online)
    _retriever = HybridRetriever()
    _documents = []
    _calibrated = False

    return f"Engine initialized with alpha={alpha}, online={'ON' if use_online else 'OFF'}"


def index_documents(text: str) -> str:
    """Index documents from newline-separated text."""
    global _documents, _retriever

    if _engine is None:
        init_engine()

    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return "No documents to index."

    _documents = [Document(text=l) for l in lines]
    _retriever.index(_documents)

    return f"Indexed {len(_documents)} documents. Ready for queries."


def index_samples() -> str:
    """Index the built-in sample documents."""
    global _documents, _retriever

    if _engine is None:
        init_engine()

    _documents = [Document(text=l) for l in SAMPLE_DOCS]
    _retriever.index(_documents)

    return f"Indexed {len(_documents)} sample documents. Try asking: 'What is Python?' or 'What is the capital of France?'"


def calibrate_engine(
    queries: str,
    contexts: str,
    responses: str,
    labels: str,
) -> str:
    """Calibrate from newline-separated data."""
    global _calibrated

    if _engine is None:
        init_engine()

    q_list = [l.strip() for l in queries.strip().split("\n") if l.strip()]
    c_list = [l.strip() for l in contexts.strip().split("\n") if l.strip()]
    r_list = [l.strip() for l in responses.strip().split("\n") if l.strip()]
    l_list = [int(l.strip()) for l in labels.strip().split("\n") if l.strip()]

    if not (len(q_list) == len(c_list) == len(r_list) == len(l_list)):
        return f"Error: All fields must have the same number of lines. Got {len(q_list)} queries, {len(c_list)} contexts, {len(r_list)} responses, {len(l_list)} labels."

    stats = _engine.calibrate(q_list, c_list, r_list, l_list)
    _calibrated = True

    return (
        f"Calibration complete!\n"
        f"  Samples: {stats.n_samples}\n"
        f"  Hallucinations: {stats.n_hallucinations}\n"
        f"  Threshold: {stats.threshold:.4f}\n"
        f"  Estimated FDR: {stats.estimated_fdr:.4f}\n"
        f"  Coverage: {stats.coverage:.2%}\n"
        f"  Alpha: {stats.alpha}"
    )


def predict(
    query: str,
    context: str,
    response: str,
    use_retrieval: bool,
) -> tuple:
    """Run conformal prediction."""
    if _engine is None:
        return "Engine not initialized. Click 'Init Engine' first.", "", ""

    if use_retrieval and _retriever and _documents:
        retrieved = _retriever.retrieve(query)
        context_str = "\n".join(retrieved) if retrieved else context
    else:
        context_str = context

    if not response:
        return "Please provide a response to risk-check.", "", ""

    result = _engine.predict(query, context_str, response)

    # Risk verdict
    if result.abstained:
        verdict = "ABSTAINED — Risk too high, response rejected"
        verdict_color = "red"
    else:
        verdict = "ACCEPTED — Response passes risk check"
        verdict_color = "green"

    # Breakdown
    bd = result.scoring_breakdown
    breakdown_text = (
        f"Risk Score: {result.risk_score:.4f}\n"
        f"Risk Bound (alpha): {result.risk_bound:.4f}\n"
        f"Abstained: {result.abstained}\n"
        f"Confidence Interval: ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})\n"
        f"Latency: {result.latency_ms:.1f}ms\n"
        f"Domain: {result.domain}\n"
        f"\nScoring Breakdown:\n"
        f"  NLI Score: {bd.nli_score:.4f}\n"
        f"  LLM Judge Score: {bd.llm_judge_score:.4f if bd.llm_judge_score else 'N/A'}\n"
        f"  Final Score: {bd.final_score:.4f}\n"
        f"  NLI Latency: {bd.nli_latency_ms:.1f}ms\n"
        f"  Used LLM Judge: {bd.used_llm_judge}"
    )

    # Online stats
    online_text = "Online ACI not enabled."
    if _engine.online is not None:
        stats = _engine.online.get_stats()
        online_text = (
            f"Online ACI Stats:\n"
            f"  Threshold: {stats['threshold']:.4f}\n"
            f"  Level (lambda): {stats.get('level', 'N/A')}\n"
            f"  Window Size: {stats['window_size']}\n"
            f"  Updates: {stats['n_updates']}"
        )

    # Monitor stats
    monitor_text = ""
    mon_metrics = _engine.fdr_monitor.get_metrics()
    monitor_text = (
        f"FDR Monitor:\n"
        f"  Total Predictions: {mon_metrics['total_predictions']}\n"
        f"  FDR: {mon_metrics['fdr']:.4f}\n"
        f"  Coverage: {mon_metrics['coverage']:.2%}\n"
        f"  Abstention Rate: {mon_metrics['abstention_rate']:.2%}"
    )

    status = f"**{verdict}**"
    return status, breakdown_text, online_text + "\n\n" + monitor_text


def record_observation(
    query: str,
    context: str,
    response: str,
    actual_hallucination: bool,
) -> str:
    """Record a labeled observation for monitoring."""
    if _engine is None:
        return "Engine not initialized."

    _engine.record(query, context, response, actual_hallucination)
    metrics = _engine.fdr_monitor.get_metrics()

    return (
        f"Recorded! Metrics updated:\n"
        f"  Total: {metrics['total_predictions']}\n"
        f"  FDR: {metrics['fdr']:.4f}\n"
        f"  Coverage: {metrics['coverage']:.2%}"
    )


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Conformal RAG — FDR-Guaranteed Hallucination Bounds",
    theme=gr.themes.Soft(),
    css="""
    .verdict-accepted { color: #16a34a; font-weight: bold; font-size: 1.2em; }
    .verdict-abstained { color: #dc2626; font-weight: bold; font-size: 1.2em; }
    .metric-box { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
    """,
) as demo:
    gr.Markdown(
        """
        # Conformal RAG — FDR-Guaranteed Hallucination Bounds
        **Production-grade conformal prediction for RAG systems.** Mathematically guaranteed
        False Discovery Rate (FDR) bounds on hallucination rates. Instead of heuristic confidence
        scores, you get provable guarantees: *"this answer has ≤5% chance of hallucinating."*
        """
    )

    with gr.Tabs():
        # -------------------------------------------------------------------
        # Tab 1: Predict
        # -------------------------------------------------------------------
        with gr.Tab("Predict", id="predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Setup")
                    alpha_slider = gr.Slider(
                        0.001, 0.2, value=0.05, step=0.005,
                        label="Alpha (FDR target)",
                    )
                    online_check = gr.Checkbox(value=True, label="Enable Online ACI")
                    init_btn = gr.Button("Init Engine", variant="primary")
                    init_status = gr.Textbox(label="Status", interactive=False)

                    gr.Markdown("### Index Documents")
                    sample_btn = gr.Button("Load Sample Documents")
                    doc_input = gr.Textbox(
                        label="Documents (one per line)",
                        lines=8,
                        placeholder="Paste documents here, one per line...",
                    )
                    index_btn = gr.Button("Index")
                    index_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("### Query & Risk Check")
                    query_input = gr.Textbox(
                        label="Query",
                        placeholder="e.g., What is the capital of France?",
                    )
                    context_input = gr.Textbox(
                        label="Context (leave empty to use retrieval)",
                        lines=3,
                        placeholder="Optional: provide context directly",
                    )
                    response_input = gr.Textbox(
                        label="Response to risk-check",
                        lines=3,
                        placeholder="The LLM response to evaluate",
                    )
                    use_retrieval = gr.Checkbox(
                        value=True,
                        label="Use auto-retrieval (query indexed documents)",
                    )
                    predict_btn = gr.Button("Run Conformal Prediction", variant="primary")

                    gr.Markdown("### Result")
                    verdict_output = gr.Markdown(label="Verdict")

                    with gr.Row():
                        with gr.Column():
                            breakdown_output = gr.Textbox(
                                label="Scoring Breakdown",
                                lines=12,
                                interactive=False,
                            )
                        with gr.Column():
                            stats_output = gr.Textbox(
                                label="Online ACI + FDR Monitor",
                                lines=12,
                                interactive=False,
                            )

            # Wire events
            init_btn.click(
                fn=init_engine,
                inputs=[alpha_slider, online_check],
                outputs=[init_status],
            )
            sample_btn.click(fn=index_samples, outputs=[index_status])
            index_btn.click(
                fn=index_documents,
                inputs=[doc_input],
                outputs=[index_status],
            )
            predict_btn.click(
                fn=predict,
                inputs=[query_input, context_input, response_input, use_retrieval],
                outputs=[verdict_output, breakdown_output, stats_output],
            )

        # -------------------------------------------------------------------
        # Tab 2: Calibrate
        # -------------------------------------------------------------------
        with gr.Tab("Calibrate", id="calibrate"):
            gr.Markdown(
                """
                ### Calibrate the Conformal Engine
                Provide labeled calibration data (query, context, response, label).
                Labels: `0` = correct, `1` = hallucination.
                """
            )
            with gr.Row():
                cal_queries = gr.Textbox(label="Queries (one per line)", lines=6)
                cal_contexts = gr.Textbox(label="Contexts (one per line)", lines=6)
                cal_responses = gr.Textbox(label="Responses (one per line)", lines=6)
                cal_labels = gr.Textbox(
                    label="Labels (one per line: 0=correct, 1=hallucination)",
                    lines=6,
                )
            cal_btn = gr.Button("Calibrate", variant="primary")
            cal_output = gr.Textbox(label="Calibration Results", lines=10, interactive=False)

            cal_btn.click(
                fn=calibrate_engine,
                inputs=[cal_queries, cal_contexts, cal_responses, cal_labels],
                outputs=[cal_output],
            )

        # -------------------------------------------------------------------
        # Tab 3: Monitor
        # -------------------------------------------------------------------
        with gr.Tab("Monitor", id="monitor"):
            gr.Markdown(
                """
                ### Live FDR Monitoring
                Record labeled observations to track the False Discovery Rate in production.
                """
            )
            with gr.Row():
                mon_query = gr.Textbox(label="Query")
                mon_context = gr.Textbox(label="Context")
                mon_response = gr.Textbox(label="Response")
            with gr.Row():
                mon_hallucination = gr.Checkbox(
                    label="Actually a hallucination?",
                    value=False,
                )
                mon_btn = gr.Button("Record Observation", variant="primary")
            mon_output = gr.Textbox(label="Monitor Status", lines=6, interactive=False)

            mon_btn.click(
                fn=record_observation,
                inputs=[mon_query, mon_context, mon_response, mon_hallucination],
                outputs=[mon_output],
            )

    gr.Markdown(
        """
        ---
        **Conformal RAG** — Built with DeBERTa-v3 NLI, LiteLLM, MAPIE, FastAPI.
        FDR guarantee: P(hallucination | not abstained) ≤ α.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
