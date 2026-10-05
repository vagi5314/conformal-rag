"""Interactive CLI demonstration of Conformal RAG.

Run directly with:
    .venv\\Scripts\\python.exe quickstart.py
"""
import sys
import time

# Ensure src is in python path
sys.path.insert(0, "src")

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.scoring import ScoringPipeline
from conformal_rag.retrieval import HybridRetriever
from conformal_rag.models import Document

def main():
    print("=" * 70)
    print("      CONFORMAL RAG PLATFORM — UNDERLYING SYSTEM CLI TEST")
    print("=" * 70)

    # 1. RETRIEVAL STEP: Index Knowledge Documents
    print("\n[Step 1] Indexing Knowledge Base into Hybrid Retriever (BM25 + Dense)...")
    docs = [
        Document(text="Albert Einstein published the theory of Special Relativity in 1905 and General Relativity in 1915. He won the Nobel Prize in Physics in 1921 for the photoelectric effect."),
        Document(text="The French Revolution began in 1789 with the Storming of the Bastille and ended in the late 1790s with the ascent of Napoleon Bonaparte."),
        Document(text="Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands as a successor to the ABC language.")
    ]
    retriever = HybridRetriever()
    retriever.index(docs)
    print(f"-> Successfully indexed {len(docs)} documents.")

    query = "Why did Einstein win the Nobel Prize?"
    print(f"\nQuerying retriever for: '{query}'")
    retrieved_chunks = retriever.retrieve(query)
    retrieved_context = retrieved_chunks[0] if retrieved_chunks else ""
    print(f"-> Top Retrieved Context: \"{retrieved_context}\"")

    # 2. CALIBRATION STEP: Calibrate Conformal Guarantees
    print("\n" + "-" * 70)
    print("[Step 2] Calibrating Conformal Risk Engine (alpha = 0.20)...")
    
    # We use NLI scoring pipeline with lightweight 'small' preset for fast, reliable local CPU execution
    scoring = ScoringPipeline(use_llm_judge=False, nli_model="small")
    engine = ConformalRAGEngine(alpha=0.20, scoring=scoring)

    # 6 labeled samples: label 0 = faithful (correct), label 1 = hallucination (wrong)
    calibration_data = [
        # Faithful samples (label = 0)
        ("What is Python?", "Python is a programming language.", "Python is a coding language.", 0),
        ("When was the Bastille stormed?", "The French Revolution began in 1789 with the storming of the Bastille.", "It occurred in 1789.", 0),
        ("Who created Python?", "Python was conceived by Guido van Rossum.", "Guido van Rossum invented Python.", 0),
        # Hallucinated samples (label = 1)
        ("What is Python?", "Python is a programming language.", "Python is an operating system by Microsoft.", 1),
        ("When was the Bastille stormed?", "The French Revolution began in 1789.", "The Bastille was stormed in 1945 during WWII.", 1),
        ("Who created Python?", "Python was conceived by Guido van Rossum.", "Elon Musk developed Python in 2021.", 1),
    ]

    stats = engine.calibrate(
        queries=[s[0] for s in calibration_data],
        contexts=[s[1] for s in calibration_data],
        responses=[s[2] for s in calibration_data],
        labels=[s[3] for s in calibration_data]
    )

    print("-> Calibration complete!")
    print(f"   * Number of samples: {stats.n_samples}")
    print(f"   * Calibrated Threshold: {stats.threshold:.4f}")
    print(f"   * In-sample Coverage: {stats.coverage * 100:.1f}%")
    print(f"   * Target Error Rate (alpha): {engine.alpha * 100:.1f}%")

    # 3. VERIFICATION STEP: Test Candidate Answers
    print("\n" + "-" * 70)
    print("[Step 3] Testing Candidate LLM Responses against Retrieved Context...")

    candidate_answers = [
        {
            "type": "FAITHFUL ANSWER",
            "text": "Einstein won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect."
        },
        {
            "type": "HALLUCINATED ANSWER",
            "text": "Einstein won the Nobel Prize for discovering the theory of General Relativity in 1935."
        }
    ]

    for candidate in candidate_answers:
        print(f"\n--- Testing: {candidate['type']} ---")
        print(f"Answer text: \"{candidate['text']}\"")
        
        t0 = time.time()
        result = engine.predict(
            query=query,
            context=retrieved_context,
            response=candidate["text"]
        )
        dt = (time.time() - t0) * 1000

        print(f"  * Faithfulness / Risk Score: {result.risk_score:.4f}  (Threshold: {engine._threshold:.4f})")
        print(f"  * Conformal CI (Lower Bound): {result.confidence_interval[0]:.4f}")
        
        if result.abstained:
            print("  * Decision: [REJECTED / ABSTAINED] (Score <= Threshold -> Risk too high!)")
        else:
            print("  * Decision: [APPROVED / CERTIFIED] (Score > Threshold -> Statistically Safe)")
        print(f"  * Latency: {dt:.1f}ms")

    print("\n" + "=" * 70)
    print("               CLI TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException as e:
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
