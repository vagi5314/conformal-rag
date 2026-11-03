# Sample PDFs — how to test document Q&A

Two real PDFs for trying the "Ask a document" flow. Open the console at
<http://127.0.0.1:8000>, go to **Check an Answer → Upload a document**, drop one
of these files in, then type a question.

| File | What it is | Pages | Passages |
| --- | --- | --- | --- |
| `conformal-risk-control.pdf` | Angelopoulos, Bates, Fisch, Lei & Schuster, *Conformal Risk Control* (arXiv:2208.02814) | 19 | 94 |
| `nist-cloud-computing.pdf` | NIST SP 800-145, *The NIST Definition of Cloud Computing* | 7 | 22 |

## Questions that work

Every question below was run against the live server and came back with an
approved verdict and a correct answer.

**Conformal Risk Control**

- What is conformal risk control?
- Who are the authors of the Conformal Risk Control paper?
- What does conformal risk control guarantee about the expected loss?
- How is it different from split conformal prediction?

**NIST SP 800-145**

- How does NIST define cloud computing?
- What are the three cloud service models?
- What is measured service?
- What are the essential characteristics of cloud computing?

Tip: questions phrased close to the document's own wording work best. A bare
"Who are the authors?" without naming the paper sometimes declines, because the
retrieved passages do not make it obvious which document is meant.

## Questions that should be declined

This is the point of the system — it abstains instead of inventing an answer.
Each of these returns an empty answer, no verdict, a note, and the closest
passage it did find:

| PDF | Off-topic question |
| --- | --- |
| `nist-cloud-computing.pdf` | What is the boiling point of mercury? |
| `conformal-risk-control.pdf` | Who won the 2018 FIFA World Cup? |

## Asking from the command line

```bash
# 1. Upload — the reply carries the doc_id
DOC_ID=$(curl -s -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@sample_docs/nist-cloud-computing.pdf" | python -c "import json,sys;print(json.load(sys.stdin)['doc_id'])")

# 2. Ask
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"doc_id\": \"$DOC_ID\", \"query\": \"What are the three cloud service models?\"}"
```

The response contains `answer`, `mode`, `grounding`, `passages`, `verdict`
(`risk_score`, `threshold`, `abstained`) and a `note`. Uploaded documents live
in memory, so a server restart drops them — re-upload after restarting.
