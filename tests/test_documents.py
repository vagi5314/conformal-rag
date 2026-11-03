"""Tests for document ingestion, answer drafting, and the upload endpoints.

Every test runs without loading a real model: the embedder is replaced by an
all-ones stub and scoring is patched where the engine is reached.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from conformal_rag.api import _MAX_UPLOADS, create_app
from conformal_rag.documents import (
    NOT_ANSWERABLE,
    _CONTROL_CHARS,
    DocumentError,
    chunk_text,
    extract_text,
    extractive_answer,
    generate_answer,
    is_decline,
    is_exact_quote,
    preview,
    split_sentences,
)
from conformal_rag.models import ScoringBreakdown

PASSAGE = (
    "Paracetamol is an analgesic and antipyretic. "
    "The maximum recommended daily dose for adults is 4 grams."
)
DOC_TEXT = (
    "Paracetamol is an analgesic and antipyretic. "
    "The maximum recommended daily dose for adults is 4 grams. "
    "Normal human body temperature is approximately 37 degrees Celsius. "
    "Penicillin was discovered by Alexander Fleming in 1928."
)


def _breakdown(score: float = 0.9) -> ScoringBreakdown:
    return ScoringBreakdown(
        nli_score=score, final_score=score, nli_latency_ms=10.0, used_llm_judge=False
    )


class _FakeEmbedder:
    """Returns deterministic unit-like vectors so no model loads."""

    def encode(self, texts):
        if isinstance(texts, str):
            return np.ones(8, dtype=np.float32)
        return np.ones((len(texts), 8), dtype=np.float32)


class _FakeClient:
    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.reply


# --- extraction ------------------------------------------------------------


class TestExtractText:
    def test_reads_utf8_text_file(self):
        parsed = extract_text("notes.txt", DOC_TEXT.encode("utf-8"))
        assert parsed.kind == "text"
        assert parsed.pages == 0
        assert parsed.text == DOC_TEXT
        assert parsed.filename == "notes.txt"

    def test_reads_file_without_suffix(self):
        parsed = extract_text("README", b"Conformal prediction covers a prediction set.")
        assert parsed.kind == "text"
        assert "Conformal prediction" in parsed.text

    def test_falls_back_to_cp1252(self):
        parsed = extract_text("latin1.txt", b"Caf\xe9 au lait is a drink.")
        assert "Caf\u00e9" in parsed.text

    def test_rejects_unsupported_suffix(self):
        with pytest.raises(DocumentError, match="Unsupported file type"):
            extract_text("table.csv", b"a,b\n1,2\n")

    def test_rejects_empty_file(self):
        with pytest.raises(DocumentError, match="no readable text"):
            extract_text("blank.txt", b"   \n\n  ")

    def test_rejects_pdf_without_text_layer(self):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)

        with pytest.raises(DocumentError, match="No text layer"):
            extract_text("scan.pdf", buffer.getvalue())

    def test_rejects_corrupt_pdf(self):
        with pytest.raises(DocumentError, match="could not be opened"):
            extract_text("broken.pdf", b"%PDF-1.4 not really a pdf")


# --- chunking --------------------------------------------------------------


class TestChunkText:
    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("   \n\n ") == []

    def test_short_text_is_one_chunk(self):
        assert chunk_text("A single short sentence.") == ["A single short sentence."]

    def test_chunks_stay_within_size(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(80))
        chunks = chunk_text(text, size=200, overlap=40)
        assert len(chunks) > 1
        assert all(len(chunk) <= 200 for chunk in chunks)

    def test_prefers_sentence_boundary(self):
        text = (
            "The first sentence is long enough to fill most of the window. "
            "The second sentence should sit in the next chunk."
        )
        chunks = chunk_text(text, size=70, overlap=10)
        assert chunks[0].endswith(".")

    def test_respects_chunk_limit(self):
        text = " ".join("word " * 20 for _ in range(100))
        chunks = chunk_text(text, size=50, overlap=10, limit=3)
        assert len(chunks) == 3

    def test_overlap_repeats_content(self):
        text = " ".join(f"token{i}" for i in range(60))
        chunks = chunk_text(text, size=100, overlap=30)
        assert len(chunks) >= 2
        assert chunks[0][-10:] in chunks[1]

    def test_control_characters_are_stripped(self):
        """PDF math fonts emit U+0000, which made the answer model decline."""
        text = "The risk is bounded\x00by alpha.\x07 Expectation\x1fholds."
        chunks = chunk_text(text)
        assert chunks == ["The risk is bounded by alpha. Expectation holds."]
        assert not any(_CONTROL_CHARS.search(chunk) for chunk in chunks)


# --- sentence split and extractive answer ---------------------------------


class TestExtractiveAnswer:
    def test_splits_sentences(self):
        sentences = split_sentences(
            "One reasonably long sentence sits here. Another sentence follows right after."
        )
        assert sentences == [
            "One reasonably long sentence sits here.",
            "Another sentence follows right after.",
        ]

    def test_picks_sentence_with_most_query_terms(self):
        passages = [PASSAGE]
        answer = extractive_answer(
            "What is the maximum daily dose for adults?", passages
        )
        assert answer == "The maximum recommended daily dose for adults is 4 grams."

    def test_falls_back_to_first_sentence(self):
        answer = extractive_answer("unrelated question", [PASSAGE])
        assert answer == "Paracetamol is an analgesic and antipyretic."

    def test_empty_passages_return_empty_string(self):
        assert extractive_answer("anything", []) == ""

    def test_preview_truncates(self):
        assert len(preview("word " * 200, limit=50)) <= 50
        assert preview("short text") == "short text"


# --- answer drafting -------------------------------------------------------


class TestGenerateAnswer:
    def test_no_passages_means_no_answer(self):
        assert generate_answer("q", []) == ("", "none")

    def test_no_client_uses_extractive_mode(self):
        answer, mode = generate_answer(
            "What is the maximum daily dose for adults?", [PASSAGE]
        )
        assert mode == "extractive"
        assert "4 grams" in answer

    def test_client_answer_is_tagged_llm(self):
        client = _FakeClient(reply="The adult maximum is 4 grams per day.")
        answer, mode = generate_answer("dose?", [PASSAGE], client=client)
        assert mode == "llm"
        assert answer == "The adult maximum is 4 grams per day."
        assert "dose?" in client.prompts[0]

    def test_client_can_decline(self):
        client = _FakeClient(reply=NOT_ANSWERABLE)
        assert generate_answer("dose?", [PASSAGE], client=client) == ("", "none")

    @pytest.mark.parametrize(
        "reply",
        [
            "NOT_IN",  # provider truncated the sentinel
            "not_in_pass",
            "  NOT_IN_PASSAGES  ",
            '"NOT_IN_PASSAGES"',
            "Not in the passages.",
        ],
    )
    def test_truncated_or_rephrased_decline_is_not_an_answer(self, reply):
        """A cut-short sentinel used to be approved as a real answer."""
        client = _FakeClient(reply=reply)
        assert generate_answer("dose?", [PASSAGE], client=client) == ("", "none")

    def test_ordinary_answers_are_not_treated_as_declines(self):
        for reply in (
            "The adult maximum is 4 grams per day.",
            "Nothing in the passage mentions children.",
        ):
            assert not is_decline(reply)

    def test_prompt_carries_no_control_characters(self):
        """A U+0000 anywhere in the prompt made the model answer NOT_IN_PASSAGES.

        Reproduces the PDF path: extraction hands `chunk_text` raw text carrying
        U+0000 from a math font, and whatever reaches the model must be clean.
        """
        client = _FakeClient(reply="The adult maximum is 4 grams per day.")
        passages = chunk_text(PASSAGE.replace("4 grams", "4\x00grams"))
        answer, mode = generate_answer("dose?", passages, client=client)
        assert mode == "llm"
        assert answer == "The adult maximum is 4 grams per day."
        assert "\x00" not in client.prompts[0]

    def test_client_failure_degrades_to_extractive(self):
        client = _FakeClient(error=RuntimeError("provider down"))
        answer, mode = generate_answer(
            "What is the maximum daily dose for adults?", [PASSAGE], client=client
        )
        assert mode == "extractive"
        assert "4 grams" in answer


class TestIsExactQuote:
    """The containment test behind the "Grounded" verdict in the UI."""

    def test_verbatim_sentence_is_grounded(self):
        assert is_exact_quote(
            "The maximum recommended daily dose for adults is 4 grams.", PASSAGE
        )
        assert is_exact_quote("Paracetamol is an analgesic and antipyretic.", PASSAGE)

    def test_whitespace_and_case_differences_still_grounded(self):
        assert is_exact_quote("paracetamol   is an\nanalgesic", PASSAGE)

    def test_paraphrase_is_not_grounded(self):
        assert not is_exact_quote("Adults may take up to four grams a day.", PASSAGE)

    def test_empty_answer_is_not_grounded(self):
        assert not is_exact_quote("", PASSAGE)
        assert not is_exact_quote("   ", PASSAGE)


# --- endpoints -------------------------------------------------------------


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def _upload(client, text: str = DOC_TEXT, name: str = "notes.txt"):
    with patch("conformal_rag.retrieval.get_embedder", return_value=_FakeEmbedder()):
        return await client.post(
            "/documents/upload",
            files={"file": (name, text.encode("utf-8"), "text/plain")},
        )


class TestUploadEndpoint:
    @pytest.mark.asyncio
    async def test_upload_returns_metadata_and_doc_id(self, app, client):
        response = await _upload(client)
        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "notes.txt"
        assert body["kind"] == "text"
        assert body["pages"] == 0
        assert body["characters"] == len(DOC_TEXT)
        assert body["passages"] >= 1
        assert body["doc_id"] in app.state.uploads
        assert app.state.uploads[body["doc_id"]].passages

    @pytest.mark.asyncio
    async def test_upload_rejects_unsupported_type(self, client):
        response = await _upload(client, text="a,b\n1,2\n", name="table.csv")
        assert response.status_code == 422
        assert "Unsupported file type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_rejects_empty_file(self, client):
        response = await _upload(client, text="   ", name="blank.txt")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_uploads_are_capped_oldest_first(self, app, client):
        ids = []
        for i in range(_MAX_UPLOADS + 1):
            response = await _upload(client, text=f"Fact number {i} sits here.", name=f"f{i}.txt")
            ids.append(response.json()["doc_id"])
        assert len(app.state.uploads) == _MAX_UPLOADS
        assert ids[0] not in app.state.uploads
        assert ids[-1] in app.state.uploads


class TestAskEndpoint:
    @pytest.mark.asyncio
    async def test_ask_answers_and_checks_the_answer(self, app, client):
        doc_id = (await _upload(client)).json()["doc_id"]
        uploaded = app.state.uploads[doc_id]
        with patch.object(uploaded.retriever, "retrieve", return_value=[PASSAGE]):
            with patch.object(
                app.state.engine.scoring, "score", return_value=_breakdown(0.9)
            ):
                response = await client.post(
                    "/ask",
                    json={
                        "doc_id": doc_id,
                        "query": "What is the maximum daily dose for adults?",
                    },
                )
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "extractive"
        assert "4 grams" in body["answer"]
        assert body["grounding"] == "exact"
        assert body["passages"] == [PASSAGE]
        assert body["grounded_in"] == PASSAGE
        assert body["verdict"]["abstained"] is False
        assert body["verdict"]["risk_score"] == 0.9
        assert "outside" in body["note"]

    @pytest.mark.asyncio
    async def test_ask_unknown_document_is_404(self, client):
        response = await client.post(
            "/ask", json={"doc_id": "missing", "query": "anything"}
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_ask_without_retrieved_passages_returns_no_answer(self, app, client):
        doc_id = (await _upload(client)).json()["doc_id"]
        uploaded = app.state.uploads[doc_id]
        with patch.object(uploaded.retriever, "retrieve", return_value=[]):
            response = await client.post(
                "/ask", json={"doc_id": doc_id, "query": "anything"}
            )
        body = response.json()
        assert body["mode"] == "none"
        assert body["answer"] == ""
        assert body["verdict"] is None

    @pytest.mark.asyncio
    async def test_ask_routes_through_a_calibrated_domain(self, app, client):
        await client.post("/domains", json={"name": "medicine", "alpha": 0.05})
        dm = app.state.domain_manager
        engine = dm.domains["medicine"]
        data = [
            {"query": "q1", "context": "c1", "response": "r1", "label": 0},
            {"query": "q2", "context": "c2", "response": "r2", "label": 1},
        ]
        with patch.object(engine.scoring, "score", return_value=_breakdown(0.8)):
            await client.post("/domains/medicine/calibrate", json={"data": data})

        doc_id = (await _upload(client)).json()["doc_id"]
        uploaded = app.state.uploads[doc_id]
        with patch.object(uploaded.retriever, "retrieve", return_value=[PASSAGE]):
            with patch.object(engine.scoring, "score", return_value=_breakdown(0.9)):
                response = await client.post(
                    "/ask",
                    json={
                        "doc_id": doc_id,
                        "query": "What is the maximum daily dose for adults?",
                        "domain": "medicine",
                    },
                )
        body = response.json()
        assert body["domain"] == "medicine"
        assert body["verdict"]["domain"] == "medicine"
        assert dm.monitors["medicine"].get_metrics()["total_predictions"] == 1

    @pytest.mark.asyncio
    async def test_ask_defaults_uploads_to_general_not_auto_routing(self, app, client):
        """An upload is off-distribution, so it is checked against general.

        Centroid routing on a brand-new document would stamp it with a topic
        from the demo corpus (a statistics paper once routed to finance). Only
        an explicit domain opts an upload into a narrower calibration.
        """
        await client.post("/domains", json={"name": "medicine", "alpha": 0.05})
        await client.post("/domains", json={"name": "general", "alpha": 0.15})
        dm = app.state.domain_manager
        data = [
            {"query": "q", "context": "c", "response": "r", "label": i % 2}
            for i in range(20)
        ]
        general = dm.domains["general"]
        with patch.object(general.scoring, "score", return_value=_breakdown(0.8)):
            await client.post("/domains/general/calibrate", json={"data": data})

        doc_id = (await _upload(client)).json()["doc_id"]
        uploaded = app.state.uploads[doc_id]
        with patch.object(uploaded.retriever, "retrieve", return_value=[PASSAGE]):
            with patch.object(dm, "_detect_domain", return_value="medicine"):
                with patch.object(general.scoring, "score", return_value=_breakdown(0.9)):
                    response = await client.post(
                        "/ask",
                        json={
                            "doc_id": doc_id,
                            "query": "What is the maximum daily dose for adults?",
                        },
                    )
        body = response.json()
        assert body["domain"] == "general"