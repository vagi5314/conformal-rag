"""Live-server tests for the Qdrant retrieval path.

Require a reachable Qdrant server and are skipped unless one responds at
CONFORMAL_RAG_QDRANT_TEST_URL (default http://localhost:6333). CI provides
one via a service container; locally use `docker compose up -d qdrant`.

These tests load the real e5 embedding model and use a unique collection per
run, deleted afterwards.
"""

import os
import socket
import uuid
from urllib.parse import urlparse

import pytest

from conformal_rag.models import Document
from conformal_rag.retrieval import HybridRetriever

_QDRANT_URL = os.environ.get("CONFORMAL_RAG_QDRANT_TEST_URL", "http://localhost:6333")


def _qdrant_reachable(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        with socket.create_connection((host, port), timeout=1):
            return True
    except (OSError, ValueError):
        return False


_QDRANT_UP = _qdrant_reachable(_QDRANT_URL)
needs_qdrant = pytest.mark.skipif(not _QDRANT_UP, reason="Qdrant not reachable")

DOCS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "Python is a high-level programming language created by Guido van Rossum.",
    "Conformal prediction provides distribution-free uncertainty quantification.",
    "The mitochondria is the powerhouse of the cell.",
]


@pytest.fixture(scope="module")
def collection():
    return f"test_docs_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def retriever(collection):
    r = HybridRetriever(qdrant_url=_QDRANT_URL, collection=collection, top_k=4, final_k=3)
    yield r
    try:
        if r._client and r._client.collection_exists(collection):
            r._client.delete_collection(collection)
    except Exception:
        pass


@needs_qdrant
class TestQdrantLive:
    def test_index_creates_collection_with_named_vectors(self, retriever, collection):
        retriever.index([Document(text=d) for d in DOCS])
        info = retriever._client.get_collection(collection)
        assert "dense" in info.config.params.vectors
        assert "sparse" in info.config.params.sparse_vectors
        assert info.points_count == len(DOCS)

    def test_retrieve_fuses_dense_and_sparse(self, retriever):
        retriever.index([Document(text=d) for d in DOCS])
        results = retriever.retrieve("capital of France")
        assert len(results) > 0
        assert any("Paris" in r for r in results)

    def test_sparse_search_matches_shared_tokens(self, retriever):
        retriever.index([Document(text=d) for d in DOCS])
        results = retriever._qdrant_sparse_search("programming language Python")
        assert any("Python" in r for r in results)

    def test_dense_search_returns_semantically_close_doc(self, retriever):
        retriever.index([Document(text=d) for d in DOCS])
        results = retriever._qdrant_dense_search("uncertainty quantification")
        assert any("Conformal" in r for r in results)

    def test_reindex_leaves_no_stale_points(self, retriever, collection):
        retriever.index([Document(text=d) for d in DOCS])
        retriever.index([Document(text="A brand new document about whales.")])
        assert retriever._client.get_collection(collection).points_count == len(DOCS) + 1
        results = retriever.retrieve("whales")
        assert any("whales" in r for r in results)

    def test_load_state_repopulates_collection(self, retriever, collection, tmp_path):
        retriever.index([Document(text=d) for d in DOCS])
        state_file = tmp_path / "index.json"
        retriever.save_state(state_file)

        # Wipe the collection to simulate a Qdrant restart with empty storage.
        retriever._client.delete_collection(collection)

        fresh = HybridRetriever(qdrant_url=_QDRANT_URL, collection=collection)
        assert fresh.load_state(state_file) is True
        assert fresh._client.get_collection(collection).points_count == len(DOCS)
        results = fresh.retrieve("capital of France")
        assert any("Paris" in r for r in results)
