import pytest
from conformal_rag.retrieval import HybridRetriever, reciprocal_rank_fusion
from conformal_rag.models import Document


class TestRRF:
    def test_rrf_merges_rankings(self):
        ranking1 = ["doc_a", "doc_b", "doc_c"]
        ranking2 = ["doc_b", "doc_c", "doc_d"]
        result = reciprocal_rank_fusion([ranking1, ranking2])
        assert result[0] == "doc_b"

    def test_rrf_empty_rankings(self):
        result = reciprocal_rank_fusion([[], []])
        assert result == []


class TestHybridRetriever:
    def setup_method(self):
        self.retriever = HybridRetriever(top_k=5, final_k=3)

    def test_index_adds_documents(self):
        docs = [Document(text="Hello world"), Document(text="Foo bar")]
        self.retriever.index(docs)
        assert len(self.retriever._documents) == 2

    def test_retrieve_returns_results(self):
        docs = [
            Document(text="Paris is the capital of France."),
            Document(text="Python is a programming language."),
            Document(text="The sun is a star."),
        ]
        self.retriever.index(docs)
        results = self.retriever.retrieve("capital of France")
        assert len(results) > 0
        assert len(results) <= 3

    def test_retrieve_relevant_document(self):
        docs = [
            Document(text="Paris is the capital of France."),
            Document(text="Python is a programming language."),
        ]
        self.retriever.index(docs)
        results = self.retriever.retrieve("capital of France")
        assert any("Paris" in r for r in results)

    def test_dense_search_returns_list(self):
        results = self.retriever._dense_search("test")
        assert isinstance(results, list)

    def test_sparse_search_returns_list(self):
        results = self.retriever._sparse_search("test")
        assert isinstance(results, list)
