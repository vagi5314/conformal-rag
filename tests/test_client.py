"""Tests for Python client SDK."""

import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.client import ConformalRAG


class TestConformalRAG:
    def test_initializes(self):
        rag = ConformalRAG(alpha=0.05)
        assert rag.alpha == 0.05

    def test_registers_domain(self):
        rag = ConformalRAG()
        rag.register_domain("finance", alpha=0.01)
        assert "finance" in rag.domains

    def test_query_returns_result(self):
        rag = ConformalRAG()
        with patch.object(rag.engine, "predict") as mock_pred:
            mock_pred.return_value = MagicMock(domain="default")
            result = rag.query("What is Python?")
            assert result is not None
