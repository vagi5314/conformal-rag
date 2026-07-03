"""Hybrid scoring pipeline: NLI ensemble + LLM judge."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from conformal_rag.models import ScoringBreakdown
from conformal_rag.llm import LLMClient, LLMConfig


_nli_cache: dict = {}


@dataclass
class NLIScorer:
    primary_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    fallback_model: str = "cross-encoder/nli-deberta-v3-small"
    confidence_threshold: float = 0.7
    _tokenizer: Optional[object] = field(default=None, repr=False)
    _model: Optional[object] = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def _load_model(self):
        if self._loaded:
            return
        if "tokenizer" in _nli_cache:
            self._tokenizer = _nli_cache["tokenizer"]
            self._model = _nli_cache["model"]
            self._loaded = True
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.primary_model)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.primary_model)
            self._model.eval()
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(self.fallback_model)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.fallback_model)
            self._model.eval()
        _nli_cache["tokenizer"] = self._tokenizer
        _nli_cache["model"] = self._model
        self._loaded = True

    def score(self, query: str, context: str, response: str) -> float:
        self._load_model()
        return self._predict(query, context, response)

    def _predict(self, query: str, context: str, response: str) -> float:
        claims = self._decompose(response)
        if not claims:
            return 0.5
        scores = [self._score_claim(context, claim) for claim in claims]
        return sum(scores) / len(scores)

    def _decompose(self, text: str) -> list:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _score_claim(self, context: str, claim: str) -> float:
        self._load_model()
        inputs = self._tokenizer(
            context, claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0][0].item()

    def get_confidence(self, query: str, context: str, response: str) -> float:
        return abs(self.score(query, context, response) - 0.5) * 2


@dataclass
class LLMJudge:
    """LLM-based faithfulness judge."""

    config: Optional[LLMConfig] = None
    client: Optional[LLMClient] = None

    def __post_init__(self):
        if self.config is None:
            self.config = LLMConfig()
        if self.client is None:
            self.client = LLMClient(self.config)

    def _build_prompt(self, query: str, context: str, response: str) -> str:
        """Build the judge prompt."""
        return (
            "You are a factuality checker. Given a query, retrieved context, "
            "and a response, determine if the response is faithful to the context.\n\n"
            f"Query: {query}\n"
            f"Context: {context}\n"
            f"Response: {response}\n\n"
            "Score the faithfulness from 0.0 (complete hallucination) to 1.0 (fully faithful). "
            "Return ONLY the numeric score."
        )

    def score(self, query: str, context: str, response: str) -> float:
        """Score faithfulness using LLM judge.

        Returns:
            Float 0-1, higher = more faithful.
        """
        prompt = self._build_prompt(query, context, response)
        result = self.client.complete(prompt)
        try:
            return float(result.strip())
        except ValueError:
            return 0.5


class ScoringPipeline:
    """Hybrid scoring: NLI fast path + LLM judge for uncertain cases."""

    def __init__(
        self,
        use_llm_judge: bool = True,
        nli_confidence_threshold: float = 0.7,
        weights: tuple[float, float] = (0.6, 0.4),
    ):
        self.nli = NLIScorer()
        self.llm_judge = LLMJudge() if use_llm_judge else None
        self.use_llm_judge = use_llm_judge
        self.nli_confidence_threshold = nli_confidence_threshold
        self.weights = weights

    def score(self, query: str, context: str, response: str) -> ScoringBreakdown:
        """Score a query-context-response triple.

        Returns:
            ScoringBreakdown with all scores and metadata.
        """
        # NLI fast path
        nli_start = time.time()
        nli_score = self.nli.score(query, context, response)
        nli_latency = (time.time() - nli_start) * 1000

        # Check NLI confidence
        nli_confidence = self.nli.get_confidence(query, context, response)

        # If NLI is confident and LLM judge is disabled, use NLI only
        if not self.use_llm_judge or nli_confidence > self.nli_confidence_threshold:
            return ScoringBreakdown(
                nli_score=nli_score,
                llm_judge_score=None,
                final_score=nli_score,
                nli_latency_ms=nli_latency,
                llm_latency_ms=None,
                used_llm_judge=False,
            )

        # LLM judge path (uncertain cases)
        llm_start = time.time()
        llm_score = self.llm_judge.score(query, context, response)
        llm_latency = (time.time() - llm_start) * 1000

        # Weighted ensemble
        final_score = (
            self.weights[0] * llm_score + self.weights[1] * nli_score
        )

        return ScoringBreakdown(
            nli_score=nli_score,
            llm_judge_score=llm_score,
            final_score=final_score,
            nli_latency_ms=nli_latency,
            llm_latency_ms=llm_latency,
            used_llm_judge=True,
        )
