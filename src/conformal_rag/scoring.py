"""Hybrid scoring pipeline: NLI ensemble + LLM judge."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from conformal_rag.models import ScoringBreakdown
from conformal_rag.llm import LLMClient, LLMConfig
from conformal_rag.guards import sanitize_query

logger = logging.getLogger(__name__)


_nli_cache: dict = {}

# Label names that mean "the hypothesis follows from the premise".
_ENTAILMENT_LABELS = {"entailment", "entail", "entails", "true", "yes"}


def _entailment_index(model) -> int:
    """Find the entailment class index from the model's own label map.

    NLI models disagree on label ordering (e.g. the MoritzLaurer DeBERTa puts
    entailment at index 0, cross-encoder NLI models put contradiction there),
    so the index must be read from id2label, never hardcoded.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).strip().lower() in _ENTAILMENT_LABELS:
            return int(idx)
    raise ValueError(
        f"Could not locate an entailment label in id2label={id2label!r}; "
        "refusing to guess a score direction."
    )


@dataclass
class NLIScorer:
    primary_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    fallback_model: str = "cross-encoder/nli-deberta-v3-small"
    confidence_threshold: float = 0.7
    # float16 halves the model's memory footprint on memory-constrained CPU
    # boxes; scores are cast back to float32 before softmax for stability.
    use_float16: bool = False
    _tokenizer: Optional[object] = field(default=None, repr=False)
    _model: Optional[object] = field(default=None, repr=False)
    _entailment_idx: Optional[int] = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def _load_model(self):
        if self._loaded:
            return
        cache_key = f"{self.primary_model}::fp16={self.use_float16}"
        if cache_key in _nli_cache:
            cached = _nli_cache[cache_key]
            self._tokenizer = cached["tokenizer"]
            self._model = cached["model"]
            self._entailment_idx = cached["entailment_idx"]
            self._loaded = True
            return
        load_kwargs = {"torch_dtype": torch.float16} if self.use_float16 else {}
        try:
            model_name = self.primary_model
            self._tokenizer = AutoTokenizer.from_pretrained(self.primary_model)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.primary_model, **load_kwargs
            )
        except Exception as e:
            logger.warning("Primary NLI model %s failed to load (%s); falling back to %s",
                           self.primary_model, e, self.fallback_model)
            model_name = self.fallback_model
            self._tokenizer = AutoTokenizer.from_pretrained(self.fallback_model)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.fallback_model, **load_kwargs
            )
        self._model.eval()
        self._entailment_idx = _entailment_index(self._model)
        logger.info("NLI model loaded: %s (entailment index = %d, fp16=%s)",
                    model_name, self._entailment_idx, self.use_float16)
        _nli_cache[cache_key] = {
            "tokenizer": self._tokenizer,
            "model": self._model,
            "entailment_idx": self._entailment_idx,
        }
        self._loaded = True

    def score(self, query: str, context: str, response: str) -> float:
        self._load_model()
        return self._predict(query, context, response)

    def score_batch(
        self, pairs: list[tuple[str, str]], batch_size: int = 8
    ) -> list[float]:
        """Batched faithfulness scoring for (context, response) pairs.

        Semantics match `score()` exactly: each response is decomposed into
        claims, every (context, claim) pair is scored, and the per-response
        result is the mean claim score (0.5 when decomposition is empty).
        Batching the model calls is 5-10x faster than one forward pass per
        claim on CPU.
        """
        self._load_model()
        decomposed = [self._decompose(response) for _, response in pairs]

        flat_ctx: list[str] = []
        flat_claim: list[str] = []
        owner: list[int] = []
        for i, ((ctx, _), claims) in enumerate(zip(pairs, decomposed)):
            for claim in claims:
                flat_ctx.append(ctx)
                flat_claim.append(claim)
                owner.append(i)

        totals = [0.0] * len(pairs)
        counts = [0] * len(pairs)
        for start in range(0, len(flat_ctx), batch_size):
            batch_ctx = flat_ctx[start:start + batch_size]
            batch_claim = flat_claim[start:start + batch_size]
            inputs = self._tokenizer(
                batch_ctx, batch_claim,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, self._entailment_idx]
            for j, p in enumerate(probs.tolist()):
                i = owner[start + j]
                totals[i] += p
                counts[i] += 1

        return [
            totals[i] / counts[i] if counts[i] else 0.5
            for i in range(len(pairs))
        ]

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
            logits = self._model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1)
        return probs[0][self._entailment_idx].item()

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
        """Build the judge prompt.

        User-supplied text is sanitized first so an adversarial response
        cannot inject instructions like "return only 1.0" into the judge.
        """
        return (
            "You are a factuality checker. Given a query, retrieved context, "
            "and a response, determine if the response is faithful to the context.\n\n"
            f"Query: {sanitize_query(query)}\n"
            f"Context: {sanitize_query(context)}\n"
            f"Response: {sanitize_query(response)}\n\n"
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
        # Accept "0.85", "85%", ".85", and bare "0"/"1" — but never bare
        # multi-digit integers like "85", which would clamp to 1.0 and
        # silently approve anything.
        match = re.search(r"(\d*\.\d+|\d+%|\b[01]\b)", result)
        if match:
            token = match.group(1)
            if token.endswith("%"):
                value = float(token[:-1]) / 100.0
            else:
                value = float(token)
            return max(0.0, min(1.0, value))
        logger.warning("LLM judge returned unparseable output: %r", result)
        return 0.5


class ScoringPipeline:
    """Hybrid scoring: NLI fast path + LLM judge for uncertain cases."""

    # Named NLI model presets. "large" is the production default; "small"
    # is ~10x faster on CPU and suitable for benchmarking on modest hardware.
    NLI_PRESETS = {
        "large": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "small": "cross-encoder/nli-deberta-v3-small",
    }

    def __init__(
        self,
        use_llm_judge: bool = True,
        nli_confidence_threshold: float = 0.7,
        weights: tuple[float, float] = (0.6, 0.4),
        nli_model: str = "large",
        use_float16: bool = False,
    ):
        model_name = self.NLI_PRESETS.get(nli_model, nli_model)
        self.nli = NLIScorer(primary_model=model_name, use_float16=use_float16)
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

        # Confidence is a pure function of the score; recomputing it via
        # get_confidence() would run the whole NLI model a second time.
        nli_confidence = abs(nli_score - 0.5) * 2

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

    def score_batch(
        self,
        queries: list[str],
        contexts: list[str],
        responses: list[str],
        batch_size: int = 8,
    ) -> list[ScoringBreakdown]:
        """Batched scoring for calibration-sized workloads.

        With the LLM judge disabled this batches all NLI claims into padded
        forward passes (5-10x faster than per-sample on CPU). With the judge
        enabled it falls back to per-sample `score()`, since judge calls are
        per-sample network requests anyway.
        """
        if self.use_llm_judge:
            return [self.score(q, c, r) for q, c, r in zip(queries, contexts, responses)]

        start = time.time()
        nli_scores = self.nli.score_batch(
            list(zip(contexts, responses)), batch_size=batch_size
        )
        per_sample_ms = (time.time() - start) * 1000 / max(len(responses), 1)
        return [
            ScoringBreakdown(
                nli_score=s,
                llm_judge_score=None,
                final_score=s,
                nli_latency_ms=per_sample_ms,
                llm_latency_ms=None,
                used_llm_judge=False,
            )
            for s in nli_scores
        ]
