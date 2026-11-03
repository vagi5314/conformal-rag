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

# Split a passage into sentences, keeping the terminator. The cross-encoder
# NLI model is punctuation sensitive: the same claim scores 0.93 as a sentence
# and 0.02 as a bare fragment, so claims and premises both keep their stops.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# A claim is checked against each sentence and against overlapping windows of
# this many sentences, keeping the best match. The small cross-encoder degrades
# badly with premise length: the same claim scores 0.99 against a three-sentence
# window, 0.005 against the whole document, and an unrelated claim reaches 0.02
# against the whole document (a false approval). Single sentences catch a
# verbatim restatement; windows supply the context that resolves pronouns.
_PREMISE_WINDOW = 3
_MAX_PREMISES = 48


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

        flat: list[tuple[str, str]] = []
        claim_of: list[tuple[int, int]] = []
        for i, ((ctx, _), claims) in enumerate(zip(pairs, decomposed)):
            premises = self._premises(ctx)
            for ci, claim in enumerate(claims):
                for premise in premises:
                    flat.append((premise, claim))
                    claim_of.append((i, ci))

        # Per (response, claim) keep the best window, then average over the
        # response's claims — the same result `score()` computes one at a time.
        best_claim: dict[tuple[int, int], float] = {}
        for key, value in zip(claim_of, self._score_pairs(flat, batch_size)):
            if value > best_claim.get(key, -1.0):
                best_claim[key] = value

        totals = [0.0] * len(pairs)
        counts = [0] * len(pairs)
        for (i, _ci), value in best_claim.items():
            totals[i] += value
            counts[i] += 1

        return [
            totals[i] / counts[i] if counts[i] else 0.5
            for i in range(len(pairs))
        ]

    def _predict(self, query: str, context: str, response: str) -> float:
        claims = self._decompose(response)
        if not claims:
            return 0.5
        premises = self._premises(context)
        scores = self._score_pairs([(p, c) for c in claims for p in premises])
        width = len(premises)
        per_claim = [
            max(scores[i * width:(i + 1) * width]) for i in range(len(claims))
        ]
        return sum(per_claim) / len(per_claim)

    def _decompose(self, text: str) -> list:
        claims = []
        for part in _SENTENCE_SPLIT.split(text):
            s = part.strip()
            if len(s.rstrip(".!? ")) <= 10:
                continue
            claims.append(s if s[-1] in ".!?" else s + ".")
        return claims

    def _premises(self, context: str) -> list:
        """Premises a claim is checked against: sentences and 3-sentence windows.

        The small NLI model is only reliable on a local premise. Scored against
        a whole multi-paragraph document, a faithful claim drops to ~0.005 while
        an unrelated claim can climb to ~0.02: wrong in both directions. A
        single sentence catches a verbatim restatement; an overlapping window
        keeps the surrounding context that resolves pronouns.
        """
        text = context.strip()
        sentences = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
        if not sentences:
            return [" "]
        premises = list(sentences)
        if len(sentences) > _PREMISE_WINDOW:
            premises.extend(
                " ".join(sentences[i:i + _PREMISE_WINDOW])
                for i in range(len(sentences) - _PREMISE_WINDOW + 1)
            )
        elif text not in premises:
            premises.append(text)
        seen: set[str] = set()
        unique = [p for p in premises if not (p in seen or seen.add(p))]
        return unique[:_MAX_PREMISES]

    def _score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        """Batched entailment probability for (premise, claim) pairs."""
        if not pairs:
            return []
        self._load_model()
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = self._tokenizer(
                [p for p, _ in batch],
                [c for _, c in batch],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, self._entailment_idx]
            scores.extend(probs.tolist())
        return scores

    def _score_claim(self, context: str, claim: str) -> float:
        """Best window score for one claim against a context."""
        premises = self._premises(context)
        return max(self._score_pairs([(p, claim) for p in premises]))

    def _score_pair(self, premise: str, claim: str) -> float:
        return self._score_pairs([(premise, claim)])[0]

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
        # Reasoning-style models emit thinking tokens before the answer, so a
        # 10-token budget can come back empty. 48 is still a fraction of a cent.
        result = self.client.complete(prompt, max_tokens=48)
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
        nli_model: str = "small",
        use_float16: bool = False,
        fallback_chain: Optional[list[str]] = None,
    ):
        model_name = self.NLI_PRESETS.get(nli_model, nli_model)
        self.nli = NLIScorer(primary_model=model_name, use_float16=use_float16)
        judge_config = (
            LLMConfig(fallback_chain=list(fallback_chain))
            if fallback_chain
            else LLMConfig()
        )
        self.llm_judge = LLMJudge(config=judge_config) if use_llm_judge else None
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

        # LLM judge path (uncertain cases). A provider outage or a missing key
        # must never take the endpoint down — on any judge failure we degrade
        # to the NLI score alone and flag it in the breakdown.
        llm_start = time.time()
        try:
            llm_score = self.llm_judge.score(query, context, response)
        except Exception as e:
            logger.warning(
                "LLM judge unavailable (%s: %s); using NLI score only",
                type(e).__name__, e,
            )
            return ScoringBreakdown(
                nli_score=nli_score,
                llm_judge_score=None,
                final_score=nli_score,
                nli_latency_ms=nli_latency,
                llm_latency_ms=(time.time() - llm_start) * 1000,
                used_llm_judge=False,
                judge_failed=True,
            )
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
