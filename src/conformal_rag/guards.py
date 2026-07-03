"""Security guards for conformal-rag."""

from __future__ import annotations

import re
from typing import List, Tuple


# Prompt injection patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior",
    r"you\s+are\s+now",
    r"new\s+instructions?:",
    r"system\s*:\s*",
    r"score\s+this\s+as\s+\d",
    r"return\s+only\s+\d+\.?\d*",
]

# PII patterns
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def sanitize_query(query: str) -> str:
    """Remove prompt injection attempts from query."""
    if not query:
        return query

    result = query
    for pattern in _INJECTION_PATTERNS:
        result = re.sub(pattern, "[REDACTED]", result, flags=re.IGNORECASE)

    return result


def clamp_alpha(alpha: float) -> float:
    """Clamp alpha to valid range [0.001, 0.2]."""
    return max(0.001, min(0.2, alpha))


def check_calibration_coverage(n_samples: int) -> List[str]:
    """Check if calibration set is large enough. Returns list of warnings."""
    warnings = []
    if n_samples < 100:
        warnings.append(
            f"Calibration set has {n_samples} samples. "
            f"Minimum recommended is 100. FDR guarantee may be unreliable."
        )
    return warnings


def detect_pii(text: str) -> bool:
    """Detect if text contains PII (email, phone, SSN)."""
    if _EMAIL_PATTERN.search(text):
        return True
    if _PHONE_PATTERN.search(text):
        return True
    if _SSN_PATTERN.search(text):
        return True
    return False


def validate_label_distribution(labels: List[int]) -> Tuple[bool, str]:
    """Validate that label distribution is not degenerate."""
    if not labels:
        return False, "Empty label list"

    unique = set(labels)
    if len(unique) == 1:
        return False, f"All labels are the same ({labels[0]}). Need both classes."

    n_hallucinations = sum(labels)
    n_total = len(labels)
    hallucination_rate = n_hallucinations / n_total

    if hallucination_rate > 0.5:
        return False, (
            f"Hallucination rate is {hallucination_rate:.1%}. "
            f"Expected ~10-20%. Calibration may be biased."
        )

    if hallucination_rate < 0.01:
        return False, (
            f"Hallucination rate is {hallucination_rate:.1%}. "
            f"Too few hallucinations for meaningful calibration."
        )

    return True, "Distribution is valid."
