"""Input guardrails: cheap deterministic checks before any LLM spend.

Output validation is handled structurally: the agent's final answer is parsed
into the `Solution` schema (guaranteed shape) and cross-checked by whether a
SymPy tool actually ran during the solve (server-observed, not self-reported).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings

MATH_KEYWORDS = {
    "algebra", "equation", "solve", "calculate", "compute", "derivative",
    "integral", "integrate", "differentiate", "function", "polynomial",
    "matrix", "vector", "geometry", "triangle", "circle", "angle", "theorem",
    "proof", "prove", "trigonometry", "sin", "cos", "tan", "logarithm", "log",
    "exponential", "statistics", "probability", "mean", "median", "variance",
    "calculus", "limit", "series", "sequence", "permutation", "combination",
    "prime", "factor", "quadratic", "linear", "differential", "sum", "product",
    "root", "fraction", "ratio", "percent", "area", "volume", "perimeter",
    "slope", "intercept", "domain", "range", "inequality", "modulo", "gcd",
    "lcm", "conjecture", "number", "digit", "graph",
}

MATH_SYMBOLS = set("+-*/=<>^√∫∑π²³×÷≤≥≠")

# Light PII scrubbing (emails / phone numbers) — questions shouldn't carry these
# to the API at all.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3}[\s-]?\d{3,4}[\s-]?\d{4}(?!\d)")


@dataclass
class GuardrailResult:
    ok: bool
    sanitized: str = ""
    error: str = ""
    guardrail: str = ""


def validate_input(question: str, has_image: bool = False) -> GuardrailResult:
    if not settings.ENABLE_INPUT_GUARDRAILS:
        return GuardrailResult(ok=True, sanitized=question)

    q = question.strip()

    if len(q) < 3:
        return GuardrailResult(False, error="Question is too short.", guardrail="length")
    if len(q) > settings.MAX_QUESTION_LENGTH:
        return GuardrailResult(
            False,
            error=f"Question exceeds {settings.MAX_QUESTION_LENGTH} characters.",
            guardrail="length",
        )

    # With an attached figure the text is often just "solve the problem in the
    # figure" — the topic check would false-positive, so skip it (the system
    # prompt still restricts the agent to mathematics).
    if not has_image and not _looks_mathematical(q):
        return GuardrailResult(
            False,
            error="This assistant only answers mathematics questions. "
                  "Please ask something math-related.",
            guardrail="topic",
        )

    sanitized = _EMAIL_RE.sub("[email removed]", q)
    sanitized = _PHONE_RE.sub("[number removed]", sanitized)
    return GuardrailResult(ok=True, sanitized=sanitized)


def _looks_mathematical(question: str) -> bool:
    lowered = question.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    if words & MATH_KEYWORDS:
        return True
    has_symbols = any(ch in MATH_SYMBOLS for ch in question)
    has_digits = bool(re.search(r"\d", question))
    return has_symbols and has_digits
