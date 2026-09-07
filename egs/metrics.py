"""Finite distribution metrics; invalid and censored outcomes retain their mass."""
import math
from collections import Counter
from .answer_parser import INVALID
TRUNCATED = "__TRUNCATED__"


def histogram(answers, weights=None):
    answers = list(answers)
    weights = [1.0] * len(answers) if weights is None else list(weights)
    if len(answers) != len(weights) or not answers:
        raise ValueError("nonempty aligned answers and weights required")
    out = Counter()
    for answer, weight in zip(answers, weights):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("invalid estimator weight")
        out[answer] += weight
    total = sum(out.values())
    if total <= 0:
        raise ValueError("zero estimator mass")
    return {key: value / total for key, value in sorted(out.items()) if value > 0}


def validate_distribution(p):
    if not p or any(not math.isfinite(v) or v < 0 for v in p.values()) or not math.isclose(sum(p.values()), 1, abs_tol=1e-8):
        raise ValueError("distribution must have unit nonnegative mass")


def entropy(p, units="bits"):
    validate_distribution(p)
    return -sum(v * math.log(v) for v in p.values() if v) / (math.log(2) if units == "bits" else 1)


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits (NOT its square-root distance)."""
    validate_distribution(p)
    validate_distribution(q)
    value = 0.0
    for key in p.keys() | q.keys():
        a, b = p.get(key, 0), q.get(key, 0)
        m = (a + b) / 2
        if a:
            value += a * math.log2(a / m) / 2
        if b:
            value += b * math.log2(b / m) / 2
    return max(value, 0.0)


def summary(p, gold, ref=None):
    validate_distribution(p)
    # Lexicographic tie breaking is frozen and includes invalid/censoring buckets.
    modal = min(p, key=lambda key: (-p[key], key))
    return {"modal_answer": modal, "modal_accuracy": int(modal == gold),
            "invalid_rate": p.get(INVALID, 0), "truncation_rate": p.get(TRUNCATED, 0),
            **({"js_bits": js_divergence(p, ref)} if ref is not None else {})}
