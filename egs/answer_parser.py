"""Strict final-marker parsing and exact rational normalization (no float rounding)."""
from __future__ import annotations
import re
from dataclasses import dataclass
from fractions import Fraction

INVALID = "__INVALID__"
# Find markers independently of their payload: a malformed LAST marker is invalid.
_MARKER = re.compile(r"final\s+answer\s*[:：]", re.I)
_INTEGER = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_DECIMAL = rf"(?:{_INTEGER}(?:\.\d*)?|\.\d+)"
_NUMBER = re.compile(rf"[+-]?\s*\$?\s*[+-]?\s*{_DECIMAL}(?:\s*/\s*[+-]?{_DECIMAL})?")

@dataclass(frozen=True)
class ParseResult:
    answer: str
    raw: str | None
    n_matches: int


def normalize_number(s: str, int_tolerance: float = 0.0) -> str | None:
    # Kept as a keyword for compatibility; approximate integer snapping is unsafe.
    if int_tolerance != 0:
        raise ValueError("normalization uses exact arithmetic; int_tolerance must be 0")
    s = s.strip().rstrip(".").strip()
    if not _NUMBER.fullmatch(s):
        return None
    s = re.sub(r"[,$\s]", "", s)
    try:
        parts = s.split("/")
        value = Fraction(parts[0])
        if len(parts) == 2:
            value /= Fraction(parts[1])
    except (ValueError, ZeroDivisionError):
        return None
    if value.denominator == 1:
        return str(value.numerator)
    # Terminating rationals are written as exact decimals; others as reduced p/q.
    d, twos, fives = value.denominator, 0, 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return f"{value.numerator}/{value.denominator}"
    places = max(twos, fives)
    scaled = abs(value.numerator) * (10 ** places // value.denominator)
    digits = str(scaled).zfill(places + 1)
    return ("-" if value < 0 else "") + digits[:-places] + "." + digits[-places:].rstrip("0")


def parse_answer(text: str, int_tolerance: float = 0.0) -> ParseResult:
    matches = list(_MARKER.finditer(text or ""))
    if not matches:
        return ParseResult(INVALID, None, 0)
    # Payload must be on the marker's line; never search the reasoning for numbers.
    tail = text[matches[-1].end():].split("\n", 1)[0].strip()
    tail = tail.lstrip("* ").rstrip("* ").strip()
    value = normalize_number(tail, int_tolerance)
    return ParseResult(value if value is not None else INVALID, tail, len(matches))


def parse_gold(answer_field: str) -> str:
    body, marker, tail = answer_field.rpartition("####")
    value = normalize_number(tail) if marker else None
    if value is None:
        raise ValueError(f"invalid GSM8K gold: {answer_field[-80:]!r}")
    return value


def clean_gsm8k_solution(answer_field: str) -> str:
    body = answer_field.rpartition("####")[0]
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nFinal answer: {parse_gold(answer_field)}"
