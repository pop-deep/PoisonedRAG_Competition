"""Shared text-feature helpers used by defense methods.

Internal module (prefixed with ``_``) - not registered as a defense.
"""
from __future__ import annotations

import re
from typing import Iterable

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_SENT = re.compile(r"[^.!?\n]+[.!?]?")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def word_count(text: str) -> int:
    return len(words(text))


def char_count(text: str) -> int:
    return len(text.encode("utf-8", errors="ignore"))


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def token_set(text: str) -> set:
    return set(words(text))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def lexical_overlap(query: str, text: str) -> float:
    """Fraction of query content words present in text (query perspective)."""
    q = {w for w in words(query) if len(w) > 2}
    if not q:
        return 0.0
    t = token_set(text)
    return len(q & t) / len(q)


def repetition_ratio(text: str) -> float:
    """Fraction of repeated word tokens (a proxy for stuffing / spamminess)."""
    w = words(text)
    if len(w) < 4:
        return 0.0
    from collections import Counter
    c = Counter(w)
    repeated = sum(v - 1 for v in c.values() if v > 1)
    return repeated / len(w)


def char_ngrams(text: str, n: int = 5) -> set:
    t = re.sub(r"\s+", " ", text.lower().strip())
    if len(t) < n:
        return {t}
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def ngram_jaccard(a: str, b: str, n: int = 5) -> float:
    return jaccard(char_ngrams(a, n), char_ngrams(b, n))
