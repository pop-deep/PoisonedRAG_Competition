"""Query-document consistency + contradiction detection.

Approach:
1. Score each document's "trustworthiness" - natural encyclopedic passages
   (moderate length, low boilerplate, low query-copy) are trustworthy; templated
   passages ("the answer is X", verbatim question, high repetition) are not.
2. Extract a short "claimed answer" span from each document.
3. The plurality claimed answer among *trustworthy* docs is the consensus.
4. Quarantine documents that contradict the consensus, prioritizing low-
   trustworthiness / high-query-overlap ones (within the op budget).

If poison outvotes gold among trustworthy docs, this defense cannot recover
the truth - which is the intended limitation.
"""
from __future__ import annotations

import re
from collections import Counter

from defenses import register
from defenses.base_defense import BaseDefense
from defenses._text_utils import words, repetition_ratio, lexical_overlap, char_count
from core.env.env import DefenseEnv

_BOILERPLATE = re.compile(r"\b(the answer is|answer:|correct answer is)\b", re.IGNORECASE)
_NUM = re.compile(r"\b\d{1,6}(?:[\.,]\d+)?\b")
_CAP_PHRASE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,3}\b")


def _claimed_answer(text: str) -> str:
    for pat in (_NUM, _CAP_PHRASE):
        m = pat.search(text)
        if m:
            return m.group(0).strip()
    w = words(text)
    return w[0] if w else ""


def _trustworthiness(doc, question: str, median_len: float) -> float:
    """Higher = more trustworthy (natural passage)."""
    text = doc.text
    wc = len(words(text))
    score = 1.0
    if median_len > 0:
        ratio = wc / median_len
        if ratio < 0.3 or ratio > 3.0:
            score -= 0.4
    if _BOILERPLATE.search(text):
        score -= 0.6
    rep = repetition_ratio(text)
    if rep > 0.2:
        score -= min(0.6, rep * 2)
    qov = lexical_overlap(question, text)
    if qov > 0.9:  # near-verbatim question copy
        score -= 0.5
    return score


@register
class QueryConsistency(BaseDefense):
    name = "query_consistency"

    def __init__(self, max_quarantine: int = 8, trust_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_quarantine = max_quarantine
        self.trust_threshold = trust_threshold

    def defend(self, env: DefenseEnv) -> None:
        docs = env.db.list_documents()
        if not docs:
            return
        import statistics
        lens = [len(words(d.text)) for d in docs]
        median_len = statistics.median(lens) if lens else 0

        scored = [(d, _trustworthiness(d, env.question, median_len)) for d in docs]
        trustworthy = [d for d, s in scored if s >= self.trust_threshold]
        if not trustworthy:
            trustworthy = [d for d, _ in sorted(scored, key=lambda x: -x[1])[: max(3, len(docs) // 3)]]

        # Consensus answer = plurality claim among trustworthy docs.
        claims = Counter(_claimed_answer(d.text) for d in trustworthy if _claimed_answer(d.text))
        consensus = claims.most_common(1)[0][0] if claims else None

        # Rank non-consensus docs by (low trust, high query overlap) for quarantine.
        suspects = []
        for d, s in scored:
            claim = _claimed_answer(d.text)
            if consensus and claim and claim != consensus:
                qov = lexical_overlap(env.question, d.text)
                suspects.append((s - qov, d.id))  # lower trust + higher overlap first
        suspects.sort()

        ids = [did for _, did in suspects][: min(self.max_quarantine, env.db.max_operations)]
        if ids:
            env.db.quarantine(ids)
            env.db.reindex()
