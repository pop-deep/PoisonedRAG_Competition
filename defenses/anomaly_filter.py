"""Anomaly text filtering.

Scores each document by anomaly signals and quarantines the most suspicious
ones (within the operation budget). Signals:

* abnormal length (very short or very long relative to the corpus median);
* high token-repetition ratio (word stuffing);
* verbatim or near-verbatim copy of the question (query stuffing);
* templated "the answer is X" phrasing repeated across the passage.
"""
from __future__ import annotations

import re
import statistics

from defenses import register
from defenses.base_defense import BaseDefense
from defenses._text_utils import words, repetition_ratio, lexical_overlap
from core.env.env import DefenseEnv

_ANSWER_PHRASE = re.compile(r"\bthe answer is\b", re.IGNORECASE)


@register
class AnomalyFilter(BaseDefense):
    name = "anomaly_filter"

    def __init__(self, max_quarantine: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_quarantine = max_quarantine

    def _score(self, doc, question: str, median_len: float) -> float:
        text = doc.text
        wc = len(words(text))
        score = 0.0
        # Length anomaly.
        if median_len > 0:
            ratio = wc / median_len
            if ratio < 0.25:
                score += 1.0
            elif ratio > 3.0:
                score += 0.8
        # Repetition.
        rep = repetition_ratio(text)
        if rep > 0.25:
            score += min(2.0, rep * 3)
        # Query stuffing.
        qov = lexical_overlap(question, text)
        if qov > 0.85:
            score += 1.2
        # "the answer is" boilerplate.
        if _ANSWER_PHRASE.search(text):
            score += 0.6
        return score

    def defend(self, env: DefenseEnv) -> None:
        docs = env.db.list_documents()
        if not docs:
            return
        lens = [len(words(d.text)) for d in docs]
        median_len = statistics.median(lens) if lens else 0
        scored = [(self._score(d, env.question, median_len), d.id) for d in docs]
        scored.sort(reverse=True)
        # Only quarantine positively-anomalous docs.
        ids = [did for sc, did in scored if sc > 0.0]
        ids = ids[: min(self.max_quarantine, env.db.max_operations)]
        if ids:
            env.db.quarantine(ids)
            env.db.reindex()
