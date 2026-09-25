"""Near-duplicate clustering + dedup.

Poison is frequently injected as several near-identical passages (the spec
allows up to 5). This defense clusters documents by character-ngram Jaccard
similarity and quarantines duplicates within a cluster, keeping one
representative (the longest passage). It also flags documents that are near-
duplicates of the question itself.
"""
from __future__ import annotations

from defenses import register
from defenses.base_defense import BaseDefense
from defenses._text_utils import ngram_jaccard, char_count
from core.env.env import DefenseEnv


@register
class DedupCluster(BaseDefense):
    name = "dedup_cluster"

    def __init__(self, dup_threshold: float = 0.6, q_threshold: float = 0.75,
                 max_quarantine: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.dup_threshold = dup_threshold
        self.q_threshold = q_threshold
        self.max_quarantine = max_quarantine

    def defend(self, env: DefenseEnv) -> None:
        docs = env.db.list_documents()
        if not docs:
            return

        # Sort by length desc so the longest (most informative) is the kept rep.
        docs_sorted = sorted(docs, key=lambda d: char_count(d.text), reverse=True)
        keep: list = []
        quarantine_ids: list[str] = []
        for d in docs_sorted:
            is_dup = False
            for k in keep:
                if ngram_jaccard(d.text, k.text) >= self.dup_threshold:
                    is_dup = True
                    break
            if is_dup:
                quarantine_ids.append(d.id)
            else:
                keep.append(d)

        # Also quarantine docs that are near-duplicates of the question.
        for d in docs:
            if ngram_jaccard(d.text, env.question) >= self.q_threshold:
                if d.id not in quarantine_ids and d.id in {x.id for x in keep}:
                    quarantine_ids.append(d.id)

        quarantine_ids = quarantine_ids[: min(self.max_quarantine, env.db.max_operations)]
        if quarantine_ids:
            env.db.quarantine(quarantine_ids)
            env.db.reindex()
