"""Random deletion baseline (spec section 六: 随机删除).

Quarantines a small random subset of documents. Serves as the lower-bound
defense baseline.
"""
from __future__ import annotations

import random

from defenses import register
from defenses.base_defense import BaseDefense
from core.env.env import DefenseEnv


@register
class RandomDelete(BaseDefense):
    name = "random_delete"

    def __init__(self, delete_ratio: float = 0.1, max_delete: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.delete_ratio = delete_ratio
        self.max_delete = max_delete

    def defend(self, env: DefenseEnv) -> None:
        docs = env.db.list_documents()
        if not docs:
            return
        n = min(self.max_delete, env.db.max_operations,
                max(1, int(len(docs) * self.delete_ratio)))
        ids = [d.id for d in random.sample(docs, min(n, len(docs)))]
        env.db.quarantine(ids)
        env.db.reindex()
