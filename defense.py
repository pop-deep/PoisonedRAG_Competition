"""Competition submission entry for the DEFENSE side.

Implements the spec interface:

    def defend(env) -> None:
        # operate on env.db (list_documents / delete / quarantine / update / reindex)

Delegates to any defense method registered in the ``defenses/`` package
(decoupled). The method is selected by the ``DEFENSE_METHOD`` environment
variable (default: ``query_consistency``).

The competition ``env`` is expected to expose ``.question`` and ``.db``. The
``db`` is wrapped in a :class:`DefenseDBProxy` so the defense sees a uniform
interface regardless of whether it runs in the local arena or the official
judge.
"""
from __future__ import annotations

import os
from typing import Any

from defenses import get_defense, list_defenses
from core.database.knowledge_db import DefenseDBProxy
from core.env.env import DefenseEnv

DEFAULT_DEFENSE = os.environ.get("DEFENSE_METHOD", "query_consistency")


def defend(env: Any) -> None:
    """Run the selected defense against ``env.db``."""
    name = os.environ.get("DEFENSE_METHOD", DEFAULT_DEFENSE)
    defense = get_defense(name)

    db = getattr(env, "db", None)
    # Wrap the competition DB in our proxy if it isn't already one.
    if not isinstance(db, DefenseDBProxy):
        db = _CompetitionDBProxy(db)
    defense_env = DefenseEnv(question=getattr(env, "question", ""), db=db)
    defense.defend(defense_env)


class _CompetitionDBProxy(DefenseDBProxy):
    """Adapts a competition ``env.db`` to the :class:`DefenseDBProxy` interface.

    The competition DB already implements list_documents/delete/quarantine/
    update/reindex (per spec); we just expose it through the proxy surface and
    provide a default operation budget if the DB doesn't track one.
    """

    def __init__(self, db: Any):
        # Bypass KnowledgeDB-specific init; we delegate to the competition db.
        self._db = db

    def list_documents(self):
        return self._db.list_documents()

    def delete(self, document_ids):
        return self._db.delete(document_ids)

    def quarantine(self, document_ids):
        return self._db.quarantine(document_ids)

    def update(self, document_id, new_text):
        return self._db.update(document_id, new_text)

    def reindex(self):
        return self._db.reindex()

    def __len__(self):
        try:
            return len(self._db.list_documents())
        except Exception:
            return 0


if __name__ == "__main__":  # quick local sanity check
    print("Registered defenses:", list_defenses())
