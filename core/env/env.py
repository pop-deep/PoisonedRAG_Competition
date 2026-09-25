"""Environment objects exposed to attacks and defenses.

Information isolation (spec section 五):
* ``AttackContext`` exposes: question, attack_target, sample_id, an optional
  black-box retrieval query callback, and the most recent Top-K. It does NOT
  expose gold_answer or the defense source.
* ``DefenseEnv`` exposes: question and a restricted ``db`` proxy. It does NOT
  expose attack_target, gold_answer, or poison labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from core.database.knowledge_db import DefenseDBProxy


@dataclass
class AttackContext:
    question: str
    attack_target: str
    sample_id: str = ""
    # Optional black-box query: given a question, return the current Top-K
    # documents the RAG would retrieve (list of {"id","text","score"}).
    blackbox_query: Optional[Callable[[str], list[dict]]] = None
    last_topk: list[dict] = field(default_factory=list)
    # Constraints (from spec)
    max_poison_docs: int = 5
    max_bytes_per_doc: int = 1024

    def query_blackbox(self, question: Optional[str] = None) -> list[dict]:
        """Query the live RAG retriever. Returns [] if unavailable."""
        if self.blackbox_query is None:
            return []
        try:
            return self.blackbox_query(question or self.question)
        except Exception:
            return []


class DefenseEnv:
    """Defense-facing environment.

    ``db`` is a :class:`DefenseDBProxy` with only spec-allowed operations.
    """

    def __init__(self, question: str, db: DefenseDBProxy):
        self.question = question
        self.db = db

    def __repr__(self) -> str:
        return f"<DefenseEnv question={self.question!r} db={self.db!r}>"
