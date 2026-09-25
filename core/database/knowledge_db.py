"""Knowledge database with the defense-facing interface.

The judge holds a :class:`KnowledgeDB` directly (full access including poison
audit info). The defense receives a :class:`DefenseDBProxy` that exposes ONLY
the spec-allowed operations (``list_documents / delete / quarantine /
update / reindex``) plus an operation counter. Poison labels, gold answers
and attack targets are never exposed through the proxy - enforcing the
isolation required by the spec (section 五).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DocumentView:
    """A read-only view of a document exposed to the defense.

    Deliberately carries NO poison label, NO gold-answer information.
    """
    id: str
    text: str
    source: str = "unknown"

    @property
    def length(self) -> int:
        return len(self.text)


class KnowledgeDB:
    """Internal database used by the judge.

    ``active`` docs are searchable; ``quarantined`` docs are removed from the
    searchable index but retained for audit; deleted docs are gone.
    Poison documents carry an internal flag used only by the evaluator.
    """

    def __init__(self, retriever, docs: Optional[list[dict]] = None,
                 max_operations: int = 10):
        self._retriever = retriever
        self._docs: dict[str, dict] = {}      # id -> {text, source, is_poison}
        self._active: set[str] = set()        # searchable ids
        self._quarantined: dict[str, dict] = {}
        self._deleted: set[str] = set()
        self._op_count = 0
        self._max_ops = max_operations
        self._index_dirty = True
        self._index_ids: list[str] = []
        self._index_emb: Optional[np.ndarray] = None
        if docs:
            for d in docs:
                self._add_doc(
                    d["id"], d["text"],
                    d.get("source", "unknown"),
                    d.get("is_poison", False),
                )

    # ------------------------------------------------------------------
    # Judge-side API (NOT exposed to the defense)
    # ------------------------------------------------------------------
    def _add_doc(self, doc_id: str, text: str, source: str = "unknown",
                 is_poison: bool = False) -> None:
        self._docs[doc_id] = {"text": text, "source": source, "is_poison": is_poison}
        self._active.add(doc_id)
        self._index_dirty = True

    def add_poison(self, texts: list[str], source: str = "web") -> list[str]:
        """Judge-only: insert poison documents.

        ``source`` is a generic, non-revealing label (poison docs blend in with
        the clean corpus - the spec forbids exposing the real poison label).
        """
        ids = []
        for i, t in enumerate(texts):
            pid = f"__poison__{id(self)}__{i}__{len(self._docs)}"
            self._add_doc(pid, t, source, is_poison=True)
            ids.append(pid)
        return ids

    def poison_ids(self) -> list[str]:
        """Judge-only audit: ids of poison docs currently active."""
        return [i for i in self._active if self._docs[i]["is_poison"]]

    def audit(self) -> dict:
        return {
            "active": len(self._active),
            "quarantined": len(self._quarantined),
            "deleted": len(self._deleted),
            "poison_active": len(self.poison_ids()),
            "poison_quarantined": sum(1 for d in self._quarantined.values() if d["is_poison"]),
            "operations": self._op_count,
        }

    # ------------------------------------------------------------------
    # Defense-facing API (public; also used by judge)
    # ------------------------------------------------------------------
    def list_documents(self) -> list[DocumentView]:
        return [
            DocumentView(id=i, text=self._docs[i]["text"], source=self._docs[i]["source"])
            for i in sorted(self._active)
        ]

    def _consume_ops(self, n: int) -> bool:
        if self._op_count + n > self._max_ops:
            print(f"[db] operation limit reached ({self._op_count}/{self._max_ops}); "
                  f"request for {n} more refused.")
            return False
        self._op_count += n
        return True

    def delete(self, document_ids: list[str]) -> None:
        ids = [i for i in document_ids if i in self._active]
        if not ids:
            return
        if not self._consume_ops(len(ids)):
            return
        for i in ids:
            self._active.discard(i)
            self._deleted.add(i)
        self._index_dirty = True

    def quarantine(self, document_ids: list[str]) -> None:
        ids = [i for i in document_ids if i in self._active]
        if not ids:
            return
        if not self._consume_ops(len(ids)):
            return
        for i in ids:
            self._quarantined[i] = self._docs[i]
            self._active.discard(i)
        self._index_dirty = True

    def update(self, document_id: str, new_text: str) -> None:
        if document_id not in self._active:
            return
        if not self._consume_ops(1):
            return
        self._docs[document_id]["text"] = new_text
        self._index_dirty = True

    def reindex(self) -> None:
        """Rebuild the embedding cache over all active documents."""
        self._index_ids = sorted(self._active)
        texts = [self._docs[i]["text"] for i in self._index_ids]
        self._index_emb = self._retriever.embed(texts) if texts else None
        self._index_dirty = False

    @property
    def operation_count(self) -> int:
        return self._op_count

    @property
    def max_operations(self) -> int:
        return self._max_ops

    def __len__(self) -> int:
        return len(self._active)

    # ------------------------------------------------------------------
    # Retrieval (used by the judge after defense runs)
    # ------------------------------------------------------------------
    def search(self, question: str, top_k: int) -> list[dict]:
        if self._index_dirty:
            self.reindex()
        if not self._index_ids or self._index_emb is None:
            return []
        q_emb = self._retriever.embed([question])[0]
        scores = self._retriever._score(q_emb, self._index_emb)
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "id": self._index_ids[i],
                "text": self._docs[self._index_ids[i]]["text"],
                "score": float(scores[i]),
                "is_poison": self._docs[self._index_ids[i]]["is_poison"],
            }
            for i in order
        ]


class DefenseDBProxy:
    """Restricted view of :class:`KnowledgeDB` handed to the defense.

    Only the spec-allowed operations are reachable. No poison labels, gold
    answers, or audit internals are exposed.
    """

    def __init__(self, db: KnowledgeDB):
        self._db = db

    def list_documents(self) -> list[DocumentView]:
        return self._db.list_documents()

    def delete(self, document_ids: list[str]) -> None:
        self._db.delete(document_ids)

    def quarantine(self, document_ids: list[str]) -> None:
        self._db.quarantine(document_ids)

    def update(self, document_id: str, new_text: str) -> None:
        self._db.update(document_id, new_text)

    def reindex(self) -> None:
        self._db.reindex()

    @property
    def operation_count(self) -> int:
        return getattr(self._db, "operation_count", 0)

    @property
    def max_operations(self) -> int:
        return getattr(self._db, "max_operations", 10)

    def __len__(self) -> int:
        return len(self._db)

    def __repr__(self) -> str:
        return (f"<DefenseDBProxy active={len(self)} "
                f"ops={self.operation_count}/{self.max_operations}>")
