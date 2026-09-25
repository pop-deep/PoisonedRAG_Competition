"""Competition submission entry for the ATTACK side.

Implements the spec interface:

    def attack(env) -> list[str]:
        return poisoned_documents   # max 5 strings, each <= 1024 UTF-8 bytes

Delegates to any attack method registered in the ``attacks/`` package
(decoupled). The method is selected by the ``ATTACK_METHOD`` environment
variable (default: ``lm_targeted``).

The competition ``env`` is expected to expose ``.question`` and
``.attack_target`` (and optionally a black-box query interface). It is adapted
to the internal :class:`AttackContext` here.
"""
from __future__ import annotations

import os
from typing import Any

from attacks import get_attack, list_attacks
from core.env.env import AttackContext

DEFAULT_ATTACK = os.environ.get("ATTACK_METHOD", "lm_targeted")


def attack(env: Any) -> list[str]:
    """Return a list of poison documents (<= 5 strings)."""
    name = os.environ.get("ATTACK_METHOD", DEFAULT_ATTACK)
    atk = get_attack(name)

    question = getattr(env, "question", "")
    attack_target = getattr(env, "attack_target", "")
    sample_id = getattr(env, "sample_id", "") or getattr(env, "id", "")

    # Adapt an optional black-box query interface on the competition env.
    bb = getattr(env, "query_blackbox", None) or getattr(env, "blackbox_query", None)
    last_topk = list(getattr(env, "last_topk", []) or [])

    ctx = AttackContext(
        question=question,
        attack_target=attack_target,
        sample_id=sample_id,
        blackbox_query=bb,
        last_topk=last_topk,
    )
    docs = atk.attack(ctx) or []
    # Hard-enforce spec limits at the submission boundary.
    docs = [d for d in docs if isinstance(d, str) and d.strip()][:5]
    out = []
    for d in docs:
        b = d.encode("utf-8", errors="ignore")
        if len(b) > 1024:
            b = b[:1024]
            d = b.decode("utf-8", errors="ignore")
        out.append(d)
    return out


if __name__ == "__main__":  # quick local sanity check
    print("Registered attacks:", list_attacks())
