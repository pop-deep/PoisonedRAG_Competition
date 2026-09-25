"""Base class for all attack methods.

Attacks are fully decoupled from defenses and from the judge: an attack only
receives an :class:`~core.env.env.AttackContext` (question + attack_target +
black-box query + constraints) and returns a list of poison strings. Adding a
new attack = drop a new file in this package that subclasses :class:`BaseAttack`
and set its ``name``; the registry picks it up automatically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.env.env import AttackContext


class BaseAttack(ABC):
    """Abstract attack interface."""

    name: str = "base"

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    def attack(self, ctx: AttackContext) -> list[str]:
        """Return a list of poison documents (<= ctx.max_poison_docs)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Attack {self.name}>"
