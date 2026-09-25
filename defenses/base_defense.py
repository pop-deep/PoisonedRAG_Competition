"""Base class for all defense methods.

Defenses are fully decoupled from attacks and from the judge: a defense only
receives a :class:`~core.env.env.DefenseEnv` (question + a restricted DB
proxy) and may call ``list_documents / delete / quarantine / update / reindex``
(up to ``max_operations`` mutations). It never sees attack_target, gold_answer
or poison labels. Adding a new defense = drop a new file in this package that
subclasses :class:`BaseDefense` and set its ``name``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.env.env import DefenseEnv


class BaseDefense(ABC):
    """Abstract defense interface."""

    name: str = "base"

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    def defend(self, env: DefenseEnv) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Defense {self.name}>"
