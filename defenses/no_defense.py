"""No-op defense baseline (does nothing)."""
from __future__ import annotations

from defenses import register
from defenses.base_defense import BaseDefense
from core.env.env import DefenseEnv


@register
class NoDefense(BaseDefense):
    name = "no_defense"

    def defend(self, env: DefenseEnv) -> None:
        return None
