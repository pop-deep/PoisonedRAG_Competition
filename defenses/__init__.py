"""Defense registry with auto-discovery (mirrors the attack registry)."""
from __future__ import annotations

import importlib
import pkgutil
from typing import Type

from defenses.base_defense import BaseDefense

_REGISTRY: dict[str, Type[BaseDefense]] = {}


def register(cls: Type[BaseDefense]) -> Type[BaseDefense]:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} has no 'name' attribute")
    _REGISTRY[name] = cls
    return cls


def _autodiscover() -> None:
    for mod_info in pkgutil.iter_modules(__path__):
        if mod_info.name.startswith("_") or mod_info.name == "base_defense":
            continue
        importlib.import_module(f"defenses.{mod_info.name}")


def list_defenses() -> list[str]:
    _autodiscover()
    return sorted(_REGISTRY.keys())


def get_defense(name: str, **kwargs) -> BaseDefense:
    _autodiscover()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown defense '{name}'. Available: {list_defenses()}")
    return _REGISTRY[name](**kwargs)


_autodiscover()
