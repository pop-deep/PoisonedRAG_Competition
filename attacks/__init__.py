"""Attack registry with auto-discovery.

Every module in this package that defines a ``BaseAttack`` subclass is imported
automatically; instances are created via :func:`get_attack` by ``name``.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Type

from attacks.base_attack import BaseAttack

_REGISTRY: dict[str, Type[BaseAttack]] = {}


def register(cls: Type[BaseAttack]) -> Type[BaseAttack]:
    """Class decorator to register an attack by its ``name`` attribute."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} has no 'name' attribute")
    _REGISTRY[name] = cls
    return cls


def _autodiscover() -> None:
    # Import every attack module so @register runs.
    for mod_info in pkgutil.iter_modules(__path__):
        if mod_info.name.startswith("_") or mod_info.name == "base_attack":
            continue
        importlib.import_module(f"attacks.{mod_info.name}")


def list_attacks() -> list[str]:
    _autodiscover()
    return sorted(_REGISTRY.keys())


def get_attack(name: str, **kwargs) -> BaseAttack:
    _autodiscover()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown attack '{name}'. Available: {list_attacks()}")
    return _REGISTRY[name](**kwargs)


_autodiscover()
