"""Configuration loading and path resolution.

Paths in the YAML/JSON configs are relative to the project root
(the folder containing this package's parent: ``PoisonedRAG_Competition/``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Return the PoisonedRAG_Competition project root."""
    # core/config.py -> core/ -> project root
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _project_root()


def resolve_path(p: str | os.PathLike) -> Path:
    """Resolve a path that may be relative to the project root."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def load_yaml(path: str | os.PathLike) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | os.PathLike) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)


def load_jsonl(path: str | os.PathLike) -> list[dict]:
    items = []
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str | os.PathLike) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


class Config:
    """Typed accessor over config/default.yaml."""

    def __init__(self, path: str | os.PathLike = "config/default.yaml"):
        self.raw = load_yaml(path)
        self.root = PROJECT_ROOT

    @property
    def dataset(self) -> dict:
        return self.raw["dataset"]

    @property
    def retriever(self) -> dict:
        return self.raw["retriever"]

    @property
    def llm(self) -> dict:
        return self.raw["llm"]

    @property
    def defense(self) -> dict:
        return self.raw["defense"]

    @property
    def attack(self) -> dict:
        return self.raw["attack"]

    @property
    def eval(self) -> dict:
        return self.raw["eval"]
