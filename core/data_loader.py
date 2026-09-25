"""Loaders for the local clean corpus and test samples.

Sample schema (data/samples/*.jsonl):
    {"id": "nq_test1", "dataset": "nq", "question": "...",
     "gold_answer": "...", "attack_target": "...",
     "gold_doc_ids": ["gold_nq_test1_0", ...]}

Corpus schema (data/corpus/clean_corpus.jsonl):
    {"id": "...", "text": "...", "source": "clean|filler", "sample_id": "..."}
"""
from __future__ import annotations

from typing import Optional

from core.config import Config, load_jsonl, resolve_path


def load_corpus(path: Optional[str] = None) -> list[dict]:
    path = path or Config().dataset["corpus_path"]
    return load_jsonl(path)


def load_samples(split: str = "public", path: Optional[str] = None) -> list[dict]:
    """split: 'public' | 'hidden' | 'all'."""
    cfg = Config()
    if path is None:
        path = cfg.dataset["samples"][split] if split != "all" else None
    if split == "all":
        return load_samples("public") + load_samples("hidden")
    return load_jsonl(path)


def load_poisons(dataset: str, poisons_dir: Optional[str] = None) -> dict:
    """Load pre-generated poison texts for a dataset, keyed by sample id."""
    cfg = Config()
    poisons_dir = poisons_dir or cfg.dataset["poisons_dir"]
    path = resolve_path(f"{poisons_dir}/{dataset}.json")
    if not path.exists():
        return {}
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
