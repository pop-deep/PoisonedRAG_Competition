"""Build a lightweight local corpus + test samples from PoisonedRAG artifacts.

Reads ``PoisonedRAG-main/results/adv_targeted_results/{nq,hotpotqa,msmarco}.json``
(300 samples total, each with question / correct answer / incorrect answer /
5 GPT-4-generated adversarial texts) and produces:

* ``data/corpus/clean_corpus.jsonl``  - clean knowledge base (gold docs + filler)
* ``data/samples/public.jsonl``       - public test split (50%)
* ``data/samples/hidden.jsonl``       - hidden test split (50%)
* ``data/poisons/{dataset}.json``     - pre-generated poison texts keyed by sample id

This makes the full attack-defense pipeline runnable with ZERO external
downloads. For real BEIR corpora, run ``scripts/download_datasets.py``.
"""
from __future__ import annotations

import argparse
import json
import os as _os
import random
import re
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import PROJECT_ROOT, resolve_path, save_jsonl, save_json

_WH = re.compile(r"^(what|who|where|when|how many|how much|which|why|how)\s+", re.IGNORECASE)

# Natural gold-doc templates. They state the correct answer via "is <answer>"
# (extractable by the LLM/mock) WITHOUT using "the answer is" boilerplate, so
# style-based defenses can still distinguish them from templated poison.
_GOLD_TEMPLATES = [
    "{cap}. {cap} is {answer}. This is recorded in standard reference material.",
    "According to documented records, {topic} is {answer}. The information is verified across sources.",
    "Reference material states that {topic} is {answer}. {cap}.",
    "{cap}: the documented value is {answer}. Confirmed by multiple sources.",
    "Records indicate that {topic} is {answer}. {cap} is widely documented.",
]

_SOURCES = ["web", "wiki", "news", "web", "wiki"]  # generic, non-revealing labels


def _topic(question: str) -> str:
    q = question.strip().rstrip("?.").strip()
    q = _WH.sub("", q)
    return q.strip() or question.strip()


def _cap(s: str) -> str:
    s = s.strip()
    return s[0].upper() + s[1:] if s else s


def _make_gold_docs(sample_id: str, question: str, answer: str, n: int) -> list[dict]:
    topic = _topic(question)
    cap = _cap(topic)
    docs = []
    for i in range(n):
        tpl = _GOLD_TEMPLATES[i % len(_GOLD_TEMPLATES)]
        text = tpl.format(cap=cap, topic=topic, answer=answer)
        docs.append({
            "id": f"{sample_id}_gold{i}",
            "text": text,
            "source": _SOURCES[(hash(sample_id) + i) % len(_SOURCES)],
            "sample_id": sample_id,
        })
    return docs


def _load_source(dataset: str, source_dir: Path) -> dict:
    """Find {dataset}.json in the PoisonedRAG-main adv_targeted_results folder,
    tolerating the nested ``PoisonedRAG-main/PoisonedRAG-main`` layout."""
    candidates = [
        source_dir / f"{dataset}.json",
        source_dir / "PoisonedRAG-main" / "results" / "adv_targeted_results" / f"{dataset}.json",
        source_dir.parent / "results" / "adv_targeted_results" / f"{dataset}.json",
        source_dir.parent / "PoisonedRAG-main" / "results" / "adv_targeted_results" / f"{dataset}.json",
        # nested layout: <root>/PoisonedRAG-main/PoisonedRAG-main/results/...
        source_dir.parent.parent / "PoisonedRAG-main" / "results" / "adv_targeted_results" / f"{dataset}.json",
        source_dir.parent.parent / "PoisonedRAG-main" / "PoisonedRAG-main" / "results" / "adv_targeted_results" / f"{dataset}.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Could not find {dataset}.json. Tried: {[str(c) for c in candidates]}")


def build(num_gold: int = 2, num_filler: int = 0, seed: int = 12,
          datasets: list[str] = None, source_dir: str = None) -> dict:
    datasets = datasets or ["nq", "hotpotqa", "msmarco"]
    if source_dir:
        source_dir = Path(resolve_path(source_dir))
    else:
        # Default: <root>/PoisonedRAG-main/PoisonedRAG-main/results/adv_targeted_results
        source_dir = PROJECT_ROOT.parent / "PoisonedRAG-main" / "results" / "adv_targeted_results"

    rng = random.Random(seed)
    corpus: list[dict] = []
    samples: list[dict] = []
    poisons: dict[str, dict] = {ds: {} for ds in datasets}

    for ds in datasets:
        data = _load_source(ds, source_dir)
        for entry in data.values():
            entry_id = entry.get("id") or entry.get("_id")
            sample_id = f"{ds}_{entry_id}"
            question = entry["question"]
            gold_answer = str(entry.get("correct answer", entry.get("correct_answer", ""))).strip()
            attack_target = str(entry.get("incorrect answer", entry.get("incorrect_answer", ""))).strip()
            adv_texts = entry.get("adv_texts", [])
            if not question or not gold_answer or not attack_target:
                continue

            gold_docs = _make_gold_docs(sample_id, question, gold_answer, num_gold)
            corpus.extend(gold_docs)
            poisons[ds][sample_id] = adv_texts
            # Also key by raw id for tolerant lookup.
            poisons[ds][entry_id] = adv_texts

            samples.append({
                "id": sample_id,
                "dataset": ds,
                "question": question,
                "gold_answer": gold_answer,
                "attack_target": attack_target,
                "gold_doc_ids": [d["id"] for d in gold_docs],
            })

    # Optional generic filler docs (topical distractors without answers).
    if num_filler > 0:
        for i in range(num_filler):
            s = samples[i % len(samples)]
            topic = _topic(s["question"])
            corpus.append({
                "id": f"__filler__{i}",
                "text": f"{_cap(topic)}. This topic is discussed in general reference works.",
                "source": _SOURCES[i % len(_SOURCES)],
                "sample_id": "filler",
            })

    # 50/50 public/hidden split (per spec).
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    half = len(idx) // 2
    public = [samples[i] for i in idx[:half]]
    hidden = [samples[i] for i in idx[half:]]

    save_jsonl(corpus, "data/corpus/clean_corpus.jsonl")
    save_jsonl(public, "data/samples/public.jsonl")
    save_jsonl(hidden, "data/samples/hidden.jsonl")
    for ds in datasets:
        save_json(poisons[ds], f"data/poisons/{ds}.json")

    return {
        "corpus_docs": len(corpus),
        "samples": len(samples),
        "public": len(public),
        "hidden": len(hidden),
        "datasets": datasets,
    }


def main():
    ap = argparse.ArgumentParser(description="Build local corpus + samples.")
    ap.add_argument("--num_gold", type=int, default=2)
    ap.add_argument("--num_filler", type=int, default=0)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa", "msmarco"])
    ap.add_argument("--source_dir", default=None)
    args = ap.parse_args()
    stats = build(num_gold=args.num_gold, num_filler=args.num_filler, seed=args.seed,
                  datasets=args.datasets, source_dir=args.source_dir)
    print("Built local corpus + samples:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
