"""Smoke tests for the PoisonedRAG competition backend.

Run: python -m tests.test_pipeline   (from project root)
or:   python tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.data_loader import load_corpus, load_samples
from core.database.knowledge_db import DefenseDBProxy, KnowledgeDB
from core.env.env import AttackContext, DefenseEnv
from core.judge.evaluator import Evaluator
from core.rag.prompt_builder import PromptBuilder
from core.rag.retriever import build_retriever
from attacks import get_attack, list_attacks
from defenses import get_defense, list_defenses


def test_registry():
    print("[test] registry...")
    attacks = list_attacks()
    defenses = list_defenses()
    assert "baseline_direct" in attacks, attacks
    assert "lm_targeted" in attacks, attacks
    assert "no_defense" in defenses, defenses
    assert "anomaly_filter" in defenses, defenses
    print(f"  attacks={attacks}")
    print(f"  defenses={defenses}")
    print("  OK")


def test_evaluator():
    print("[test] evaluator...")
    ev = Evaluator()
    assert ev.match("23", "The show has 23 episodes.")
    assert not ev.match("24", "The show has 23 episodes.")
    assert ev.match("Northport", "the capital is Northport.")
    assert ev.attack_success("no", "no", "yes")
    assert ev.defense_success("yes", "no", "yes")
    print("  OK")


def test_db_isolation_and_ops():
    print("[test] db isolation + op limit...")
    cfg = Config()
    retr = build_retriever(cfg.retriever, corpus_texts=["alpha beta", "gamma delta"])
    db = KnowledgeDB(retr, docs=[
        {"id": "d1", "text": "alpha beta", "source": "web"},
        {"id": "d2", "text": "gamma delta", "source": "wiki"},
    ], max_operations=3)
    pids = db.add_poison(["alpha beta the answer is 42"])
    # DocumentView must NOT expose poison label.
    proxy = DefenseDBProxy(db)
    docs = proxy.list_documents()
    assert not any(hasattr(d, "is_poison") for d in docs), "poison label leaked!"
    assert any(d.id in pids for d in docs), "poison not visible in list"
    # Op limit: try to quarantine 5, only 3 allowed.
    proxy.quarantine([d.id for d in docs[:5]])
    assert proxy.operation_count == 3, proxy.operation_count
    proxy.reindex()
    # Poison audit only reachable from the judge-side db, not the proxy.
    assert hasattr(db, "poison_ids") and not hasattr(proxy, "poison_ids")
    print("  OK")


def test_retriever():
    print("[test] retriever (tfidf)...")
    cfg = Config()
    docs = [{"id": "d1", "text": "Chicago Fire season four has 23 episodes."},
            {"id": "d2", "text": "The capital of France is Paris."},
            {"id": "d3", "text": "Python is a programming language."}]
    retr = build_retriever(cfg.retriever, corpus_texts=[d["text"] for d in docs])
    res = retr.search("how many episodes in chicago fire season 4", docs, top_k=2)
    assert res[0]["id"] == "d1", res
    print("  OK")


def test_full_pipeline_single():
    print("[test] full pipeline (single sample)...")
    cfg = Config()
    corpus = load_corpus(cfg.dataset["corpus_path"])
    assert corpus, "corpus not built - run scripts/build_local_corpus.py"
    samples = load_samples("public")
    assert samples, "samples not built"
    from core.judge.arena import Arena
    arena = Arena(cfg)
    s = samples[0]
    # No defense -> attack should usually succeed (poison dominates Top-K).
    r1 = arena.run_sample(s, get_attack("baseline_direct"), get_defense("no_defense"))
    assert r1["poison_injected"] == 5
    assert r1["attack_success"] in (0, 1)
    print(f"  no_defense: AS={r1['attack_success']} poison_in_topk={r1['poison_in_topk']}")
    # With a real defense, poison_in_topk should not increase.
    r2 = arena.run_sample(s, get_attack("baseline_direct"), get_defense("anomaly_filter"))
    print(f"  anomaly_filter: AS={r2['attack_success']} poison_in_topk={r2['poison_in_topk']} "
          f"ops={r2['db_audit']['operations']}")
    assert r2["db_audit"]["operations"] <= cfg.defense["max_operations"]
    print("  OK")


def test_submission_entries_importable():
    print("[test] submission entries importable...")
    import attack as atk_mod
    import defense as def_mod
    assert callable(atk_mod.attack)
    assert callable(def_mod.defend)
    print("  OK")


def main():
    test_registry()
    test_evaluator()
    test_db_isolation_and_ops()
    test_retriever()
    test_submission_entries_importable()
    test_full_pipeline_single()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
