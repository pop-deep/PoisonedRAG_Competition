"""Run a single sample with a chosen attack + defense and print full details.

Useful for debugging the pipeline end-to-end.

Examples:
    python scripts/run_single.py --attack baseline_direct --defense anomaly_filter
    python scripts/run_single.py --attack lm_targeted --defense no_defense --sample 0
    python scripts/run_single.py --attack retrieval_optimized --defense query_consistency --split hidden
"""
from __future__ import annotations

import argparse
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import Config
from core.data_loader import load_samples
from core.judge.arena import Arena
from attacks import get_attack, list_attacks
from defenses import get_defense, list_defenses


def main():
    ap = argparse.ArgumentParser(description="Run one sample with one attack + defense.")
    ap.add_argument("--attack", default="baseline_direct", choices=list_attacks())
    ap.add_argument("--defense", default="no_defense", choices=list_defenses())
    ap.add_argument("--split", default="public", choices=["public", "hidden"])
    ap.add_argument("--sample", type=int, default=0, help="sample index in the split")
    ap.add_argument("--no_poison", action="store_true", help="run on clean DB (no attack)")
    ap.add_argument("--retriever", default=None)
    args = ap.parse_args()

    cfg = Config()
    arena = Arena(cfg, retriever_backend=args.retriever)
    samples = load_samples(args.split)
    if args.sample >= len(samples):
        print(f"sample index {args.sample} out of range (split has {len(samples)})")
        return
    sample = samples[args.sample]
    attack = get_attack(args.attack)
    defense = get_defense(args.defense)

    print(f"Sample: {sample['id']}  (dataset={sample['dataset']})")
    print(f"Question:      {sample['question']}")
    print(f"Gold answer:   {sample['gold_answer']}")
    print(f"Attack target: {sample['attack_target']}")
    print(f"Attack: {attack} | Defense: {defense}\n")

    res = arena.run_sample(sample, attack, defense, poisoned=not args.no_poison, verbose=True)

    print("\n--- Result ---")
    print(f"Response:      {res['response']!r}")
    print(f"Top-K ids:     {res['topk_ids']}")
    print(f"Poison injected: {res['poison_injected']} | Poison in Top-K: {res['poison_in_topk']}")
    print(f"DB audit:      {res['db_audit']}")
    print(f"matches_gold:  {res['matches_gold']}")
    print(f"matches_target:{res['matches_target']}")
    print(f"AttackSuccess: {res['attack_success']}")
    print(f"DefenseSuccess:{res['defense_success']}")


if __name__ == "__main__":
    main()
