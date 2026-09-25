"""Run the full attack-defense round-robin arena.

Examples:
    python scripts/run_arena.py
    python scripts/run_arena.py --retriever tfidf --max-public 20 --max-hidden 20
    python scripts/run_arena.py --attacks baseline_direct lm_targeted \
        --defenses no_defense anomaly_filter

Results are printed and saved to results/arena_<timestamp>.json.
"""
from __future__ import annotations

import argparse
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import Config, load_yaml, resolve_path, save_json
from core.judge.arena import Arena
from attacks import get_attack, list_attacks
from defenses import get_defense, list_defenses


def _print_matrix(results: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Retriever: {results['retriever_backend']} | LLM: {results['llm_model']}")
    print(f"Samples: {results['num_samples']}")
    print("-" * 70)
    print("ASR matrix (rows=attack, cols=defense):")
    defs = results["defenses"]
    header = f"{'attack':<22}" + "".join(f"{d[:16]:>18}" for d in defs)
    print(header)
    for aname, row in results["asr_matrix"].items():
        print(f"{aname:<22}" + "".join(f"{row.get(d, 0):>18.3f}" for d in defs))

    print("\nAttack scores (avg ASR across defenses):")
    for aname, s in results["attack_scores"].items():
        print(f"  {aname:<22} ASR_avg={s['asr_avg']:.3f}")

    print("\nDefense scores (RobustAccuracy / CleanAccuracy / DefenseScore):")
    for dname, s in results["defense_scores"].items():
        print(f"  {dname:<22} RA_avg={s['robust_accuracy_avg']:.3f}  "
              f"CleanAcc={s['clean_accuracy']:.3f}  "
              f"Score={s['defense_score']:.3f}")
    print("=" * 70 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Run the attack-defense arena.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--arena_config", default="config/arena.yaml")
    ap.add_argument("--retriever", default=None, help="override retriever backend")
    ap.add_argument("--attacks", nargs="+", default=None, help="attack names (default: arena.yaml)")
    ap.add_argument("--defenses", nargs="+", default=None, help="defense names (default: arena.yaml)")
    ap.add_argument("--max_public", type=int, default=None)
    ap.add_argument("--max_hidden", type=int, default=None)
    ap.add_argument("--no_clean", action="store_true", help="skip CleanAccuracy measurement")
    ap.add_argument("--clean_samples", type=int, default=None)
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = Config(args.config)
    arena_cfg = load_yaml(args.arena_config)
    attack_names = args.attacks or arena_cfg.get("attacks", [])
    defense_names = args.defenses or arena_cfg.get("defenses", [])

    print(f"Available attacks: {list_attacks()}")
    print(f"Available defenses: {list_defenses()}")
    attacks = [get_attack(n) for n in attack_names]
    defenses = [get_defense(n) for n in defense_names]
    print(f"Running: attacks={attack_names} x defenses={defense_names}")

    arena = Arena(cfg, retriever_backend=args.retriever)
    ms = arena_cfg.get("max_samples") or {}
    max_samples = {}
    if args.max_public is not None:
        max_samples["public"] = args.max_public
    elif ms.get("public"):
        max_samples["public"] = ms["public"]
    if args.max_hidden is not None:
        max_samples["hidden"] = args.max_hidden
    elif ms.get("hidden"):
        max_samples["hidden"] = ms["hidden"]

    clean_samples = args.clean_samples or arena_cfg.get("clean_samples", 30)

    results = arena.run_matrix(
        attacks=attacks, defenses=defenses,
        splits=["public", "hidden"],
        max_samples=max_samples,
        measure_clean=not args.no_clean and arena_cfg.get("measure_clean_accuracy", True),
        clean_samples=clean_samples,
        verbose=not args.quiet,
    )
    _print_matrix(results)

    out = args.out or str(resolve_path("results/arena_latest.json"))
    save_json(results, out)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
