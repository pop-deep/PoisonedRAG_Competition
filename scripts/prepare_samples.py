"""Inspect / re-split the prepared test samples.

``scripts/build_local_corpus.py`` already produces the public/hidden split
(50/50 per spec). This utility prints sample stats and can re-split with a
different seed for cross-validation.
"""
from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
from collections import Counter

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import load_jsonl, save_jsonl


def stats(samples: list[dict]) -> str:
    ds = Counter(s["dataset"] for s in samples)
    return f"total={len(samples)} datasets={dict(ds)}"


def main():
    ap = argparse.ArgumentParser(description="Inspect / re-split samples.")
    ap.add_argument("--split", default=None, choices=["public", "hidden", None])
    ap.add_argument("--reseed", type=int, default=None, help="re-split with this seed")
    args = ap.parse_args()

    if args.reseed is not None:
        import random
        pub = load_jsonl("data/samples/public.jsonl")
        hid = load_jsonl("data/samples/hidden.jsonl")
        all_s = pub + hid
        rng = random.Random(args.reseed)
        rng.shuffle(all_s)
        half = len(all_s) // 2
        save_jsonl(all_s[:half], "data/samples/public.jsonl")
        save_jsonl(all_s[half:], "data/samples/hidden.jsonl")
        print(f"Re-split with seed={args.reseed}: public={half}, hidden={len(all_s)-half}")
        return

    if args.split:
        s = load_jsonl(f"data/samples/{args.split}.jsonl")
        print(f"{args.split}: {stats(s)}")
        print(json.dumps(s[0], ensure_ascii=False, indent=2))
    else:
        for sp in ["public", "hidden"]:
            s = load_jsonl(f"data/samples/{sp}.jsonl")
            print(f"{sp}: {stats(s)}")


if __name__ == "__main__":
    main()
