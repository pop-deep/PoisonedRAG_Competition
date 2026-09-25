"""Download the Contriever retriever model locally (optional).

Enables the ``contriever`` retriever backend (faithful to PoisonedRAG). The
default ``tfidf`` backend needs no download.

Example:
    python scripts/download_retriever.py --model facebook/contriever
"""
from __future__ import annotations

import argparse
import os as _os
import sys as _sys
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.config import resolve_path


def main():
    ap = argparse.ArgumentParser(description="Download a retriever model to the local cache.")
    ap.add_argument("--model", default="facebook/contriever")
    ap.add_argument("--cache_dir", default="data/models")
    args = ap.parse_args()

    cache_dir = Path(resolve_path(args.cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model} -> {cache_dir}")
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("transformers is not installed. Run: pip install transformers torch")
        return
    AutoTokenizer.from_pretrained(args.model, cache_dir=str(cache_dir))
    AutoModel.from_pretrained(args.model, cache_dir=str(cache_dir))
    print(f"Done. Set retriever.backend=contriever and "
          f"retriever.model_cache_dir={args.cache_dir} in config/default.yaml.")


if __name__ == "__main__":
    main()
