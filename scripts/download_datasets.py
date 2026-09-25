"""Download real BEIR datasets locally (optional, for stronger evaluation).

BEIR datasets are large (nq ~500MB, hotpotqa ~3.5GB, msmarco ~3GB). This
script downloads and unzips them into ``data/datasets/<dataset>/`` in the
standard BEIR layout (corpus.jsonl, queries.jsonl, qrels/<split>.tsv).

The default local-corpus pipeline (``scripts/build_local_corpus.py``) needs no
download; this script is provided for users who want the real corpus.

Examples:
    python scripts/download_datasets.py --dataset nq
    python scripts/download_datasets.py --datasets nq hotpotqa
"""
from __future__ import annotations

import argparse
import os
import os as _os
import sys as _sys
import zipfile
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import requests
from tqdm import tqdm

from core.config import resolve_path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def download(dataset: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / dataset
    if target.exists() and any(target.iterdir()):
        print(f"[skip] {target} already exists.")
        return target
    url = BEIR_URL.format(dataset)
    zip_path = out_dir / f"{dataset}.zip"
    print(f"Downloading {url} -> {zip_path}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(zip_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=1 << 20), total=total // (1 << 20),
                              unit="MB", desc=dataset):
                if chunk:
                    f.write(chunk)
    print(f"Unzipping {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    os.remove(zip_path)
    print(f"Done: {target}")
    return target


def main():
    ap = argparse.ArgumentParser(description="Download BEIR datasets.")
    ap.add_argument("--dataset", default=None, help="single dataset (nq|hotpotqa|msmarco)")
    ap.add_argument("--datasets", nargs="+", default=None, help="multiple datasets")
    ap.add_argument("--out", default="data/datasets", help="output directory")
    args = ap.parse_args()

    datasets = args.datasets or ([args.dataset] if args.dataset else ["nq"])
    out_dir = Path(resolve_path(args.out))
    for ds in datasets:
        download(ds, out_dir)


if __name__ == "__main__":
    main()
