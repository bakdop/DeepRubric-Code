#!/usr/bin/env python3
"""
Merge per-shard FAISS indexes produced by build_e5_index.py --shard-id/--num-shards.

Shards cover CONTIGUOUS line ranges of the corpus, so concatenating them in
shard-id order reproduces corpus line order — which is the invariant
local_retrieval_server.py depends on (`load_docs(corpus, idxs)` treats a FAISS
row number as a corpus line number).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", required=True,
                    help="e.g. /path/e5.index/shard_*.index (sorted by shard id)")
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--corpus", default=None,
                    help="Optional corpus jsonl; if given, verify rows == lines.")
    args = ap.parse_args()

    paths = sorted(Path().glob(args.shard_glob) if not args.shard_glob.startswith("/")
                   else Path("/").glob(args.shard_glob.lstrip("/")))
    if not paths:
        raise FileNotFoundError(f"no shards matched: {args.shard_glob}")

    def shard_key(p: Path) -> int:
        digits = "".join(c for c in p.stem if c.isdigit())
        return int(digits) if digits else 0

    paths = sorted(paths, key=shard_key)
    print("[merge] shards in order:")
    for p in paths:
        print(f"   {p}")

    merged = None
    for p in paths:
        idx = faiss.read_index(str(p))
        if merged is None:
            merged = faiss.IndexFlatIP(idx.d)
        merged.add(idx.reconstruct_n(0, idx.ntotal))
        print(f"[merge] +{idx.ntotal:,} -> total {merged.ntotal:,}", flush=True)

    Path(args.out_index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(merged, args.out_index)
    print(f"[merge] DONE ntotal={merged.ntotal:,} -> {args.out_index}")

    if args.corpus:
        n = sum(1 for line in open(args.corpus, "rb") if line.strip())
        print(f"[merge] corpus lines={n:,} index rows={merged.ntotal:,} "
              f"{'OK' if n == merged.ntotal else 'MISMATCH!'}")


if __name__ == "__main__":
    main()
