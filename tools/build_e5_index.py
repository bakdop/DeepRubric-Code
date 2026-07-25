#!/usr/bin/env python3
"""
Build the FAISS index that `retrievers/wiki/code/local_retrieval_server.py` expects.

Contract enforced here (must match DenseRetriever/Encoder in that server):
  - e5 passages are encoded with the "passage: " prefix
  - mean pooling over the last hidden state, masked by attention_mask
  - L2 normalisation, float32
  - IndexFlatIP  => inner product on normalised vectors == cosine
  - index row i corresponds to line i of the corpus jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def iter_passages(corpus_path: Path, shard_id: int = 0, num_shards: int = 1):
    """Yield passage texts. With sharding, yield only this shard's CONTIGUOUS block
    so that concatenating shard indexes in id order reproduces corpus line order."""
    total = count_lines(corpus_path)
    per = (total + num_shards - 1) // num_shards
    lo, hi = shard_id * per, min((shard_id + 1) * per, total)
    with corpus_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= hi:
                break
            if i < lo or not line.strip():
                continue
            yield json.loads(line).get("contents", "")


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--shard-id", type=int, default=0,
                    help="Build only this contiguous shard (for multi-GPU builds).")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    out_index = Path(args.out_index)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # transformers>=5 renamed `torch_dtype` to `dtype`; support both.
    try:
        model = AutoModel.from_pretrained(args.model, dtype=torch.float16)
    except TypeError:
        model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.cuda().eval()
    dim = model.config.hidden_size
    print(f"[index] model={args.model} dim={dim}", flush=True)

    index = faiss.IndexFlatIP(dim)

    buf: list[str] = []
    n = 0

    @torch.no_grad()
    def flush(batch: list[str]) -> None:
        nonlocal n
        if not batch:
            return
        inputs = tok([f"passage: {t}" for t in batch],
                     max_length=args.max_length, padding=True,
                     truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        out = model(**inputs, return_dict=True)
        emb = mean_pool(out.last_hidden_state, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        index.add(emb.float().cpu().numpy().astype(np.float32, order="C"))
        n += len(batch)
        if n % (args.batch_size * 50) == 0:
            print(f"[index] encoded {n:,}", flush=True)

    for text in iter_passages(corpus_path, args.shard_id, args.num_shards):
        buf.append(text)
        if len(buf) >= args.batch_size:
            flush(buf)
            buf = []
    flush(buf)

    faiss.write_index(index, str(out_index))
    print(f"[index] DONE ntotal={index.ntotal:,} -> {out_index}", flush=True)

    # Row-alignment sanity check: index size must equal this shard's line count.
    n_lines = sum(1 for _ in iter_passages(corpus_path, args.shard_id, args.num_shards))
    print(f"[index] shard {args.shard_id}/{args.num_shards} lines={n_lines:,} "
          f"rows={index.ntotal:,} {'OK' if n_lines == index.ntotal else 'MISMATCH!'}", flush=True)


if __name__ == "__main__":
    main()
