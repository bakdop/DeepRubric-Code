#!/usr/bin/env python3
"""
Build a self-consistent Wikipedia subset for DeepRubric single-sample runs.

The released ASearcher assets ship only the raw jsonl files; the FAISS index and
the e5 encoder are not included, and `wikilinks` ships as jsonl while
`recursive_qa_agent_v4.py` expects a single json dict. This script produces a
scaled-down but structurally identical asset directory:

  <out>/wiki_corpus.jsonl     retrieval passages   (index row i == line i)
  <out>/wiki_webpages.jsonl   full pages for root seeding
  <out>/wikilinks.json        single json dict, as json.load() expects

Sampling is done at ARTICLE level (keep every passage of a kept article) so that
retrieval inside the subset still supports multi-hop drilling on kept topics.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path


def keep_article(wikipedia_id: str, keep_one_in: int) -> bool:
    """Stable hash so corpus / webpages / links agree on the same article set."""
    return zlib.crc32(str(wikipedia_id).encode()) % keep_one_in == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/scratch/bl4363/data/ASearcher-Local-Knowledge")
    ap.add_argument("--out", default="/scratch/bl4363/data/ASearcher-subset")
    ap.add_argument("--keep-one-in", type=int, default=20,
                    help="Keep 1 of every N articles (20 => ~5%% of Wikipedia).")
    ap.add_argument("--max-passages", type=int, default=2_000_000,
                    help="Hard cap on retrieval passages.")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: corpus (defines the index rows) -------------------------
    kept_urls: set[str] = set()
    n_in = n_out = 0
    with (src / "wiki_corpus.jsonl").open("r", encoding="utf-8") as fin, \
         (out / "wiki_corpus.jsonl").open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not keep_article(rec.get("wikipedia_id", ""), args.keep_one_in):
                continue
            # Re-number `id` so it matches the row index in the subset file.
            rec["id"] = n_out
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept_urls.add(rec.get("url", ""))
            n_out += 1
            if n_out >= args.max_passages:
                break
            if n_out % 200_000 == 0:
                print(f"[corpus] kept {n_out:,} / read {n_in:,}", flush=True)
    print(f"[corpus] DONE kept {n_out:,} passages from {n_in:,} lines, "
          f"{len(kept_urls):,} distinct urls", flush=True)

    # ---- pass 2: webpages (root seeding) ---------------------------------
    n_pages = 0
    with (src / "wiki_webpages.jsonl").open("r", encoding="utf-8") as fin, \
         (out / "wiki_webpages.jsonl").open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("url", "") not in kept_urls:
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_pages += 1
    print(f"[pages] DONE kept {n_pages:,} pages", flush=True)

    # ---- pass 3: wikilinks jsonl -> single json dict ---------------------
    links: dict[str, dict] = {}
    n_link_lines = 0
    with (src / "wikilinks.jsonl").open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            n_link_lines += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # Each line is {url: {...}}; keep only urls whose page survived.
            for url, payload in rec.items():
                if url in kept_urls:
                    links[url] = payload
    with (out / "wikilinks.json").open("w", encoding="utf-8") as fout:
        json.dump(links, fout, ensure_ascii=False)
    print(f"[links] DONE kept {len(links):,} of {n_link_lines:,} link records", flush=True)

    meta = {
        "keep_one_in": args.keep_one_in,
        "passages": n_out,
        "pages": n_pages,
        "link_entries": len(links),
        "source": str(src),
    }
    (out / "subset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
