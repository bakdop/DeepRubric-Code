#!/usr/bin/env bash
# Build the retriever assets the ASearcher release does NOT ship:
#   1. a self-consistent corpus/pages/links subset (and wikilinks.json)
#   2. the FAISS index, built in parallel across all GPUs, then merged
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "==> [1/3] building subset (KEEP_ONE_IN=$KEEP_ONE_IN MAX_PASSAGES=$MAX_PASSAGES)"
"$PY" "$DR_ROOT/tools/prep_wiki_subset.py" \
  --src "$WIKI_RAW" --out "$WIKI_ASSETS" \
  --keep-one-in "$KEEP_ONE_IN" --max-passages "$MAX_PASSAGES"

IFS=',' read -ra GPUS <<< "$INDEX_GPUS"
N=${#GPUS[@]}
mkdir -p "$WIKI_ASSETS/e5.index"

echo "==> [2/3] encoding on $N GPUs"
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" "$DR_ROOT/tools/build_e5_index.py" \
    --corpus "$WIKI_ASSETS/wiki_corpus.jsonl" \
    --model  "$WIKI_ASSETS/e5-base-v2" \
    --out-index "$WIKI_ASSETS/e5.index/shard_$(printf '%02d' "$i").index" \
    --shard-id "$i" --num-shards "$N" \
    --batch-size "${INDEX_BATCH:-1024}" \
    > "$LOGDIR/index_shard_$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "a shard failed; see $LOGDIR/index_shard_*.log" >&2; exit 1; }

echo "==> [3/3] merging shards in order"
"$PY" "$DR_ROOT/tools/merge_faiss_shards.py" \
  --shard-glob "$WIKI_ASSETS/e5.index/shard_*.index" \
  --out-index  "$WIKI_ASSETS/e5.index/e5_Flat.index" \
  --corpus     "$WIKI_ASSETS/wiki_corpus.jsonl"

rm -f "$WIKI_ASSETS"/e5.index/shard_*.index
ls -la "$WIKI_ASSETS/e5.index/"
