#!/usr/bin/env bash
# Import an ALREADY-BUILT subset from another machine instead of downloading
# 61GB of raw corpus from HuggingFace. Meant for bandwidth-capped clusters.
#
#   ~3.4GB  three subset files + the minimal e5 encoder  (rebuild index locally)
#   +4.0GB  the FAISS index too, if you would rather not spend the GPU minutes
#
# Run this ON THE TARGET BOX. Set SRC to user@host:/path/to/wiki-assets.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

: "${SRC:?set SRC=user@host:/path/to/wiki-assets  (the dir holding wiki_corpus.jsonl)}"
WITH_INDEX="${WITH_INDEX:-0}"

mkdir -p "$WIKI_ASSETS"

echo "==> pulling subset files (~2.9GB)"
rsync -avP --partial \
  "$SRC/wiki_corpus.jsonl" \
  "$SRC/wiki_webpages.jsonl" \
  "$SRC/wikilinks.json" \
  "$SRC/subset_meta.json" \
  "$WIKI_ASSETS/"

echo "==> pulling minimal e5 encoder (~439MB, skipping onnx/openvino/bin duplicates)"
mkdir -p "$WIKI_ASSETS/e5-base-v2"
rsync -avP --partial \
  --include='config.json' --include='model.safetensors' \
  --include='tokenizer.json' --include='tokenizer_config.json' \
  --include='vocab.txt' --include='special_tokens_map.json' \
  --exclude='*' \
  "$SRC/e5-base-v2/" "$WIKI_ASSETS/e5-base-v2/"

if [ "$WITH_INDEX" = "1" ]; then
  echo "==> pulling prebuilt FAISS index (~4GB)"
  mkdir -p "$WIKI_ASSETS/e5.index"
  rsync -avP --partial "$SRC/e5.index/e5_Flat.index" "$WIKI_ASSETS/e5.index/"
else
  echo "==> index NOT pulled; build it locally (minutes on 8 GPUs):"
  echo "    bash scripts/local/03_build_index.sh --index-only"
fi

du -sh "$WIKI_ASSETS"
echo
echo "Verify row alignment before serving:"
echo "  wc -l $WIKI_ASSETS/wiki_corpus.jsonl   # must equal index ntotal"
