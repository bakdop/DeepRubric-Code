#!/usr/bin/env bash
# Generate evidence trees + questions + rubrics. Requires 04_serve.sh running.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

OUT="$OUT_DIR/wiki_tree_depth${MAX_DEPTH}"
mkdir -p "$OUT"

cd "$DR_ROOT/data_construction"     # imports are relative to this dir
"$PY" recursive_qa_agent_v4.py \
  --pages "$WIKI_ASSETS/wiki_webpages.jsonl" \
  --links "$WIKI_ASSETS/wikilinks.json" \
  --save "$OUT" \
  --retriever-url "$WIKI_RETRIEVER_URL" \
  --qa-model "$SERVED_MODEL_NAME" \
  --extract-model "$SERVED_MODEL_NAME" \
  --max_depth "$MAX_DEPTH" \
  --n "$N_SAMPLES" 2>&1 | tee "$LOGDIR/generate_depth${MAX_DEPTH}.log"

echo "artifacts:"; ls -la "$OUT"
