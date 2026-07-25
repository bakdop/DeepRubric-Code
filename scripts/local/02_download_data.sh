#!/usr/bin/env bash
# Download corpora + the e5 encoder.
#   Wikipedia   ~61GB   (required)
#   OpenScholar ~693GiB (only needed for the recursive_qa_agent_v44 branch)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
HF="$DR_VENV/bin/hf"
mkdir -p "$DR_DATA"

# Mirrors are configured in config.sh (HF_ENDPOINT / PIP_INDEX_URL).
echo "HF_ENDPOINT           = ${HF_ENDPOINT:-https://huggingface.co}"
echo "HF_HUB_ENABLE_HF_TRANSFER = ${HF_HUB_ENABLE_HF_TRANSFER:-0}"
[ -n "${HTTPS_PROXY:-}" ] && {
  echo "WARNING: HTTPS_PROXY is set. hf_transfer does not reliably honour it;"
  echo "         export HF_HUB_ENABLE_HF_TRANSFER=0 if downloads stall."; }
echo "NOTE: downloads resume - just re-run this script after an interruption."

echo "==> Wikipedia (ASearcher-Local-Knowledge, ~61GB)"
"$HF" download inclusionAI/ASearcher-Local-Knowledge --repo-type dataset --local-dir "$WIKI_RAW"

echo "==> e5-base-v2 encoder (NOT shipped with the ASearcher release)"
# The repo carries onnx/openvino/pytorch_model.bin duplicates; fetching all of it
# costs 2.1GB for 439MB of useful weights.
"$HF" download intfloat/e5-base-v2 --local-dir "$WIKI_ASSETS/e5-base-v2" \
  --include "config.json" "model.safetensors" "tokenizer.json" \
             "tokenizer_config.json" "vocab.txt" "special_tokens_map.json"

# Required for the recursive_qa_agent_v44 (academic) branch. 694GiB:
#   passages/   353GiB  raw text, required
#   embeddings/ 341GiB  precomputed pes2o_contriever vectors; you can instead
#               recompute them locally with retrieval-scaling's src/embed.py
if [ "${WITH_OPENSCHOLAR:-0}" = "1" ]; then
  echo "==> OpenScholar DataStore V3 (~693GiB)"
  "$HF" download OpenSciLM/OpenScholar-DataStore-V3 --repo-type dataset --local-dir "$OPENSCHOLAR_RAW"
fi

du -sh "$WIKI_RAW" "$WIKI_ASSETS/e5-base-v2" 2>/dev/null || true
