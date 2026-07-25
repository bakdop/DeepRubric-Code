#!/usr/bin/env bash
# Download corpora + the e5 encoder.
#   Wikipedia   ~61GB   (required)
#   OpenScholar ~693GiB (only needed for the recursive_qa_agent_v44 branch)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
HF="$DR_VENV/bin/hf"
mkdir -p "$DR_DATA"

# --- behind a proxy / bandwidth-capped network -------------------------------
# Set HTTP_PROXY / HTTPS_PROXY / NO_PROXY in your shell; huggingface_hub uses
# requests and honours them. hf_transfer is a Rust downloader that does NOT
# reliably honour proxy env vars, so it is disabled here.
# A mirror usually beats a proxy outright:  export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
[ -n "${HF_ENDPOINT:-}" ] && echo "using HF_ENDPOINT=$HF_ENDPOINT"
[ -n "${HTTPS_PROXY:-}" ] && echo "using HTTPS_PROXY=$HTTPS_PROXY"
echo "NOTE: downloads resume; re-run this script after an interruption."

echo "==> Wikipedia (ASearcher-Local-Knowledge, ~61GB)"
"$HF" download inclusionAI/ASearcher-Local-Knowledge --repo-type dataset --local-dir "$WIKI_RAW"

echo "==> e5-base-v2 encoder (NOT shipped with the ASearcher release)"
# The repo carries onnx/openvino/pytorch_model.bin duplicates; fetching all of it
# costs 2.1GB for 439MB of useful weights.
"$HF" download intfloat/e5-base-v2 --local-dir "$WIKI_ASSETS/e5-base-v2" \
  --include "config.json" "model.safetensors" "tokenizer.json" \
             "tokenizer_config.json" "vocab.txt" "special_tokens_map.json"

if [ "${WITH_OPENSCHOLAR:-0}" = "1" ]; then
  echo "==> OpenScholar DataStore V3 (~693GiB)"
  "$HF" download OpenSciLM/OpenScholar-DataStore-V3 --repo-type dataset --local-dir "$OPENSCHOLAR_RAW"
fi

du -sh "$WIKI_RAW" "$WIKI_ASSETS/e5-base-v2" 2>/dev/null || true
