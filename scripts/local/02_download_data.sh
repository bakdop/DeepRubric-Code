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
[ -n "${HTTPS_PROXY:-}${ALL_PROXY:-}" ] && {
  echo "WARNING: a proxy variable is set. hf_transfer does not reliably honour it;"
  echo "         export HF_HUB_ENABLE_HF_TRANSFER=0 if downloads stall, or unset"
  echo "         HTTP_PROXY HTTPS_PROXY ALL_PROXY to go direct via the mirror."; }

# Fail early with a readable message instead of an httpx RemoteProtocolError deep
# in snapshot_download: repo_info() is the first thing hf hits, and a blocked or
# proxied endpoint drops that connection before any HTTP status is returned.
if ! curl -sf -o /dev/null --max-time 20 "${HF_ENDPOINT:-https://huggingface.co}/api/datasets/inclusionAI/ASearcher-Local-Knowledge"; then
  echo "ERROR: cannot reach ${HF_ENDPOINT:-https://huggingface.co}" >&2
  echo "  - is HF_ENDPOINT exported before starting this script?" >&2
  echo "  - stale proxy vars? unset HTTP_PROXY HTTPS_PROXY ALL_PROXY" >&2
  exit 1
fi
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
