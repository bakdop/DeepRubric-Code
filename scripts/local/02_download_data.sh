#!/usr/bin/env bash
# Download corpora + the e5 encoder.
#   Wikipedia   ~61GB   (required)
#   OpenScholar ~693GiB (only needed for the recursive_qa_agent_v44 branch)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
HF="$DR_VENV/bin/hf"
mkdir -p "$DR_DATA"

echo "==> Wikipedia (ASearcher-Local-Knowledge, ~61GB)"
"$HF" download inclusionAI/ASearcher-Local-Knowledge --repo-type dataset --local-dir "$WIKI_RAW"

echo "==> e5-base-v2 encoder (NOT shipped with the ASearcher release)"
"$HF" download intfloat/e5-base-v2 --local-dir "$WIKI_ASSETS/e5-base-v2"

if [ "${WITH_OPENSCHOLAR:-0}" = "1" ]; then
  echo "==> OpenScholar DataStore V3 (~693GiB)"
  "$HF" download OpenSciLM/OpenScholar-DataStore-V3 --repo-type dataset --local-dir "$OPENSCHOLAR_RAW"
fi

du -sh "$WIKI_RAW" "$WIKI_ASSETS/e5-base-v2" 2>/dev/null || true
