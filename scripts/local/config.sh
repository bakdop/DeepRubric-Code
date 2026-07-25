#!/usr/bin/env bash
# Shared configuration for a standalone multi-GPU box (no slurm).
# Override any of these by exporting them before sourcing / running the scripts.

# ---- where everything lives -------------------------------------------------
export DR_ROOT="${DR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DR_DATA="${DR_DATA:-$DR_ROOT/../deeprubric-data}"
export DR_VENV="${DR_VENV:-$DR_ROOT/../deeprubric-venv}"
export PY="${PY:-$DR_VENV/bin/python}"
# The scripts call python by absolute path rather than activating the venv, so
# put the venv's bin/ on PATH explicitly: vLLM shells out to `ninja` (and other
# build tools) by name when it JIT-compiles Qwen3.5's GDN attention kernels.
export PATH="$DR_VENV/bin:$PATH"

export WIKI_RAW="${WIKI_RAW:-$DR_DATA/ASearcher-Local-Knowledge}"
export OPENSCHOLAR_RAW="${OPENSCHOLAR_RAW:-$DR_DATA/OpenScholar-DataStore-V3}"
export WIKI_ASSETS="${WIKI_ASSETS:-$DR_DATA/wiki-assets}"   # subset + index + encoder

# ---- corpus subsetting ------------------------------------------------------
# KEEP_ONE_IN=1  -> full Wikipedia (~26M passages, Flat index ~80GB, needs ~100GB RAM to serve)
# KEEP_ONE_IN=20 -> ~5% sample (~1.3M passages, ~4GB index)  <-- good for learning the pipeline
export KEEP_ONE_IN="${KEEP_ONE_IN:-20}"
export MAX_PASSAGES="${MAX_PASSAGES:-2000000}"

# ---- models -----------------------------------------------------------------
# 8x A800-80G = 640GB, so a 234GB bf16 122B fits comfortably at tp=2 or tp=4.
# A800 is Ampere (sm80): bf16 only, NO fp8.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-122B-A10B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b}"
export TP_SIZE="${TP_SIZE:-4}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# GPUs: vLLM takes the first TP_SIZE, the retriever takes RETRIEVER_GPU.
export VLLM_GPUS="${VLLM_GPUS:-$(seq -s, 0 $((TP_SIZE-1)))}"
export RETRIEVER_GPU="${RETRIEVER_GPU:-7}"
export INDEX_GPUS="${INDEX_GPUS:-0,1,2,3,4,5,6,7}"

# ---- mirrors (China networks) ----------------------------------------------
# HuggingFace: hf-mirror.com is the usual community mirror. If your site runs its
# own HF mirror, point HF_ENDPOINT at it instead. Unset it to use huggingface.co.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# PyPI: Tsinghua TUNA.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-$PIP_INDEX_URL}"
# hf_transfer parallelises downloads and is a big win on a direct (non-proxied)
# link. It does NOT reliably honour HTTP(S)_PROXY, so turn it off if you must
# go through a proxy.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# huggingface_hub defaults both timeouts to 10s, which a cross-border hop to a
# mirror routinely exceeds -> httpx.ReadTimeout before anything downloads.
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
# hf_hub >=1.24 fetches an agent-detection registry on top of the real request;
# it is unrelated to the transfer but still blocks and can time out.
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"

# ---- endpoints --------------------------------------------------------------
export VLLM_PORT="${VLLM_PORT:-8008}"
export RETRIEVER_PORT="${RETRIEVER_PORT:-8888}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:$VLLM_PORT/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export WIKI_RETRIEVER_URL="${WIKI_RETRIEVER_URL:-http://localhost:$RETRIEVER_PORT/retrieve}"

# ---- generation -------------------------------------------------------------
export N_SAMPLES="${N_SAMPLES:-1}"
export MAX_DEPTH="${MAX_DEPTH:-3}"
export OUT_DIR="${OUT_DIR:-$DR_ROOT/outputs}"

export LOGDIR="${LOGDIR:-$OUT_DIR/logs}"
mkdir -p "$LOGDIR"
