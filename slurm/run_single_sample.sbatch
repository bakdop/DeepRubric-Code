#!/bin/bash
#SBATCH --job-name=dr_sample
#SBATCH --partition=h200_tandon
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/run_sample_%j.log
#SBATCH --error=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/run_sample_%j.log

# End-to-end single-sample run of the DeepRubric question+rubric synthesis.
#   GPU 0,1 -> vLLM serving Qwen3.5-122B-A10B (tp=2), OpenAI-compatible on :8008
#   GPU 0   -> e5/FAISS retriever on :8888, sharing vLLM's spare memory.
# Only 2 GPUs are requested, so the retriever piggybacks on GPU 0:
# e5-base is ~0.3GB and vLLM is capped at 0.92 utilisation of 141GB, leaving ~11GB there.
# It MUST start after vLLM has finished reserving its KV pool, otherwise vLLM's
# startup profiling would hand that memory out twice.
# Runs depth=1 first (fast plumbing check + a complete artifact), then the
# shipped default depth=3 on the same allocation.

set -x
REPO=/scratch/bl4363/DeepRubric-Code
SUB=/scratch/bl4363/data/ASearcher-subset
PY=/scratch/bl4363/uvenvs/deeprubric/bin/python
LOGDIR=$REPO/outputs/setup_logs

# Qwen3.5's GDN linear-attention kernels JIT-compile at load and look up `ninja`
# on PATH. We invoke python by absolute path (never `activate`), so the venv's
# bin/ is not on PATH and vLLM dies with FileNotFoundError: 'ninja' even though
# the package is installed. Put it on PATH explicitly.
export PATH="$(dirname "$PY"):$PATH"

export HF_HOME=/scratch/bl4363/hf_cache
# torch.compile/Triton emit thousands of tiny files. Their default home is
# ~/.cache, and /home here is capped at 30k inodes (91% used) -> the build dies
# with [Errno 122] Disk quota exceeded even though 46GB is free. Keep every
# cache on /scratch, which is capped at 5M inodes.
export XDG_CACHE_HOME=/scratch/bl4363/.cache
export VLLM_CACHE_ROOT=/scratch/bl4363/.cache/vllm
export TRITON_CACHE_DIR=/scratch/bl4363/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/bl4363/torchinductor_cache
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export OPENAI_BASE_URL=http://localhost:8008/v1
export OPENAI_API_KEY=EMPTY
export MODEL_NAME=qwen3.5-122b
export TOKENIZERS_PARALLELISM=false

mkdir -p $LOGDIR

# ---------------------------------------------------------------- vLLM (GPU 0,1)
CUDA_VISIBLE_DEVICES=0,1 $PY -m vllm.entrypoints.openai.api_server \
  --model /scratch/bl4363/models/Qwen3.5-122B-A10B \
  --served-model-name $MODEL_NAME \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --port 8008 \
  > $LOGDIR/vllm_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!

cleanup() { kill $VLLM_PID $RETR_PID 2>/dev/null; }
trap cleanup EXIT

# ------------------------------------------------------------------- wait vLLM
echo "waiting for vLLM ..."
for i in $(seq 1 180); do
  curl -sf http://localhost:8008/v1/models > /dev/null && { echo "vLLM UP after $((i*10))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM DIED - see $LOGDIR/vllm_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
curl -s http://localhost:8008/v1/models | head -c 400; echo
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# ------------------------------- retriever (GPU 0, after vLLM reserved its pool)
CUDA_VISIBLE_DEVICES=0 $PY $REPO/retrievers/wiki/code/local_retrieval_server.py \
  --index_path $SUB/e5.index/e5_Flat.index \
  --corpus_path $SUB/wiki_corpus.jsonl \
  --pages_path $SUB/wiki_webpages.jsonl \
  --topk 5 \
  --retriever_name e5 \
  --retriever_model $SUB/e5-base-v2 \
  --port 8888 \
  > $LOGDIR/retriever_$SLURM_JOB_ID.log 2>&1 &
RETR_PID=$!

echo "waiting for retriever ..."
for i in $(seq 1 180); do
  curl -sf -X POST http://localhost:8888/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["test"],"topk":1,"return_scores":true}' > /dev/null && { echo "retriever UP after $((i*10))s"; break; }
  kill -0 $RETR_PID 2>/dev/null || { echo "retriever DIED - see $LOGDIR/retriever_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done

# smoke-test the retriever so we can see a real hit in the log
curl -s -X POST http://localhost:8888/retrieve -H 'Content-Type: application/json' \
  -d '{"queries":["Alain Connes Fields Medal"],"topk":2,"return_scores":true}' | head -c 800; echo

# ------------------------------------------------------- generate the samples
cd $REPO/data_construction

for D in 1 3; do
  OUT=$REPO/outputs/wiki_tree_depth${D}
  mkdir -p $OUT
  echo "=============== GENERATING depth=$D ==============="
  date
  $PY recursive_qa_agent_v4.py \
    --pages $SUB/wiki_webpages.jsonl \
    --links $SUB/wikilinks.json \
    --save $OUT \
    --retriever-url http://localhost:8888/retrieve \
    --qa-model $MODEL_NAME \
    --extract-model $MODEL_NAME \
    --max_depth $D \
    --n 1
  echo "GEN_EXIT_depth${D}=$?"
  date
  ls -la $OUT/
done
