#!/usr/bin/env bash
# Start vLLM + the local retriever, wait until both answer, then stay in foreground.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "vLLM  : $MODEL_PATH  tp=$TP_SIZE  gpus=$VLLM_GPUS  port=$VLLM_PORT"
echo "retr. : gpu=$RETRIEVER_GPU  port=$RETRIEVER_PORT"

CUDA_VISIBLE_DEVICES="$VLLM_GPUS" "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --port "$VLLM_PORT" > "$LOGDIR/vllm.log" 2>&1 &
VLLM_PID=$!

CUDA_VISIBLE_DEVICES="$RETRIEVER_GPU" "$PY" "$DR_ROOT/retrievers/wiki/code/local_retrieval_server.py" \
  --index_path "$WIKI_ASSETS/e5.index/e5_Flat.index" \
  --corpus_path "$WIKI_ASSETS/wiki_corpus.jsonl" \
  --pages_path "$WIKI_ASSETS/wiki_webpages.jsonl" \
  --topk 5 --retriever_name e5 \
  --retriever_model "$WIKI_ASSETS/e5-base-v2" \
  --port "$RETRIEVER_PORT" > "$LOGDIR/retriever.log" 2>&1 &
RETR_PID=$!

trap 'kill $VLLM_PID $RETR_PID 2>/dev/null' EXIT

echo "waiting for vLLM (weights load takes several minutes) ..."
for i in $(seq 1 240); do
  curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null && { echo "vLLM UP"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM DIED - see $LOGDIR/vllm.log"; exit 1; }
  sleep 10
done

echo "waiting for retriever (loads corpus + index) ..."
for i in $(seq 1 240); do
  curl -sf -X POST "http://localhost:$RETRIEVER_PORT/retrieve" -H 'Content-Type: application/json' \
    -d '{"queries":["test"],"topk":1,"return_scores":true}' >/dev/null && { echo "retriever UP"; break; }
  kill -0 $RETR_PID 2>/dev/null || { echo "retriever DIED - see $LOGDIR/retriever.log"; exit 1; }
  sleep 10
done

echo "both services ready. run scripts/local/05_generate.sh in another shell."
wait
