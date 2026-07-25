#!/usr/bin/env bash
# Create the python environment. Needs ~30GB disk for wheels + CUDA libs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 "$DR_VENV"
uv pip install --python "$DR_VENV/bin/python" \
  vllm openai aiohttp tqdm transformers datasets fastapi uvicorn faiss-cpu "huggingface_hub[cli]"

"$DR_VENV/bin/python" - <<'PY'
import vllm, torch, transformers, faiss
print("vllm        ", vllm.__version__)
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("faiss       ", faiss.__version__)
print("gpus        ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name} {p.total_memory/2**30:.0f}GiB sm{p.major}{p.minor}")
# Qwen3.5 is a recent architecture; confirm this vLLM build knows it.
from vllm.model_executor.models.registry import ModelRegistry
arch = "Qwen3_5MoeForConditionalGeneration"
print(f"{arch} supported:", arch in ModelRegistry.get_supported_archs())
PY
echo "env ready at $DR_VENV"
