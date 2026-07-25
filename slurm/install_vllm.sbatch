#!/bin/bash
#SBATCH --job-name=dr_install
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_520_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/install_vllm_%j.log
#SBATCH --error=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/install_vllm_%j.log

set -x
export UV_CACHE_DIR=/scratch/bl4363/uv-cache
/home/bl4363/.local/bin/uv pip install --python /scratch/bl4363/uvenvs/deeprubric/bin/python \
  vllm openai aiohttp tqdm transformers
echo "EXIT=$?"
/scratch/bl4363/uvenvs/deeprubric/bin/python -c "import vllm;print('vllm',vllm.__version__)"
