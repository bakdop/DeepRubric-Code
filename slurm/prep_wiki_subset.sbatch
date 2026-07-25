#!/bin/bash
#SBATCH --job-name=dr_prep_wiki
#SBATCH --partition=cs
#SBATCH --account=torch_pr_520_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/prep_wiki_%j.log
#SBATCH --error=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/prep_wiki_%j.log

set -x
export HF_HOME=/scratch/bl4363/hf_cache
PY=/scratch/bl4363/uvenvs/deeprubric/bin/python

# server-side deps (the vendored FastAPI retriever needs these)
/home/bl4363/.local/bin/uv pip install --python $PY faiss-cpu datasets fastapi uvicorn || exit 1

# e5 encoder is NOT part of the ASearcher release; fetch it separately
/scratch/bl4363/uvenvs/rgym/bin/hf download intfloat/e5-base-v2 \
  --local-dir /scratch/bl4363/data/ASearcher-subset/e5-base-v2 || exit 1

$PY /scratch/bl4363/DeepRubric-Code/tools/prep_wiki_subset.py \
  --src /scratch/bl4363/data/ASearcher-Local-Knowledge \
  --out /scratch/bl4363/data/ASearcher-subset \
  --keep-one-in 20 \
  --max-passages 2000000
echo "EXIT=$?"
du -sh /scratch/bl4363/data/ASearcher-subset
