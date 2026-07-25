#!/bin/bash
#SBATCH --job-name=dr_dl_wiki
#SBATCH --partition=cs
#SBATCH --account=torch_pr_520_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/download_wiki_%j.log
#SBATCH --error=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/download_wiki_%j.log

set -x
export HF_HOME=/scratch/bl4363/hf_cache
/scratch/bl4363/uvenvs/rgym/bin/hf download inclusionAI/ASearcher-Local-Knowledge \
  --repo-type dataset \
  --local-dir /scratch/bl4363/data/ASearcher-Local-Knowledge
echo "EXIT=$?"
du -sh /scratch/bl4363/data/ASearcher-Local-Knowledge
