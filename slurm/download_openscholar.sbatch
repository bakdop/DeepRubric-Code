#!/bin/bash
#SBATCH --job-name=dr_dl_oschol
#SBATCH --partition=cs
#SBATCH --account=torch_pr_520_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/download_openscholar_%j.log
#SBATCH --error=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/download_openscholar_%j.log

set -x
export HF_HOME=/scratch/bl4363/hf_cache
/scratch/bl4363/uvenvs/rgym/bin/hf download OpenSciLM/OpenScholar-DataStore-V3 \
  --repo-type dataset \
  --local-dir /scratch/bl4363/data/OpenScholar-DataStore-V3
echo "EXIT=$?"
du -sh /scratch/bl4363/data/OpenScholar-DataStore-V3
