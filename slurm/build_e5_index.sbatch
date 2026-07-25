#!/bin/bash
#SBATCH --job-name=dr_e5_index
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_520_general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/e5_index_%j.log
#SBATCH --error=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/e5_index_%j.log

set -x
export HF_HOME=/scratch/bl4363/hf_cache
SUB=/scratch/bl4363/data/ASearcher-subset
/scratch/bl4363/uvenvs/deeprubric/bin/python \
  /scratch/bl4363/DeepRubric-Code/tools/build_e5_index.py \
  --corpus $SUB/wiki_corpus.jsonl \
  --model $SUB/e5-base-v2 \
  --out-index $SUB/e5.index/e5_Flat.index \
  --batch-size 1024
echo "EXIT=$?"
ls -la $SUB/e5.index/
