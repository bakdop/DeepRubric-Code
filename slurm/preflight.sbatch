#!/bin/bash
#SBATCH --job-name=dr_preflight
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_520_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/preflight_%j.log
#SBATCH --error=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/preflight_%j.log
cd /scratch/bl4363/DeepRubric-Code/data_construction
/scratch/bl4363/uvenvs/deeprubric/bin/python /scratch/bl4363/DeepRubric-Code/tools/preflight_check.py
echo "EXIT=$?"
