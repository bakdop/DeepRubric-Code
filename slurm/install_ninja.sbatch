#!/bin/bash
#SBATCH --job-name=dr_ninja
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_520_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/install_ninja_%j.log
#SBATCH --error=/scratch/bl4363/DeepRubric-Code/outputs/setup_logs/install_ninja_%j.log
set -x
export UV_CACHE_DIR=/scratch/bl4363/uv-cache
/home/bl4363/.local/bin/uv pip install --python /scratch/bl4363/uvenvs/deeprubric/bin/python ninja
/scratch/bl4363/uvenvs/deeprubric/bin/python -c "import shutil;print('ninja at:', shutil.which('ninja'))"
/scratch/bl4363/uvenvs/deeprubric/bin/ninja --version
