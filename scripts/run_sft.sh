#!/usr/bin/env bash
set -euo pipefail

#============================== Slurm ==========================================
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodelist=dgx-H100-02
#SBATCH --job-name=sft-train
#SBATCH --output=jobs/job.%j.out
#SBATCH --error=jobs/job.%j.err

#============================== Runtime ========================================

# Ensure output directory exists
mkdir -p jobs

# Go to project root (adjust if needed)
cd "$SLURM_SUBMIT_DIR"

# Activate your Python environment (adjust to your setup)
# source .venv/bin/activate
# or: source /path/to/conda/bin/activate your_env

# Optional: set Hugging Face cache to a fast disk
# export HF_HOME=/path/to/hf_cache

python train_sft.py
