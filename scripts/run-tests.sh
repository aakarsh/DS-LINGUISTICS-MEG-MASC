#!/bin/bash
#SBATCH --partition=dev_cpu
#SBATCH --job-name=pytest-meg-decoder-remaining
#SBATCH --mem=32G
#SBATCH --ntasks=64
#SBATCH --output=slurm_logs/pytest_%A_%a.out
#SBATCH --error=slurm_logs/pytest_%A_%a.err
#SBATCH --time=00:29:00

export PROJECT_ROOT=/home/tu/tu_tu/tu_zxoxo45/TUE-SUMMER-2025/projects/DS-LINGUISTICS-MEG-MASC

source ~/.bashrc
source "${PROJECT_ROOT}/.env.slurm"

uv run python -m spacy download en_core_web_sm

echo "--- Job Details ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "-------------------"

uv run pytest -s -vvv ./test/integration/test_decoding.py::test_run_decoding_voiced

echo "--- Job Finished ---"
