#!/bin/bash

#SBATCH --job-name=meg-decoder-single-subject
#SBATCH --ntasks=64
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --partition=dev_cpu_il
#SBATCH --time=00:29:00
#SBATCH --array=4-7%1

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoxo45/TUE-SUMMER-2025/projects/DS-LINGUISTICS-MEG-MASC

SUBJECT_IDS=(
    "01"
    "02"
    "03"
    "04"
    "05"
    "06"
    "07"
    "08"
    "09"
    "10"
    "11"
)

CURRENT_SUBJECT=${SUBJECT_IDS[$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
source "${PROJECT_ROOT}/.env.slurm"

echo "Starting Slurm job $SLURM_ARRAY_TASK_ID for subject: $CURRENT_SUBJECT"

uv run python ${PROJECT_ROOT}/scripts/1-run-single-subject.py  --subject-id $CURRENT_SUBJECT --max-workers 128
