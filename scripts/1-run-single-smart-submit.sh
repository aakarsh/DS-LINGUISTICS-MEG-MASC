#!/bin/bash
# filepath: scripts/submit-remaining.sh

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoxo45/TUE-SUMMER-2025/projects/DS-LINGUISTICS-MEG-MASC

SUBJECT_IDS=(
    # "01" "02" 
    "03" "04" 
    #"05" "06" "07" "08" "09" "10" "11"
)

# Check which subjects are already processed
PROCESSED=()
REMAINING=()

for i in "${!SUBJECT_IDS[@]}"; do
    SUBJECT=${SUBJECT_IDS[$i]}
    OUTPUT_FILE="${PROJECT_ROOT}/output/${SUBJECT}_decoding_results.csv"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Subject $SUBJECT (index $i) - Already processed"
        PROCESSED+=($i)
    else
        echo "Subject $SUBJECT (index $i) - Needs processing"
        REMAINING+=($i)
    fi
done

if [ ${#REMAINING[@]} -eq 0 ]; then
    echo "All subjects already processed!"
    exit 0
fi

# Convert remaining indices to array format
if [ ${#REMAINING[@]} -eq 1 ]; then
    ARRAY_SPEC="${REMAINING[0]}"
else
    # Join array elements with commas
    ARRAY_SPEC=$(IFS=,; echo "${REMAINING[*]}")
fi

echo "Submitting jobs for remaining subjects: indices [$ARRAY_SPEC]"

# Create temporary SLURM script
cat > temp_remaining.sh << EOF
#!/bin/bash

#SBATCH --job-name=meg-decoder-remaining
#SBATCH --ntasks=64
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --partition=dev_cpu
#SBATCH --time=00:29:00
#SBATCH --array=${ARRAY_SPEC}%1

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoxo45/TUE-SUMMER-2025/projects/DS-LINGUISTICS-MEG-MASC

SUBJECT_IDS=(
    "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11"
)

CURRENT_SUBJECT=\${SUBJECT_IDS[\$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
source "\${PROJECT_ROOT}/.env.slurm"

echo "Starting Slurm job \$SLURM_ARRAY_TASK_ID for subject: \$CURRENT_SUBJECT"
uv run python -m spacy download en_core_web_sm

uv run python \${PROJECT_ROOT}/scripts/1-run-single-subject.py --subject-id \$CURRENT_SUBJECT --max-workers 128
EOF

sbatch temp_remaining.sh
rm temp_remaining.sh
