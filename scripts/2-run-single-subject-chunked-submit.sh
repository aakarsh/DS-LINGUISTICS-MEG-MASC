#!/bin/bash

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoxo45/TUE-SUMMER-2025/projects/DS-LINGUISTICS-MEG-MASC

SUBJECT_IDS=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11")
FEATURE_CHUNKS=(
    "phonation"
    "manner"
    "place" 
    "frontback"
    "roundness"
    "centrality"
    "voiced"
    "part_of_speach"
)

# Function to submit chunked job
submit_chunked_job() {
    local subject=$1
    local chunk_index=$2
    local features="${FEATURE_CHUNKS[$chunk_index]}"
    local job_name="meg-decoder-${subject}-chunk${chunk_index}"
    
    echo "Submitting chunk $chunk_index for subject $subject: features [$features]"
    
    cat > "temp_${subject}_chunk${chunk_index}.sh" << EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --ntasks=64
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/${subject}_chunk${chunk_index}_%j.out
#SBATCH --error=slurm_logs/${subject}_chunk${chunk_index}_%j.err
#SBATCH --partition=dev_cpu_il
#SBATCH --time=00:29:00

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoxo45/TUE-SUMMER-2025/projects/DS-LINGUISTICS-MEG-MASC

source ~/.bashrc
source "\${PROJECT_ROOT}/.env.slurm"

echo "Starting chunk $chunk_index for subject: $subject"
uv run python -m spacy download en_core_web_sm

# Save results with chunk suffix
OUTPUT_FILE="\${PROJECT_ROOT}/output/${subject}_chunk${chunk_index}_decoding_results.csv"

uv run python \${PROJECT_ROOT}/scripts/1-run-single-subject.py \\
    --subject-id $subject \\
    --max-workers 64 \\
    --feature-prefix-list $features \\
    --output-file "\$OUTPUT_FILE"
EOF

    sbatch "temp_${subject}_chunk${chunk_index}.sh"
    rm "temp_${subject}_chunk${chunk_index}.sh"
}

# Smart queue management for chunks
echo "Starting chunked queue manager..."

for subject in "${SUBJECT_IDS[@]}"; do
    for chunk_idx in "${!FEATURE_CHUNKS[@]}"; do
        # Wait for queue capacity
        while [ $(squeue -u $USER --noheader | wc -l) -ge 1 ]; do
            echo "Queue full, waiting..."
            sleep 30
        done
        
        output_file="${PROJECT_ROOT}/output/${subject}_chunk${chunk_idx}_decoding_results.csv"
        if [ ! -f "$output_file" ]; then
            submit_chunked_job "$subject" "$chunk_idx"
            sleep 5  # Small delay between submissions
        else
            echo "Chunk $chunk_idx for subject $subject already processed"
        fi
    done
done

echo "All chunks submitted!"