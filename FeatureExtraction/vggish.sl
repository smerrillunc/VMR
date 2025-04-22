#!/bin/bash

# Paths
AUDIO_LIST_DIR="/work/users/s/m/smerrill/Youtube8m"
LOG_DIR="/work/users/s/m/smerrill/log"
SCRIPT_DIR="/work/users/s/m/smerrill/preprocess"

# Loop through each segmented audio path file
for i in {0..9}; do
  AUDIO_FILE="${AUDIO_LIST_DIR}/audio_paths_${i}.txt"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vggish_extraction_${i}
#SBATCH --time=4-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=${LOG_DIR}/download_job_%A.out
#SBATCH --error=${LOG_DIR}/download_job_%A.err

module load anaconda
conda activate video_features

cd ${SCRIPT_DIR}

conda run -n video_features python vggishProcessor.py \\
  --audio_file_path="${AUDIO_FILE}" \\
  --save_path="${AUDIO_LIST_DIR}"
EOF

done
