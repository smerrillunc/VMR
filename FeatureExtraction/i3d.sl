#!/bin/bash

# Paths
VIDEO_LIST_DIR="/work/users/s/m/smerrill/Youtube8m"
LOG_DIR="/work/users/s/m/smerrill/log"
SCRIPT_DIR="/work/users/s/m/smerrill/video_features"

# Loop through each segmented video path file
for i in {0..9}; do
  VIDEO_FILE="${VIDEO_LIST_DIR}/video_paths_${i}.txt"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=i3d_extraction_${i}
#SBATCH --time=1-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=${LOG_DIR}/download_job_%A.out
#SBATCH --error=${LOG_DIR}/download_job_%A.err

module load anaconda
conda activate video_features

cd ${SCRIPT_DIR}

conda run -n video_features python main.py \\
  feature_type="i3d" \\
  extraction_fps=32 \\
  stack_size=32 \\
  step_size=32 \\
  on_extraction="save_numpy" \\
  output_path="${VIDEO_LIST_DIR}" \\
  file_with_video_paths="${VIDEO_FILE}"
EOF

done
