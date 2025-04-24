#!/bin/bash
#SBATCH --job-name=flow_extraction
#SBATCH --time=3-12:00:00             # Adjust based on expected runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=/work/users/s/m/smerrill/log/download_job_%A_%a.out
#SBATCH --error=/work/users/s/m/smerrill/log/download_job_%A_%a.err
#SBATCH --array=0-9

# Load necessary modules (adjust as needed for your environment)
module load anaconda
conda activate video_features

# Change to the working directory
cd /work/users/s/m/smerrill/preprocess

# Construct the input file name based on the SLURM_ARRAY_TASK_ID
VIDEO_FILE_PATH="/work/users/s/m/smerrill/Youtube8m/video_paths_${SLURM_ARRAY_TASK_ID}.txt"

# Run the Python command
conda run -n video_features python flowProcessor.py \
--video_file_path="$VIDEO_FILE_PATH" \
--save_path=/work/users/s/m/smerrill/Youtube8m