#!/bin/bash
#SBATCH --job-name=resnet_extraction
#SBATCH --time=3-12:00:00             # Adjust based on expected runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G                   # Adjust based on memory needs
#SBATCH --output=/work/users/s/m/smerrill/log/download_job_%A.out   # STDOUT file
#SBATCH --error=/work/users/s/m/smerrill/log/download_job_%A.err    # STDERR file

# Load necessary modules (adjust as needed for your environment)
module load anaconda
conda activate video_features

# Change to the working directory
cd /work/users/s/m/smerrill/video_features

# Run the Python command
conda run -n video_features python main.py \
feature_type="resnet" \
model_name="resnet101" \
extraction_fps=1 \
on_extraction="save_numpy" \
output_path="./../Youtube8m" \
file_with_video_paths="/work/users/s/m/smerrill/Youtube8m/video_paths.txt"