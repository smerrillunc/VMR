#!/bin/bash
#SBATCH --job-name=flow_extraction
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
cd /work/users/s/m/smerrill/preprocess

# Run the Python command
conda run -n video_features python 	flowProcessor.py \
--video_file_path=/work/users/s/m/smerrill/Youtube8m/video_paths.txt \
--save_path=/work/users/s/m/smerrill/Youtube8m