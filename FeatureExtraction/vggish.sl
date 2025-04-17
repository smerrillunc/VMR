#!/bin/bash
#SBATCH --job-name=clip_extraction
#SBATCH --time=2-04:00:00             # Adjust based on expected runtime
#SBATCH --partition=standard        # Adjust partition as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # Adjust based on your script's needs
#SBATCH --mem=16G                   # Adjust based on memory needs
#SBATCH --gres=gpu:1                # Request GPU if needed
#SBATCH --output=/work/users/s/m/smerrill/log/download_job_%A.out   # STDOUT file
#SBATCH --error=/work/users/s/m/smerrill/log/download_job_%A.err    # STDERR file

# Load necessary modules (adjust as needed for your environment)
module load anaconda
conda activate video_features

# Change to the working directory
cd /work/users/s/m/smerrill/preprocess

# Run the Python command
conda run -n video_features python vggishProcessor.py \
--audio_file_path=/work/users/s/m/smerrill/Youtube8m/audio_files.txt \
--save_path=/work/users/s/m/smerrill/Youtube8m