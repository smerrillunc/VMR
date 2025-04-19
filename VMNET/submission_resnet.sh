#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20g
#SBATCH -t 2-
#SBATCH -p a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=smerrill@unc.edu
#SBATCH --output=/proj/mcavoy_lab/logs/job_%j.out 
#SBATCH --error=/proj/mcavoy_lab/logs/job_%j.err

module purge
module load python/3.9.6
source /proj/mcavoy_lab/VMNET/vmnet_env/bin/activate
python train.py --video_dir /work/users/s/m/smerrill/Youtube8m/resnet/resnet101 \
--audio_dir  /work/users/s/m/smerrill/Youtube8m/vggish \
--train_csv_path /work/users/s/m/smerrill/Youtube8m/train.csv \
--summaries_dir /proj/mcavoy_lab/Youtube8m/models/resnet \
--vid_dim 2048 \
--aud_dim 128

