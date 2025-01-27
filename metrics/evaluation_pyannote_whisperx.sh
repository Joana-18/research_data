#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=15GB
#SBATCH --job-name=METRICS
#SBATCH --output=METRICS_%j.out

source /opt/lmod/lmod/init/profile
source /path/to/source/file/source_file_name.sh
module use /pkgs/environment-modules/
module load cuda-11.8

python evaluation_pyannote_whisperx.py