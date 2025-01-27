#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=m2
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=15GB
#SBATCH --job-name=ASR_DIA
#SBATCH --output=ASR_DIA_%j.out

source /opt/lmod/lmod/init/profile
source /path/to/source/file/source_file_name.sh
module use /pkgs/environment-modules/
module load cuda-11.8

python asr_diarization.py