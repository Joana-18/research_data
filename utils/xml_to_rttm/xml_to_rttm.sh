#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p t4v2
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=15GB
#SBATCH --job-name=RTTM_CONVERSION
#SBATCH --output=RTTM_%j.out

source /opt/lmod/lmod/init/profile
source /path/to/source/file/source_file_name.sh
module use /pkgs/environment-modules/
module load cuda-11.8

python xml_to_rttm.py