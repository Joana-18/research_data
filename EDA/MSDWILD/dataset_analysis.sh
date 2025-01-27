#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=m
#SBATCH --gres=gpu:4
#SBATCH -p a40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joana.amorim@dal.ca
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=15GB
#SBATCH --job-name=analysis_MSD
#SBATCH --output=analysis_%j.out

source /opt/lmod/lmod/init/profile
source /path/to/source/file/source_file_name.sh
module use /pkgs/environment-modules/
module load cuda-11.8

python dataset_analysis.py