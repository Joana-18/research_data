#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=m2
#SBATCH --gres=gpu:1
#SBATCH -p t4v2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joana.amorim@dal.ca
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=15GB
#SBATCH --job-name=VOX_analysis
#SBATCH --output=VOX_analysis_%j.out

source /opt/lmod/lmod/init/profile
source /path/to/source/file/source_file_name.sh
module use /pkgs/environment-modules/
module load cuda-11.0

python dataset_analysis.py