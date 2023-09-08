#!/bin/bash
#SBATCH --partition=leauthaud
#SBATCH --account=leauthaud
#SBATCH --job-name=lensing_mock_challenge
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=7-0:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jolange@ucsc.edu
#SBATCH --output=log/compute_%a.out

cd /data/groups/leauthaud/jolange/Zebu/lux
source init.sh
cd ../stacks/
python compute.py $SLURM_ARRAY_TASK_ID
