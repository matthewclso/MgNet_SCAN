#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=v100:8

export XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
submit_host=${SLURM_SUBMIT_HOST}
port=8888

module load cuda

cd /ibex/scratch/somc/MgNet_SCAN
source /home/${USER}/.bashrc
conda activate env

echo -e "
ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa
"

jupyter lab --no-browser --port=${port} --ip=${node} --ServerApp.token=''
