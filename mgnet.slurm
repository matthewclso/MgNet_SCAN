#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=v100:8
#SBATCH --mem=64G

module load cuda

source /home/${USER}/.bashrc
conda activate env

python /ibex/scratch/somc/mgnet_tf/mgnet.py --dataset cifar10 --iterations 2,2,2,2 --u-channels 256,256,256,256 --f-channels 256,256,256,256 -batch-size 1024 --epochs 100 --lr .01 --graph True