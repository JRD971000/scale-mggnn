#!/bin/bash
#SBATCH --account=def-scottmac
#SBATCH --output=output.txt
#SBATCH --gpus-per-node=1
#SBATCH --mem=50000               # memory per node
#SBATCH --time=0-1:00

source makeNumml.sh
python train-gpu.py --num-epoch = 2                     # you can use 'nvidia-smi' for a test
