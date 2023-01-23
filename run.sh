#!/bin/bash
#SBATCH --account=def-scottmac
#SBATCH --gpus-per-node=1
#SBATCH --mem=50000               # memory per node
#SBATCH --time=0-1:00

source testEnv/bin/activate
python train-gpu.py --num-epoch = 2 &> output.txt                      # you can use 'nvidia-smi' for a test
