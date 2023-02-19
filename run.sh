#!/bin/bash
#SBATCH --account=def-scottmac
#SBATCH --output=output.txt
#SBATCH --gpus-per-node=1
#SBATCH --mem=10G               # memory per node
#SBATCH --time=0-1:00

source makeNumml.sh
#sourse testEnv/bin/activate
python train-gpu.py --num-epoch=50 --lvl=2 --data-set='Data/old_data'    # you can use 'nvidia-smi' for a test
