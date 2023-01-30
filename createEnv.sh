#!/bin/bash

[ -d testEnv ] && rm -rf testEnv && echo Removing existing venv...

module load cuda/11.7
module load python/3.8

python3 -m venv testEnv
source testEnv/bin/activate

pip install --upgrade pip
pip install wheel
pip install -r reqs.txt
pip install pygmsh
pip install gmsh-sdk

#pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
#pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
#pip install torch-geometric

echo
echo ---
echo Created venv \"testEnv\".  Activate by running \"source testEnv/bin/activate\".
