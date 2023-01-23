#!/bin/bash

[ -d testEnv ] && rm -rf testEnv && echo Removing existing venv...

module load cuda/11.7
module load python/3.8
python3 -m venv testEnv
source testEnv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -r reqs.txt

echo
echo ---
echo Created venv \"testEnv\".  Activate by running \"source testEnv/bin/activate\".
