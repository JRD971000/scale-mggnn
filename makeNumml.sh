module purge
module load cuda/11.7
module load python/3.8
module load scipy-stack

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

cp -r testEnv $SLURM_TMPDIR/env
sourse $SLURM_TMPDIR/env/bin/activate

cp -r numml $SLURM_TMPDIR/numml
pip3 install $SLURM_TMPDIR/numml/.
