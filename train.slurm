#!/bin/bash
#SBATCH --job-name=coinrun-wm
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:h100-80:1
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis250234p/jyang45/open-oasis/runs/slurm_%j.out
#SBATCH --error=/ocean/projects/cis250234p/jyang45/open-oasis/runs/slurm_%j.err

PYTHON=/ocean/projects/cis250234p/jyang45/conda/envs/open-oasis/bin/python
TORCHRUN=/ocean/projects/cis250234p/jyang45/conda/envs/open-oasis/bin/torchrun
WORKDIR=/ocean/projects/cis250234p/jyang45/open-oasis
NGPU=${SLURM_GPUS_ON_NODE:-1}

cd "$WORKDIR"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "GPUs: $NGPU — $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"

if [ "$NGPU" -gt 1 ]; then
    $TORCHRUN --nproc_per_node=$NGPU train_coinrun.py "$@"
else
    $PYTHON train_coinrun.py "$@"
fi
