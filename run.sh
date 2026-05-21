#!/bin/bash
#SBATCH --job-name=submit
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdyplomancipw2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# --- Komendy do wykonania ---
START=$(date +%s)

echo "Hello SLURM"
hostname
nproc
sleep 60

cd /net/tscratch/people/plgzwaszczuk/llm_mamba
uv sync
module load cuda/12.8

source .venv/bin/activate
OMP_NUM_THREADS=1 torchrun --nproc-per-node=1 main.py

END=$(date +%s)
ELAPSED=$((END - START))

printf "Elapsed: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) \
    $(((ELAPSED%3600)/60)) \
    $((ELAPSED%60))