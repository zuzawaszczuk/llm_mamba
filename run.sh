#!/bin/bash
#SBATCH --job-name=mamba        # nazwa zadania
#SBATCH --output=job_output%j.txt    # plik wyjściowy (stdout)
#SBATCH --error=job_error%j.txt      # plik błędów (stderr)
#SBATCH --partition=gpu          # nazwa partycji
#SBATCH --cpus-per-task=12       # liczba CPU na zadanie
#SBATCH --mem=40G                   # pamięć RAM
#SBATCH --time=10:00:00            # maksymalny czas wykonania
#SBATCH --gres=gpu:nvidia-96G:6             # liczba GPU

# --- Komendy do wykonania ---
START=$(date +%s)

echo "Hello SLURM"
hostname
nproc
sleep 60

module load uv
cd /scratch/zwaszczu/llm_mamba
uv sync --reinstall

# choose model openai-community/gpt2  state-spaces/mamba-130m-hf

source .venv/bin/activate
OMP_NUM_THREADS=1 torchrun --nproc-per-node=6 main.py --model "state-spaces/mamba-130m-hf" --rank 16 --epoch 20 --ddp_setup True

END=$(date +%s)
ELAPSED=$((END - START))

printf "Elapsed: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) \
    $(((ELAPSED%3600)/60)) \
    $((ELAPSED%60))