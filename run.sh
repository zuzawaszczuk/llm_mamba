#!/bin/bash
#SBATCH --job-name=mamba        # nazwa zadania
#SBATCH --output=job_output%j.txt    # plik wyjściowy (stdout)
#SBATCH --error=job_error%j.txt      # plik błędów (stderr)
#SBATCH --partition=gpu          # nazwa partycji
#SBATCH --nodes=1                  # liczba węzłów
#SBATCH --ntasks=1                 # liczba zadań
#SBATCH --cpus-per-task=24       # liczba CPU na zadanie
#SBATCH --mem=60G                   # pamięć RAM
#SBATCH --time=10:00:00            # maksymalny czas wykonania
#SBATCH --gres=gpu:nvidia-96G:7               # liczba GPU

# --- Komendy do wykonania ---
START=$(date +%s)

echo "Hello SLURM"
hostname
nproc
sleep 60

module load uv
cd /scratch/zwaszczu/llm_mamba
uv sync --reinstall

source .venv/bin/activate
OMP_NUM_THREADS=1 torchrun --nproc-per-node=7 main.py

END=$(date +%s)
ELAPSED=$((END - START))

printf "Elapsed: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) \
    $(((ELAPSED%3600)/60)) \
    $((ELAPSED%60))