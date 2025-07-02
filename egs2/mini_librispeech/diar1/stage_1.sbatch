#!/usr/bin/env bash
#SBATCH --job-name=espnet_eda
#SBATCH --output=logs/run_eda_%j.out
#SBATCH --error=logs/run_eda_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-gpu=5
cd /ocean/projects/cis210027p/ttao3/espnet/egs2/mini_librispeech/diar1
srun ./run.sh --stage 1 --stop_stage 1
