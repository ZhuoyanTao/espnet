#!/usr/bin/env bash
#
# Reserve 1 GPU for up to 2 days by sleeping
#
#SBATCH --job-name=gpu_reserve
#SBATCH --output=gpu_reserve_%j.out
#SBATCH --error=gpu_reserve_%j.err

#SBATCH --export=PATH
#SBATCH -p ghx4
#SBATCH --gres=gpu:1
#SBATCH -c 32                         # 32 CPU cores
#SBATCH --mem=240000M                 # 240 GB RAM
#SBATCH --account=bbjs-dtai-gh
#SBATCH --time=2-0:00:00              # 2 days walltime

# Duration to sleep (in seconds). Override by passing a value when you sbatch:
#   sbatch reserve_gpu.sh 3600
DURATION=${1:-172800}

echo "[$(date)] Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST: holding GPU for $DURATION seconds"
sleep "$DURATION"
echo "[$(date)] Job $SLURM_JOB_ID: done"
