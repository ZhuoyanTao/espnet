#!/usr/bin/env bash
#SBATCH --job-name=espnet_train
#SBATCH --output=logs/train_model.out
#SBATCH --error=logs/train_model.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=terryt@usc.edu

cd /ocean/projects/cis210027p/ttao3/espnet/egs/an4/asr1

# Run RNN language model training
./run.sh --stage 3 --stop_stage 3

# # Run ASR model training
# ./run.sh --stage 4 --stop_stage 4
