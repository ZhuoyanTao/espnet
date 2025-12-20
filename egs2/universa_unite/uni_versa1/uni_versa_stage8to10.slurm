#!/bin/bash
#SBATCH --job-name=uni_versa_s8_10
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=uni_versa_s8_10_%j.log
#SBATCH --error=uni_versa_s8_10_%j.err

# Load env
source ~/.bashrc
conda activate espnet39   # change if you use a different env

# Go to your UniVersa experiment dir
cd /work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1   # adjust if needed

./uni_versa.sh \
  --skip_data_prep true \
  --universa_exp exp/universa_ar_overall_base_token_wavlm_large_hificodec_quantizer \
  --universa_stats_dir exp/universa_ar_overall_base_token_wavlm_large_hificodec_quantizer_stats \
  --inference_tag eval_hificodec_quantizer_generation_test \
  --train_config ./conf/train_aruniversa_hificodec_quantizer.yaml \
  --inference_config ./conf/decode_universa.yaml \
  --train_set overall_base \
  --valid_set overall_dev \
  --test_sets generation_test \
  --audio_format wav.ark \
  --use_ref_text false \
  --use_ref_wav false \
  --ngpu 2 \
  --train_args "--use_ref_text false --use_ref_audio false --unused_parameters true --resume true" \
  --stage 8 --stop_stage 10 > run_stage8to10.log 2>&1
