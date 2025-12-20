#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


./uni_versa.sh \
    --nj 20 \
    --inference_nj 8 \
    --train_config conf/train_universa_flex.yaml \
    --inference_config conf/decode_universa.yaml \
    --train_set overall_scale \
    --valid_set overall_dev \
    --test_sets "overall_dev enhancement_simu_test enhancement_real_test generation_test" \
    --tag overall_scale_flex \
    --dumpdir dump \
    --audio_format flac.ark \
    --use_ref_text false \
    --inference_nj 1 \
    --gpu_inference true \
    --tokenize_metric false \
    --inference_model valid.loss.best.pth \
    --tag universa_flex_overall_scale_no-token \
    --universa_stats_dir exp/universa_stats_overall_scale  \
    --ngpu 1 "$@"
