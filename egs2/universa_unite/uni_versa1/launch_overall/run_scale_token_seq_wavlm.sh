#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#     --test_sets "overall_dev enhancement_simu_test enhancement_real_test generation_test" \

./uni_versa.sh \
    --nj 20 \
    --inference_nj 8 \
    --train_config conf/train_aruniversa_wavlm.yaml \
    --inference_config conf/decode_universa.yaml \
    --train_set overall_scale \
    --valid_set overall_dev \
    --test_sets "overall_dev enhancement_simu_test enhancement_real_test generation_test" \
    --dumpdir dump \
    --audio_format flac.ark \
    --use_ref_text false \
    --inference_nj 8 \
    --gpu_inference true \
    --tokenize_metric true \
    --metric_token_size 500 \
    --metric_token_method percentile \
    --metric_token_percentile_distribution linear \
    --tag universa_ar_overall_scale_token_wavlm \
    --inference_model latest.pth \
    --universa_stats_dir exp/universa_stats_overall_scale  \
    --ngpu 1 "$@"
