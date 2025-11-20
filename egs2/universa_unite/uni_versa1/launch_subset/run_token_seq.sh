#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_domain=enhancement_simu
train_tag=_base
# data_domain=enhancement_real

./uni_versa.sh \
    --nj 20 \
    --inference_nj 8 \
    --train_config conf/train_aruniversa.yaml \
    --inference_config conf/decode_universa.yaml \
    --train_set ${data_domain}${train_tag} \
    --valid_set ${data_domain}_dev \
    --test_sets "${data_domain}_dev ${data_domain}_test" \
    --tag ${data_domain}_flex \
    --dumpdir dump \
    --audio_format flac.ark \
    --use_ref_text false \
    --inference_nj 1 \
    --gpu_inference true \
    --tokenize_metric true \
    --tag universa_ar_${data_domain}${train_tag}_token \
    --metric_token_size 500 \
    --metric_token_method percentile \
    --metric_token_percentile_distribution linear \
    --universa_stats_dir exp/universa_stats_${data_domain}${train_tag}  \
    --ngpu 1 "$@"
