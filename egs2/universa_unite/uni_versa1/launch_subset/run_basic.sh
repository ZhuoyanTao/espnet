#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_domain=basic_speech
train_tag=_base
# data_domain=enhancement_real

./uni_versa.sh \
    --nj 20 \
    --inference_nj 8 \
    --train_config conf/train_universa_flex.yaml \
    --inference_config conf/decode_universa.yaml \
    --train_set ${data_domain}${train_tag} \
    --valid_set ${data_domain}_scale \
    --test_sets "${data_domain}_base" \
    --tag ${data_domain}_flex \
    --dumpdir dump \
    --audio_format flac.ark \
    --use_ref_text false \
    --inference_nj 1 \
    --gpu_inference true \
    --tokenize_metric false \
    --universa_stats_dir exp/universa_stats_${data_domain}${train_tag}  \
    --ngpu 1 "$@"
