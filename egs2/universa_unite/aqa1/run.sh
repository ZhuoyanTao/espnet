#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Every row gets the same 1 s silent reference clip (local/data.sh), as in the
# released model's training; local/data.sh writes data/metric2id.
./aqa.sh \
    --train_config conf/train.yaml \
    --inference_config conf/decode.yaml \
    --train_set train \
    --valid_set valid \
    --test_sets "dev" \
    --use_ref_wav true \
    --use_ref_text false \
    --metric2id data/metric2id \
    --ngpu 4 "$@"
