#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Stages and their options: egs2/TEMPLATE/rst1/rst.sh. Options given to this
# script override the ones below, e.g.
#   ./run.sh --train_config conf/tuning/train_rst_xeus.yaml --expdir exp/rst_xeus
./rst.sh \
    --ngpu 4 \
    --nj 64 \
    --train_config conf/train.yaml \
    --inference_config conf/decode.yaml \
    --expdir exp/rst_w2v_bert2 \
    --voc_pretrain_config conf/tuning/train_rst_vocoder_dac_pretrain.yaml \
    --voc_finetune_config conf/tuning/train_rst_vocoder_dac_finetune.yaml \
    --voc_pretrain_exp exp/rst_vocoder_dac_pretrain \
    --voc_finetune_exp exp/rst_vocoder_dac_finetune \
    --test_sets "test-clean test-other" "$@"
