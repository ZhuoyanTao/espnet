#!/usr/bin/env bash
# CPU integration test of the restoration recipe on the an4 sample: a tiny
# randomly initialised w2v-BERT 2.0 stands in for the released encoder
# (local/data.sh writes it), the vocoders are shrunk to a few thousand
# parameters, and every stage up to inference runs in a few minutes. Stage 10
# and 11 (scoring) need NISQA / VERSA and are not part of the test.
set -euo pipefail

./rst.sh \
    --ngpu 0 \
    --nj 1 \
    --stop_stage 9 \
    --rir_pool_size 4 \
    --train_config conf/train_debug.yaml \
    --expdir exp/rst_debug \
    --voc_pretrain_config conf/train_rst_vocoder_dac_pretrain_debug.yaml \
    --voc_finetune_config conf/train_rst_vocoder_dac_finetune_debug.yaml \
    --voc_pretrain_exp exp/rst_vocoder_dac_pretrain_debug \
    --voc_finetune_exp exp/rst_vocoder_dac_finetune_debug \
    --test_sets test \
    --train_args "--num_workers 0" \
    --voc_args "--num_workers 0" \
    --inference_args "--device cpu" \
    "$@"
