#!/usr/bin/env bash
# Quality-Aware Turn-Taking Recipe
# Extends ANCHOR (ar_universa) with causal WavLM-Large and turn-taking tokens.
#
# Stages:
#   1  Generate token list (quality + turn-taking)
#   2  Prepare chunk-level Switchboard data (wav.scp + metric.scp)
#   3  Train model
#   4  Decode / inference (40 ms stride)
#   5  Evaluate (turn-taking metrics A-E + ROC-AUC)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# ===========================================================================
# General options
# ===========================================================================
stage=1
stop_stage=5
ngpu=1
python=python3
expdir=exp

# ===========================================================================
# Data paths — set before running
# ===========================================================================
# Directory containing *_Mono*.csv files (output of slu1 local/data.sh stage 3)
swbd_label_dir=        # e.g. /work/nvme/bbjs/ttao3/workspace/espnet/egs2/swbd/slu1
# Combined per-conversation mono wav.scp (sw0XXXX → sox command).
# Build it after slu1 data.sh stage 3 completes:
#   cat slu1/data/train/wav.scp slu1/data/valid/wav.scp slu1/data/test/wav.scp > /tmp/swbd_all_mono.scp
swbd_wavscp=           # e.g. /tmp/swbd_all_mono.scp

# ===========================================================================
# Training options
# ===========================================================================
# Choose one of the three configs:
#   conf/train_ar_turn_taking.yaml                    (full proposed model)
#   conf/train_ar_turn_taking_noncausal_ablation.yaml (non-causal ablation)
#   conf/train_ar_turn_taking_only.yaml               (TT-only ablation)
train_config=conf/train_ar_turn_taking.yaml
tag=turn_taking_causal

# Context window and training stride for chunk data prep
context_s=4.0    # seconds of audio context per training example
stride_s=2.0     # hop between training windows (use 0.04 for dense sampling)

# ANCHOR warm-start checkpoint (ignore_init_mismatch=true allows head mismatches)
anchor_ckpt=/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/exp/universa_universa_ar_overall_base_token_wavlm_large/valid.loss.best.prefix_65828.pth

# ===========================================================================
# Inference options
# ===========================================================================
inference_model=valid.loss.best.pth
infer_context_s=4.0   # must match context_s used in training
infer_stride_s=0.04   # 40 ms for frame-level output

# Switchboard test reference CSV for evaluation
swbd_test_ref=   # e.g. /path/to/slu1/Test_Two_Channel_Label_Mono.csv
# Path to egs2/swbd/slu1 (for pyscripts/utils/compute_turn_take_metrics.py)
slu1_root=/work/nvme/bbjs/ttao3/workspace/espnet/egs2/swbd/slu1

# ===========================================================================

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

slu_exp="${expdir}/universa_${tag}"
token_dir=data/token_list
metric_dir=data/local   # metric2id and metric2type go here

# ---------------------------------------------------------------------------
# Stage 1: Generate combined token list
# ---------------------------------------------------------------------------
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Generating token list (quality + turn-taking)"

    # Full model and non-causal ablation: quality + TT tokens
    ${python} local/create_turn_taking_token_list.py \
        --output "${token_dir}/turn_taking_tokens/tokens.json"

    # TT-only ablation: only 5-class turn-taking tokens
    ${python} - <<'PYEOF'
import json
from pathlib import Path

categories = ["C", "NA", "I", "T", "BC"]
vocab = ["turn_taking@meta_label"] + [f"turn_taking@{i}" for i in range(len(categories))]
data = {
    "tokenizer": {"turn_taking": categories},
    "VOCAB": vocab,
    "offset": {"turn_taking": [0, len(vocab)]},
}
out = Path("data/token_list/turn_taking_only_tokens/tokens.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(data, f, indent=2)
print(f"Written turn-taking-only token list to {out}")
PYEOF
    log "Stage 1 done."
fi

# ---------------------------------------------------------------------------
# Stage 2: Prepare chunk-level Switchboard data
# ---------------------------------------------------------------------------
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Preparing chunk-level Switchboard data"

    if [ -z "${swbd_label_dir}" ] || [ -z "${swbd_wavscp}" ]; then
        echo "ERROR: Set --swbd-label-dir and --swbd-wavscp before running stage 2."
        exit 1
    fi

    ${python} local/prep_swbd_turn_taking.py \
        --csv-dir    "${swbd_label_dir}" \
        --wavscp     "${swbd_wavscp}" \
        --output-dir data \
        --splits train valid test \
        --context-s  "${context_s}" \
        --stride-s   "${stride_s}"

    for split in train valid test; do
        n=$(wc -l < "data/${split}/wav.scp" 2>/dev/null || echo 0)
        log "  ${split}: ${n} chunks"
    done

    log "Stage 2 done."
fi

# ---------------------------------------------------------------------------
# Stage 3: Train
# ---------------------------------------------------------------------------
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: Training — config=${train_config}, tag=${tag}"

    mkdir -p "${slu_exp}"

    # Determine metric2id and metric2type from what was generated in stage 2
    _metric2id="${metric_dir}/metric2id"
    _metric2type="${metric_dir}/metric2type"

    if [ ! -f "${_metric2id}" ]; then
        echo "ERROR: ${_metric2id} not found. Run stage 2 first."
        exit 1
    fi

    sbatch -p gpuA100x4 \
        --gres=gpu:${ngpu} \
        -c 16 \
        --mem 60000M \
        --account=bbjs-delta-gpu \
        -t 2-00:00:00 \
        --job-name "tt_${tag}" \
        --output "${slu_exp}/train_%j.log" \
        --wrap "
            ${python} -m espnet2.bin.universa_train \
                --config ${train_config} \
                --ngpu ${ngpu} \
                --num_workers 4 \
                --use_preprocessor true \
                --metric2id ${_metric2id} \
                --metric2type ${_metric2type} \
                --train_data_path_and_name_and_type 'data/train/wav.scp,audio,sound' \
                --train_data_path_and_name_and_type 'data/train/metric.scp,metrics,metric' \
                --valid_data_path_and_name_and_type 'data/valid/wav.scp,audio,sound' \
                --valid_data_path_and_name_and_type 'data/valid/metric.scp,metrics,metric' \
                --init_param ${anchor_ckpt} \
                --ignore_init_mismatch true \
                --output_dir ${slu_exp}
        "

    log "Stage 3: Training job submitted. Monitor: squeue -u \$USER"
fi

# ---------------------------------------------------------------------------
# Stage 4: Inference (40 ms frame-level turn-taking probabilities)
# ---------------------------------------------------------------------------
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: Inference on test set (stride=${infer_stride_s}s)"

    decode_dir="${slu_exp}/decode_test"
    mkdir -p "${decode_dir}"

    ${python} pyscripts/run_turn_taking_inference.py \
        --model-dir  "${slu_exp}" \
        --wavscp     data/test/wav.scp \
        --output     "${decode_dir}/text" \
        --context-s  "${infer_context_s}" \
        --stride-s   "${infer_stride_s}"

    log "Stage 4 done. Output: ${decode_dir}/text"
fi

# ---------------------------------------------------------------------------
# Stage 5: Evaluate (Talking Turns metrics A-E + ROC-AUC)
# ---------------------------------------------------------------------------
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    log "Stage 5: Evaluating turn-taking metrics"

    if [ -z "${swbd_test_ref}" ]; then
        echo "ERROR: Set --swbd-test-ref to Test_Two_Channel_Label_Mono.csv path."
        exit 1
    fi

    decode_dir="${slu_exp}/decode_test"

    ${python} pyscripts/eval_turn_taking.py \
        --hyp        "${decode_dir}/text" \
        --ref        "${swbd_test_ref}" \
        --output-dir "${decode_dir}/eval" \
        --slu1-root  "${slu1_root}"

    log "Stage 5 done. Results: ${decode_dir}/eval/results.json"
fi

log "Finished (${SECONDS}s)"
