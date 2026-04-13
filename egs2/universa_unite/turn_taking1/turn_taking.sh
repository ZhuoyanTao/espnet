#!/usr/bin/env bash
# Quality-Aware Turn-Taking Recipe
# Extends ANCHOR (ar_universa) with causal WavLM-Large and turn-taking tokens.
#
# Stages:
#   1  Generate token list (quality + turn-taking)
#   2  Prepare chunk-level Switchboard data (wav.scp + metric.scp)
#   3  Validate data (print chunk counts, verify all required files)
#   4  Train (submit all three configs to SLURM; skip any already submitted)
#   5  Decode / inference (40 ms stride)
#   6  Evaluate (turn-taking metrics A-E + ROC-AUC)

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
stop_stage=6
ngpu=1
python=python3
expdir=exp

# ===========================================================================
# Data paths — set before running stage 2
# ===========================================================================
# Directory containing *_Mono*.csv files (output of swbd/slu1 local/data.sh)
swbd_label_dir=        # e.g. /work/nvme/bbjs/ttao3/workspace/espnet/egs2/swbd/slu1
# Combined per-conversation mono wav.scp (sw0XXXX → sox command).
# Build after slu1 stage 3: cat slu1/data/{train,valid,test}/wav.scp | sort -u > swbd_all.scp
swbd_wavscp=           # e.g. /work/nvme/bbjs/ttao3/workspace/espnet/egs2/swbd/slu1/data/all_splits.wav.scp

# ===========================================================================
# Chunk data parameters (stage 2)
# ===========================================================================
context_s=4.0    # audio context window per training example (seconds)
stride_s=0.04    # hop between training windows — 0.04 matches 40 ms label granularity

# ===========================================================================
# Training configurations (stage 4)
# Each entry: "config_path:experiment_tag"
# ===========================================================================
train_config_tags=(
    "conf/train_ar_turn_taking.yaml:turn_taking_causal"
    "conf/train_ar_turn_taking_noncausal_ablation.yaml:turn_taking_noncausal"
    "conf/train_ar_turn_taking_only.yaml:turn_taking_only"
)

# SLURM resource settings
slurm_partition=gpuA100x4
slurm_account=bbjs-delta-gpu
slurm_time=2-00:00:00
slurm_mem=60000M
slurm_cpus=16

# ===========================================================================
# Inference options (stage 5)
# ===========================================================================
inference_model=valid.loss.best.pth
infer_context_s=4.0   # must match context_s used in training
infer_stride_s=0.04   # 40 ms for frame-level turn-taking output

# ===========================================================================
# Evaluation options (stage 6)
# ===========================================================================
# Switchboard test reference CSV
swbd_test_ref=   # e.g. egs2/swbd/slu1/Test_Two_Channel_Label_Mono.csv
# Path to egs2/swbd/slu1 (for compute_turn_take_metrics.py)
slu1_root=/work/nvme/bbjs/ttao3/workspace/espnet/egs2/swbd/slu1

# ===========================================================================

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

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

    # TT-only ablation: 5-class turn-taking tokens only
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
print(f"Written turn-taking-only token list: {len(vocab)} tokens → {out}")
PYEOF
    log "Stage 1 done."
fi

# ---------------------------------------------------------------------------
# Stage 2: Prepare chunk-level Switchboard data
# ---------------------------------------------------------------------------
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Preparing chunk-level Switchboard data"

    if [ -z "${swbd_label_dir}" ] || [ -z "${swbd_wavscp}" ]; then
        log "ERROR: Set --swbd-label-dir and --swbd-wavscp before running stage 2."
        exit 1
    fi

    ${python} local/prep_swbd_turn_taking.py \
        --csv-dir    "${swbd_label_dir}" \
        --wavscp     "${swbd_wavscp}" \
        --output-dir data \
        --splits train valid test \
        --context-s  "${context_s}" \
        --stride-s   "${stride_s}" \
        --include-quality-meta

    log "Stage 2 done."
fi

# ---------------------------------------------------------------------------
# Stage 3: Validate data
# ---------------------------------------------------------------------------
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: Validating data"

    ok=true
    for split in train valid test; do
        for f in wav.scp metric.scp utt2spk; do
            if [ ! -f "data/${split}/${f}" ]; then
                log "  MISSING: data/${split}/${f}"
                ok=false
            fi
        done
        if [ -f "data/${split}/wav.scp" ]; then
            n=$(wc -l < "data/${split}/wav.scp")
            log "  ${split}: ${n} chunks in wav.scp"
        fi
    done

    for f in "${metric_dir}/metric2id" "${metric_dir}/metric2type"; do
        if [ ! -f "${f}" ]; then
            log "  MISSING: ${f} — run stage 2 first"
            ok=false
        else
            log "  Found: ${f}"
        fi
    done

    if [ "${ok}" = false ]; then
        log "ERROR: Missing files — run stage 2 first."
        exit 1
    fi

    # Generate shape files (required by ESPnet batch sampler even for unsorted batching)
    # All chunks are ~4s @ 16kHz = 64000 samples; unsorted sampler ignores the value.
    for split in train valid test; do
        shape_file="data/${split}/audio_shape"
        if [ ! -f "${shape_file}" ]; then
            log "  Generating ${shape_file}"
            awk '{print $1, 64000}' "data/${split}/wav.scp" > "${shape_file}"
        else
            log "  ${shape_file} already exists ($(wc -l < ${shape_file}) entries)"
        fi
    done

    log "Stage 3: Data validation passed."
fi

# ---------------------------------------------------------------------------
# Stage 4: Training (all three configs submitted to SLURM)
#
# Each experiment directory is checked before submission:
#   - If train_*.log already exists → job was previously submitted, skip.
#   - Otherwise submit a new sbatch job.
#
# To force re-submission of a specific config, remove its exp dir first:
#   rm -rf exp/universa_turn_taking_causal
# ---------------------------------------------------------------------------
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: Submitting training jobs"

    _metric2id="${metric_dir}/metric2id"
    _metric2type="${metric_dir}/metric2type"
    if [ ! -f "${_metric2id}" ]; then
        log "ERROR: ${_metric2id} not found. Run stage 2 first."
        exit 1
    fi

    PYTHON=$(which ${python})
    WD=$(pwd -P)

    for config_tag in "${train_config_tags[@]}"; do
        _config="${config_tag%%:*}"
        _tag="${config_tag##*:}"
        _expdir="${expdir}/universa_${_tag}"

        mkdir -p "${_expdir}"

        # Skip if this experiment was already submitted (sentinel file or checkpoint present)
        if [ -f "${_expdir}/.submitted" ] || ls "${_expdir}"/train_*.log &>/dev/null 2>&1 || \
           [ -f "${_expdir}/valid.loss.best.pth" ]; then
            _reason="checkpoint or sentinel exists"
            [ -f "${_expdir}/.submitted" ] && _reason="previously submitted (job $(<${_expdir}/.submitted))"
            log "  [skip] ${_tag}: ${_reason}"
            continue
        fi

        log "  Submitting: ${_tag} (config=${_config})"
        _jobid=$(sbatch \
            -p "${slurm_partition}" \
            --gres=gpu:"${ngpu}" \
            -c "${slurm_cpus}" \
            --mem "${slurm_mem}" \
            --account="${slurm_account}" \
            -t "${slurm_time}" \
            --job-name "tt_${_tag}" \
            --output "${WD}/${_expdir}/train_%j.log" \
            --wrap "
                cd ${WD}
                export PYTHONPATH=/work/nvme/bbjs/ttao3/workspace/espnet:${PYTHONPATH:-}
                ${PYTHON} -m espnet2.bin.universa_train \
                    --config ${_config} \
                    --ngpu ${ngpu} \
                    --num_workers 4 \
                    --use_preprocessor true \
                    --metric2id ${_metric2id} \
                    --metric2type ${_metric2type} \
                    --use_ref_audio true \
                    --use_ref_text false \
                    --train_shape_file data/train/audio_shape \
                    --valid_shape_file data/valid/audio_shape \
                    --train_data_path_and_name_and_type 'data/train/wav.scp,audio,kaldi_ark' \
                    --train_data_path_and_name_and_type 'data/train/metric.scp,metrics,metric' \
                    --valid_data_path_and_name_and_type 'data/valid/wav.scp,audio,kaldi_ark' \
                    --valid_data_path_and_name_and_type 'data/valid/metric.scp,metrics,metric' \
                    --output_dir ${WD}/${_expdir}
            " | grep -oP '(?<=Submitted batch job )\d+')
        echo "${_jobid}" > "${_expdir}/.submitted"
        log "  Submitted job ${_jobid} for ${_tag} → ${_expdir}/train_${_jobid}.log"
    done

    log "Stage 4 done. Monitor: squeue -u \$USER"
fi

# ---------------------------------------------------------------------------
# Stage 5: Inference (40 ms frame-level turn-taking probabilities)
#
# Runs inference for every trained model found under expdir.
# ---------------------------------------------------------------------------
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    log "Stage 5: Inference on test set (stride=${infer_stride_s}s)"

    for config_tag in "${train_config_tags[@]}"; do
        _tag="${config_tag##*:}"
        _expdir="${expdir}/universa_${_tag}"
        _model="${_expdir}/${inference_model}"

        if [ ! -f "${_model}" ]; then
            log "  [skip] ${_tag}: ${_model} not found (training not complete?)"
            continue
        fi

        decode_dir="${_expdir}/decode_test"
        mkdir -p "${decode_dir}"

        log "  Running inference for ${_tag}"
        ${python} pyscripts/run_turn_taking_inference.py \
            --model-dir  "${_expdir}" \
            --wavscp     data/test/wav.scp \
            --output     "${decode_dir}/text" \
            --context-s  "${infer_context_s}" \
            --stride-s   "${infer_stride_s}"

        log "  Done: ${decode_dir}/text"
    done

    log "Stage 5 done."
fi

# ---------------------------------------------------------------------------
# Stage 6: Evaluate (Talking Turns metrics A-E + ROC-AUC)
# ---------------------------------------------------------------------------
if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    log "Stage 6: Evaluating turn-taking metrics"

    if [ -z "${swbd_test_ref}" ]; then
        log "ERROR: Set --swbd-test-ref to Test_Two_Channel_Label_Mono.csv path."
        exit 1
    fi

    for config_tag in "${train_config_tags[@]}"; do
        _tag="${config_tag##*:}"
        _expdir="${expdir}/universa_${_tag}"
        decode_dir="${_expdir}/decode_test"

        if [ ! -f "${decode_dir}/text" ]; then
            log "  [skip] ${_tag}: ${decode_dir}/text not found (run stage 5 first)"
            continue
        fi

        log "  Evaluating ${_tag}"
        ${python} pyscripts/eval_turn_taking.py \
            --hyp        "${decode_dir}/text" \
            --ref        "${swbd_test_ref}" \
            --output-dir "${decode_dir}/eval" \
            --slu1-root  "${slu1_root}"

        log "  Results: ${decode_dir}/eval/results.json"
    done

    log "Stage 6 done."
fi

log "Finished (${SECONDS}s)"
