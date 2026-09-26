#!/usr/bin/env bash
set -euo pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

# Restoration (rst) driver: feature predictor (stages 4-5), vocoder (6-8),
# inference (9), scoring (10-11), results, packing and upload (12-14). A
# recipe's run.sh sets the corpus-specific defaults and calls this script;
# local/data.sh (stage 1) must produce data/{train,dev}_fp (feature-predictor
# sets, any rate), data/{train,dev}_voc (48 kHz vocoder sets), data/<test_set>
# (with text for scoring), and data/noise_pool.
stage=1
stop_stage=14
skip_packing=true       # Skip the packing stage (13).
skip_upload_hf=true     # Skip the Hugging Face upload stage (14).
ngpu=4
nj=64
python=python3
local_data_opts=        # Options given to local/data.sh.
rir_pool_size=50000     # Room impulse responses simulated in stage 3.
# --train_config, not --config: utils/parse_options.sh sources a --config file as shell.
train_config=conf/train.yaml
decode_config=conf/decode.yaml
expdir=exp/rst_w2v_bert2
train_args=             # Extra arguments to rst_train (stage 5).
voc_args=               # Extra arguments to rst_vocoder_train (stages 7-8).
inference_args=         # Extra arguments to rst_inference (stage 9).
hf_repo=                # Hugging Face repository for stage 14, e.g. user/name.
# Vocoder (stages 6-8). Stage 7 pretrains on ground-truth features, stage 8
# finetunes on the stage-5 predictor's features. The config's vocoder_type
# picks the model and with it the objective: dac (default) and hifigan train
# adversarially, cfm and periodwave by conditional flow matching; the same
# rst_vocoder_train command serves all four (conf/tuning/train_rst_vocoder_*).
voc_pretrain_config=conf/tuning/train_rst_vocoder_dac_pretrain.yaml
voc_finetune_config=conf/tuning/train_rst_vocoder_dac_finetune.yaml
voc_pretrain_exp=exp/rst_vocoder_dac_pretrain
voc_finetune_exp=exp/rst_vocoder_dac_finetune
# Stage 8 initialises the generator and the discriminator from the stage-7
# best checkpoint (this recipe's own run); --vocoder_init/--discriminator_init
# only point it at a different stage-7 run. Nothing is initialised from the
# published Sidon weights.
vocoder_init=
discriminator_init=
# Vocoder used at inference: an ESPnet-trained one (default the stage-8
# best) or, if --external_vocoder is set, an externally released TorchScript
# decoder such as the Sidon v0.1 one (comparison only).
vocoder_exp=
vocoder_model_file=
external_vocoder=
test_sets="test-clean test-other"
versa_config=conf/versa_enh.yaml
versa_ref_config=conf/versa_enh_ref_based.yaml
# Optional clean reference for synthetically degraded inputs.  The placeholder
# {test_set} is replaced per evaluation set.
ref_wav_scp=

. utils/parse_options.sh

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*"; }
if [ $# -ne 0 ]; then
    log "Error: no positional arguments are accepted."
    exit 2
fi

# The vocoder_type of a config (or of an experiment's config.yaml) also fixes
# which checkpoint the later stages pick up: the GAN types are selected by the
# validation mel loss, the flow-matching types by the flow-matching loss, and
# only the GAN types have a discriminator to initialise in stage 8.
vocoder_type_of() { awk '$1 == "vocoder_type:" {print $2}' "$1"; }
vocoder_best_of() {
    case "$(vocoder_type_of "$1")" in
        cfm|periodwave) echo valid.loss.best.pth ;;
        *) echo valid.loss_mel.best.pth ;;
    esac
}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: data preparation"
    local/data.sh ${local_data_opts}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: resample feature-predictor data to 16 kHz"
    for split in train dev; do
        scripts/audio/format_wav_scp.sh \
            --nj "${nj}" --cmd "${train_cmd}" --fs 16000 --audio-format wav \
            "data/${split}_fp/wav.scp" "data/${split}_fp_16k"
    done
    for test_set in ${test_sets}; do
        scripts/audio/format_wav_scp.sh \
            --nj "${nj}" --cmd "${decode_cmd}" --fs 16000 --audio-format wav \
            "data/${test_set}/wav.scp" "data/${test_set}_16k"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: generate RIR pool"
    ${python} pyscripts/utils/prepare_rir_pool.py \
        --out_dir data/rir_pool --n_rirs ${rir_pool_size} --nj ${nj}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: collect feature-predictor statistics"
    ${python} -m espnet2.bin.rst_train \
        --config ${train_config} \
        --train_data_path_and_name_and_type data/train_fp_16k/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_fp_16k/wav.scp,speech_ref1,sound \
        --output_dir ${expdir} --collect_stats true --ngpu 0
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: train feature predictor"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        ${python} -m espnet2.bin.rst_train \
        --config ${train_config} \
        --train_data_path_and_name_and_type data/train_fp_16k/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_fp_16k/wav.scp,speech_ref1,sound \
        --train_shape_file ${expdir}/train/speech_ref1_shape \
        --valid_shape_file ${expdir}/valid/speech_ref1_shape \
        --output_dir ${expdir} --ngpu ${ngpu} \
        --multiprocessing_distributed true --unused_parameters true --resume true \
        ${train_args}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: collect vocoder statistics"
    ${python} -m espnet2.bin.rst_vocoder_train \
        --config ${voc_pretrain_config} \
        --train_data_path_and_name_and_type data/train_voc/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_voc/wav.scp,speech_ref1,sound \
        --output_dir ${voc_pretrain_exp} --collect_stats true --ngpu 0
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: pretrain vocoder on ground-truth SSL features"
    ${cuda_cmd} --gpu ${ngpu} ${voc_pretrain_exp}/train.log \
        ${python} -m espnet2.bin.rst_vocoder_train \
        --config ${voc_pretrain_config} \
        --train_data_path_and_name_and_type data/train_voc/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_voc/wav.scp,speech_ref1,sound \
        --train_shape_file ${voc_pretrain_exp}/train/speech_ref1_shape \
        --valid_shape_file ${voc_pretrain_exp}/valid/speech_ref1_shape \
        --output_dir ${voc_pretrain_exp} --ngpu ${ngpu} \
        --multiprocessing_distributed true --unused_parameters true --resume true \
        ${voc_args}
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: finetune vocoder on predicted SSL features"
    vocoder_init=${vocoder_init:-${voc_pretrain_exp}/$(vocoder_best_of "${voc_pretrain_config}")}
    init_opts=(--init_param "${vocoder_init}:vocoder:vocoder")
    case "$(vocoder_type_of "${voc_finetune_config}")" in
        cfm|periodwave) ;;  # no discriminator
        *)
            discriminator_init=${discriminator_init-${vocoder_init}}
            if [ -n "${discriminator_init}" ]; then
                init_opts+=(--init_param "${discriminator_init}:discriminator:discriminator")
            fi
            ;;
    esac
    # Same utterances as stage 7, so its shape files are reused.
    ${cuda_cmd} --gpu ${ngpu} ${voc_finetune_exp}/train.log \
        ${python} -m espnet2.bin.rst_vocoder_train \
        --config ${voc_finetune_config} \
        --fp_model_path ${expdir}/valid.loss.best.pth \
        "${init_opts[@]}" \
        --train_data_path_and_name_and_type data/train_voc/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_voc/wav.scp,speech_ref1,sound \
        --train_shape_file ${voc_pretrain_exp}/train/speech_ref1_shape \
        --valid_shape_file ${voc_pretrain_exp}/valid/speech_ref1_shape \
        --output_dir ${voc_finetune_exp} --ngpu ${ngpu} \
        --multiprocessing_distributed true --unused_parameters true --resume true \
        ${voc_args}
fi

# The vocoder stages 9 and 13 use: this recipe's stage-8 run by default, or
# --vocoder_exp / --vocoder_model_file for another trained one. Sets
# vocoder_exp and vocoder_model_file; fails if the files are missing.
resolve_vocoder() {
    vocoder_exp=${vocoder_exp:-${voc_finetune_exp}}
    vocoder_model_file=${vocoder_model_file:-${vocoder_exp}/$(vocoder_best_of "${vocoder_exp}/config.yaml")}
    for required_file in "${vocoder_exp}/config.yaml" "${vocoder_model_file}"; do
        [ -f "${required_file}" ] || {
            log "Missing vocoder file ${required_file}: train one (stages 6-8) or set --external_vocoder"
            exit 1
        }
    done
}

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    if [ -n "${external_vocoder}" ]; then
        vocoder_opts=(--external_vocoder "${external_vocoder}")
    else
        resolve_vocoder
        vocoder_opts=(--vocoder_train_config "${vocoder_exp}/config.yaml"
                      --vocoder_model_file "${vocoder_model_file}")
    fi
    for test_set in ${test_sets}; do
        log "Stage 9: inference (${test_set})"
        ${python} -m espnet2.bin.rst_inference \
            --config ${decode_config} \
            --train_config ${expdir}/config.yaml \
            --model_file ${expdir}/valid.loss.best.pth \
            "${vocoder_opts[@]}" \
            --wav_scp data/${test_set}_16k/wav.scp \
            --output_dir ${expdir}/inference_${test_set} \
            ${inference_args}
    done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    for test_set in ${test_sets}; do
        # The paper's four metrics without VERSA; stage 11 (VERSA) covers them
        # and more, so this stage can be skipped when VERSA is installed.
        log "Stage 10: scoring (${test_set})"
        text_opt=()
        if [ -f "data/${test_set}/text" ]; then
            text_opt=(--text "data/${test_set}/text")
        fi
        ${python} local/score.py \
            --restored_dir ${expdir}/inference_${test_set}/wav \
            --ref_wav_scp data/${test_set}/wav.scp \
            --noisy_wav_scp data/${test_set}_16k/wav.scp \
            "${text_opt[@]}" \
            --output_dir ${expdir}/score_${test_set}
    done
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    ${python} -c "import versa" || {
        log "VERSA is required for stage 11; run tools/installers/install_versa.sh"
        exit 1
    }
    for test_set in ${test_sets}; do
        log "Stage 11: VERSA scoring (${test_set})"
        inf_dir=${expdir}/inference_${test_set}
        eval_dir=${inf_dir}/scoring/versa_eval
        pred_scp=${inf_dir}/wav.scp
        input_scp=data/${test_set}_16k/wav.scp
        text=data/${test_set}/text

        for required_file in "${pred_scp}" "${input_scp}" "${text}"; do
            [ -f "${required_file}" ] || {
                log "Missing VERSA input: ${required_file}"
                exit 1
            }
        done
        mkdir -p "${eval_dir}"
        num_pred=$(wc -l < "${pred_scp}")
        score_nj=$(( nj < num_pred ? nj : num_pred ))
        [ "${score_nj}" -gt 0 ] || { log "No inference output to score"; exit 1; }

        split_pred=()
        for n in $(seq "${score_nj}"); do
            split_pred+=("${eval_dir}/pred.${n}")
        done
        utils/split_scp.pl "${pred_scp}" "${split_pred[@]}"

        ${decode_cmd} JOB=1:"${score_nj}" "${eval_dir}/versa.JOB.log" \
            ${python} -m versa.bin.scorer \
                --pred "${eval_dir}/pred.JOB" \
                --gt "${input_scp}" \
                --text "${text}" \
                --score_config "${versa_config}" \
                --cache_folder "${eval_dir}/cache" \
                --output_file "${eval_dir}/result.JOB.txt" \
                --io soundfile
        ${python} pyscripts/utils/aggregate_eval.py \
            --logdir "${eval_dir}" --scoredir "${eval_dir}" --nj "${score_nj}"

        if [ -n "${ref_wav_scp}" ]; then
            ref_scp=${ref_wav_scp//\{test_set\}/${test_set}}
            [ -f "${ref_scp}" ] || { log "Missing clean reference: ${ref_scp}"; exit 1; }
            ref_dir=${inf_dir}/scoring/versa_ref
            mkdir -p "${ref_dir}"
            ${decode_cmd} JOB=1:"${score_nj}" "${ref_dir}/versa.JOB.log" \
                ${python} -m versa.bin.scorer \
                    --pred "${eval_dir}/pred.JOB" \
                    --gt "${ref_scp}" \
                    --score_config "${versa_ref_config}" \
                    --cache_folder "${ref_dir}/cache" \
                    --output_file "${ref_dir}/result.JOB.txt" \
                    --io soundfile
            ${python} pyscripts/utils/aggregate_eval.py \
                --logdir "${ref_dir}" --scoredir "${ref_dir}" --nj "${score_nj}"
        else
            log "Skipping reference-based VERSA metrics"
        fi
    done
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    log "Stage 12: collect results: ${expdir}/RESULTS.md"
    ${python} scripts/utils/show_rst_result.py \
        --expdir "${expdir}" --test_sets "${test_sets}" > "${expdir}/RESULTS.md"
    cat "${expdir}/RESULTS.md"
fi

packed_model="${expdir}/${expdir##*/}_restoration.zip"
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ] && ! "${skip_packing}"; then
    log "Stage 13: pack model: ${packed_model}"
    if [ -n "${external_vocoder}" ]; then
        log "An externally released vocoder is not packed; train one (stages 6-8) or unset --external_vocoder"
        exit 1
    fi
    resolve_vocoder
    _opts=()
    [ -f "${expdir}/RESULTS.md" ] && _opts+=(--option "${expdir}/RESULTS.md")
    [ -d "${expdir}/images" ] && _opts+=(--option "${expdir}/images")
    ${python} -m espnet2.bin.pack rst \
        --train_config "${expdir}/config.yaml" \
        --model_file "${expdir}/valid.loss.best.pth" \
        --vocoder_train_config "${vocoder_exp}/config.yaml" \
        --vocoder_model_file "${vocoder_model_file}" \
        "${_opts[@]}" \
        --outpath "${packed_model}"
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ] && ! "${skip_upload_hf}"; then
    [ -n "${hf_repo}" ] || {
        log "ERROR: set --hf_repo <user/name> (an existing Hugging Face repository); see https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes"
        exit 1
    }
    log "Stage 14: upload model to Hugging Face: ${hf_repo}"
    [ -f "${packed_model}" ] || {
        log "ERROR: ${packed_model} does not exist. Run stage 13 first (--skip_packing false)."
        exit 1
    }
    git lfs --version > /dev/null 2>&1 || {
        log "ERROR: git-lfs is required"
        exit 1
    }
    dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
    [ -d "${dir_repo}" ] || git clone "https://huggingface.co/${hf_repo}" "${dir_repo}"
    if command -v git > /dev/null 2>&1; then
        _creator_name="$(git config user.name)"
        _checkout="git checkout $(git show -s --format=%H)"
    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/rst1/ -> foo/rst1 -> foo
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    _corpus="${_task%/*}"
    unzip -o "${packed_model}" -d "${dir_repo}"
    {
        echo "---"
        echo "tags:"
        echo "- espnet"
        echo "- audio"
        echo "- audio-to-audio"
        echo "- speech-restoration"
        echo "datasets:"
        echo "- ${_corpus}"
        echo "license: cc-by-4.0"
        echo "---"
        echo
        echo "## ESPnet2 restoration (rst) model"
        echo
        echo "### \`${hf_repo}\`"
        echo
        echo "This model was trained by ${_creator_name} using the ${_task} recipe in [espnet](https://github.com/espnet/espnet/)."
        echo "It holds the feature predictor (stage 5) and the vocoder (stage 8) that \`espnet2.bin.rst_inference\` runs together."
        echo
        echo "### Demo: how to use in ESPnet2"
        echo
        echo "\`\`\`bash"
        echo "cd espnet"
        echo "${_checkout}"
        echo "pip install -e \".[rst]\""
        echo "cd egs2/${_task}"
        echo "python -m espnet2.bin.rst_inference --train_config config.yaml --model_file valid.loss.best.pth \\"
        echo "    --vocoder_train_config vocoder/config.yaml --vocoder_model_file vocoder/model.pth \\"
        echo "    --wav_scp data/test_16k/wav.scp --output_dir restored"
        echo "\`\`\`"
        echo
        [ -f "${expdir}/RESULTS.md" ] && cat "${expdir}/RESULTS.md"
        echo
        echo "## Feature predictor config"
        echo
        echo "<details><summary>expand</summary>"
        echo
        echo "\`\`\`"
        cat "${expdir}/config.yaml"
        echo "\`\`\`"
        echo
        echo "</details>"
    } > "${dir_repo}/README.md"
    this_folder=${PWD}
    cd "${dir_repo}"
    if [ -n "$(git status --porcelain)" ]; then
        git add .
        git commit -m "Update model"
    fi
    git push
    cd "${this_folder}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
