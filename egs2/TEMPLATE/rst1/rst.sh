#!/usr/bin/env bash
set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

. ./path.sh
. ./cmd.sh
. ./db.sh

stage=1
stop_stage=13
ngpu=4
nj=64
python=python3
skip_data_prep=false # Skip data preparation stages (1-3).
skip_train=false     # Skip training stages (3-8).
skip_eval=false      # Skip inference and scoring stages (9-11).
skip_packing=true    # Skip the packing stage (12).
skip_upload_hf=true  # Skip uploading to Hugging Face (13).
skip_stages=         # Stages to skip, e.g., "3 10".
# Feature-predictor training config. Not "config": utils/parse_options.sh
# sources a file passed as --config as shell.
fp_config=conf/train.yaml
decode_config=conf/decode.yaml
expdir=exp/rst_w2v_bert2
# Vocoder (stages 6-8). Stage 7 pretrains on ground-truth features, stage 8
# finetunes on the stage-5 predictor's features. The config's vocoder_type
# picks the DAC decoder (default) or ESPnet's HiFi-GAN generator.
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
# Feature-predictor checkpoint used at inference (stage 9) and packed (12).
inference_model=valid.loss.best.pth
test_sets="test-clean test-other"
# Stage 3: number of simulated room impulse responses in data/rir_pool.
n_rirs=50000
versa_config=conf/versa_enh.yaml
versa_ref_config=conf/versa_enh_ref_based.yaml
# Optional clean reference for synthetically degraded inputs.  The placeholder
# {test_set} is replaced per evaluation set.
ref_wav_scp=
# Stage 13: Hugging Face repository (<user>/<name>) and the corpus language
# for its model card.
hf_repo=
lang=noinfo

help_message=$(cat << EOF
Usage: $0 [options]

Options:
    # General configuration
    --stage       # Processes starts from the specified stage (default="${stage}").
    --stop_stage  # Processes is stopped at the specified stage (default="${stop_stage}").
    --ngpu        # The number of GPUs for training (default="${ngpu}").
    --nj          # The number of parallel jobs (default="${nj}").
    --python      # Specify python to execute espnet commands (default="${python}").
    --skip_data_prep # Skip data preparation stages (1-3) (default="${skip_data_prep}").
    --skip_train     # Skip training stages (3-8) (default="${skip_train}").
    --skip_eval      # Skip inference and scoring stages (9-11) (default="${skip_eval}").
    --skip_packing   # Skip the packing stage (12) (default="${skip_packing}").
    --skip_upload_hf # Skip uploading to Hugging Face (13) (default="${skip_upload_hf}").
    --skip_stages    # Stages to skip, e.g., "3 10" (default="${skip_stages}").

    # Data preparation (stages 1-3)
    --test_sets   # Names of test sets (default="${test_sets}").
    --n_rirs      # Number of simulated room impulse responses (default="${n_rirs}").

    # Feature predictor (stages 4-5)
    --fp_config   # Training config of the feature predictor (default="${fp_config}").
    --expdir      # Directory of the feature-predictor experiment (default="${expdir}").

    # Vocoder (stages 6-8)
    --voc_pretrain_config # Vocoder pretraining config (default="${voc_pretrain_config}").
    --voc_finetune_config # Vocoder finetuning config (default="${voc_finetune_config}").
    --voc_pretrain_exp    # Vocoder pretraining directory (default="${voc_pretrain_exp}").
    --voc_finetune_exp    # Vocoder finetuning directory (default="${voc_finetune_exp}").
    --vocoder_init        # Generator initialisation for stage 8
                          # (default: the stage-7 best checkpoint).
    --discriminator_init  # Discriminator initialisation for stage 8
                          # (default: same as --vocoder_init).

    # Inference and scoring (stages 9-11)
    --decode_config      # Inference config (default="${decode_config}").
    --vocoder_exp        # Vocoder used at inference (default: --voc_finetune_exp).
    --vocoder_model_file # Its checkpoint (default: <vocoder_exp>/valid.loss_mel.best.pth).
    --external_vocoder   # A released TorchScript vocoder instead, for comparison
                         # (default="${external_vocoder}").
    --inference_model    # Feature-predictor checkpoint for inference and packing
                         # (default="${inference_model}").
    --versa_config       # VERSA config for stage 11 (default="${versa_config}").
    --versa_ref_config   # VERSA config of the reference-based metrics
                         # (default="${versa_ref_config}").
    --ref_wav_scp        # Clean reference wav.scp of synthetically degraded inputs;
                         # {test_set} is replaced per test set (default="${ref_wav_scp}").

    # Packing and uploading (stages 12-13)
    --hf_repo   # Hugging Face repository to upload to, e.g. <user>/<name> (default="${hf_repo}").
    --lang      # Language of the corpus, for the model card (default="${lang}").
EOF
)

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

# Stage 3's room impulse responses serve training only.
if "${skip_data_prep}"; then
    skip_stages+=" 1 2 3"
fi
if "${skip_train}"; then
    skip_stages+=" 3 4 5 6 7 8"
fi
if "${skip_eval}"; then
    skip_stages+=" 9 10 11"
fi
if "${skip_packing}"; then
    skip_stages+=" 12"
fi
if "${skip_upload_hf}"; then
    skip_stages+=" 13"
fi
skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "Stage 1: data preparation"
    local/data.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
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

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    log "Stage 3: generate RIR pool"
    ${python} local/prepare_rir_pool.py \
        --out_dir data/rir_pool --n_rirs ${n_rirs} --nj ${nj}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
    log "Stage 4: collect feature-predictor statistics"
    ${python} -m espnet2.bin.rst_train \
        --config ${fp_config} \
        --train_data_path_and_name_and_type data/train_fp_16k/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_fp_16k/wav.scp,speech_ref1,sound \
        --output_dir ${expdir} --collect_stats true --ngpu 0
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! [[ " ${skip_stages} " =~ [[:space:]]5[[:space:]] ]]; then
    log "Stage 5: train feature predictor"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        ${python} -m espnet2.bin.rst_train \
        --config ${fp_config} \
        --train_data_path_and_name_and_type data/train_fp_16k/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_fp_16k/wav.scp,speech_ref1,sound \
        --train_shape_file ${expdir}/train/speech_ref1_shape \
        --valid_shape_file ${expdir}/valid/speech_ref1_shape \
        --output_dir ${expdir} --ngpu ${ngpu} \
        --multiprocessing_distributed true --unused_parameters true --resume true
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && ! [[ " ${skip_stages} " =~ [[:space:]]6[[:space:]] ]]; then
    log "Stage 6: collect vocoder statistics"
    ${python} -m espnet2.bin.rst_vocoder_train \
        --config ${voc_pretrain_config} \
        --train_data_path_and_name_and_type data/train_voc/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_voc/wav.scp,speech_ref1,sound \
        --output_dir ${voc_pretrain_exp} --collect_stats true --ngpu 0
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && ! [[ " ${skip_stages} " =~ [[:space:]]7[[:space:]] ]]; then
    log "Stage 7: pretrain vocoder on ground-truth SSL features"
    ${cuda_cmd} --gpu ${ngpu} ${voc_pretrain_exp}/train.log \
        ${python} -m espnet2.bin.rst_vocoder_train \
        --config ${voc_pretrain_config} \
        --train_data_path_and_name_and_type data/train_voc/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_voc/wav.scp,speech_ref1,sound \
        --train_shape_file ${voc_pretrain_exp}/train/speech_ref1_shape \
        --valid_shape_file ${voc_pretrain_exp}/valid/speech_ref1_shape \
        --output_dir ${voc_pretrain_exp} --ngpu ${ngpu} \
        --multiprocessing_distributed true --unused_parameters true --resume true
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && ! [[ " ${skip_stages} " =~ [[:space:]]8[[:space:]] ]]; then
    log "Stage 8: finetune vocoder on predicted SSL features"
    vocoder_init=${vocoder_init:-${voc_pretrain_exp}/valid.loss_mel.best.pth}
    discriminator_init=${discriminator_init-${vocoder_init}}
    init_opts=(--init_param "${vocoder_init}:vocoder:vocoder")
    if [ -n "${discriminator_init}" ]; then
        init_opts+=(--init_param "${discriminator_init}:discriminator:discriminator")
    fi
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
        --multiprocessing_distributed true --unused_parameters true --resume true
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ] && ! [[ " ${skip_stages} " =~ [[:space:]]9[[:space:]] ]]; then
    if [ -n "${external_vocoder}" ]; then
        vocoder_opts=(--external_vocoder "${external_vocoder}")
    else
        vocoder_exp=${vocoder_exp:-${voc_finetune_exp}}
        vocoder_model_file=${vocoder_model_file:-${vocoder_exp}/valid.loss_mel.best.pth}
        for required_file in "${vocoder_exp}/config.yaml" "${vocoder_model_file}"; do
            [ -f "${required_file}" ] || {
                log "Missing vocoder file ${required_file}: train one (stages 6-8) or set --external_vocoder"
                exit 1
            }
        done
        vocoder_opts=(--vocoder_train_config "${vocoder_exp}/config.yaml"
                      --vocoder_model_file "${vocoder_model_file}")
    fi
    for test_set in ${test_sets}; do
        log "Stage 9: inference (${test_set})"
        ${python} -m espnet2.bin.rst_inference \
            --config ${decode_config} \
            --train_config ${expdir}/config.yaml \
            --model_file ${expdir}/${inference_model} \
            "${vocoder_opts[@]}" \
            --wav_scp data/${test_set}_16k/wav.scp \
            --output_dir ${expdir}/inference_${test_set}
    done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
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
    ${python} pyscripts/utils/show_rst_result.py "${expdir}" > "${expdir}"/RESULTS.md
    cat "${expdir}"/RESULTS.md
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ] && ! [[ " ${skip_stages} " =~ [[:space:]]11[[:space:]] ]]; then
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
    ${python} pyscripts/utils/show_rst_result.py "${expdir}" > "${expdir}"/RESULTS.md
    cat "${expdir}"/RESULTS.md
fi

packed_model="${expdir}/${expdir##*/}_${inference_model%.*}.zip"
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ] && ! [[ " ${skip_stages} " =~ [[:space:]]12[[:space:]] ]]; then
    log "Stage 12: Pack model: ${packed_model}"
    if [ -n "${external_vocoder}" ]; then
        log "Error: an --external_vocoder cannot be packed; pack a vocoder trained by stages 6-8"
        exit 1
    fi
    vocoder_exp=${vocoder_exp:-${voc_finetune_exp}}
    vocoder_model_file=${vocoder_model_file:-${vocoder_exp}/valid.loss_mel.best.pth}
    # The vocoder checkpoint also holds the SSL encoder and the discriminator,
    # which only training uses; pack just the vocoder weights.
    packed_vocoder=${vocoder_exp}/$(basename "${vocoder_model_file}" .pth).vocoder_only.pth
    ${python} pyscripts/utils/extract_rst_vocoder.py "${vocoder_model_file}" "${packed_vocoder}"
    _opts=()
    for f in "${expdir}/RESULTS.md" "${expdir}/images"; do
        if [ -e "${f}" ]; then
            _opts+=(--option "${f}")
        fi
    done
    ${python} -m espnet2.bin.pack rst \
        --train_config "${expdir}/config.yaml" \
        --model_file "${expdir}/${inference_model}" \
        --vocoder_train_config "${vocoder_exp}/config.yaml" \
        --vocoder_model_file "${packed_vocoder}" \
        "${_opts[@]}" \
        --outpath "${packed_model}"
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ] && ! [[ " ${skip_stages} " =~ [[:space:]]13[[:space:]] ]]; then
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#133-publishing-models" && \
    exit 1
    log "Stage 13: Upload model to HuggingFace: ${hf_repo}"

    if [ ! -f "${packed_model}" ]; then
        log "ERROR: ${packed_model} does not exist. Please run stage 12 first."
        exit 1
    fi

    gitlfs=$(git lfs --version 2> /dev/null || true)
    [ -z "${gitlfs}" ] && \
        log "ERROR: You need to install git-lfs first" && \
        exit 1

    dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
    [ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

    if command -v git &> /dev/null; then
        _creator_name="$(git config user.name)"
        _checkout="git checkout $(git show -s --format=%H)"
    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/rst1/ -> foo/rst1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/rst1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=audio-to-audio
    # shellcheck disable=SC2034
    espnet_task=RST
    # shellcheck disable=SC2034
    task_exp=${expdir}
    eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

    this_folder=${PWD}
    cd ${dir_repo}
    if [ -n "$(git status --porcelain)" ]; then
        git add .
        git commit -m "Update model"
    fi
    git push
    cd ${this_folder}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
