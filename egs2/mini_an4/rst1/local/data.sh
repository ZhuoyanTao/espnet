#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100
an4_root=./downloads/an4

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Untar downloads.tar.gz"
    if [ ! -e downloads/ ]; then
        tar -xvf downloads.tar.gz
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/{train,test}
    if [ ! -f ${an4_root}/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi
    python3 local/data_prep.py ${an4_root} sph2pipe
    for x in test train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    done
    # Feature-predictor sets: one utterance for validation, the rest for training.
    utils/subset_data_dir.sh --first data/train 1 data/dev_fp
    n=$(($(wc -l < data/train/text) - 1))
    utils/subset_data_dir.sh --last data/train ${n} data/train_fp
    # Vocoder sets must be 48 kHz: an4 is 16 kHz, so upsample. The audio is
    # what it is; the point is to exercise the 48 kHz code path.
    for split in train dev; do
        scripts/audio/format_wav_scp.sh --nj 1 --cmd "${train_cmd}" \
            --fs 48000 --audio-format wav \
            "data/${split}_fp/wav.scp" "data/${split}_voc"
    done
    # Noise pool for the online degradation.
    mkdir -p data/noise_pool
    for f in downloads/noise/*.wav; do
        ln -sf "$(realpath "${f}")" "data/noise_pool/$(basename "${f}")"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Tiny w2v-BERT 2.0 stand-in for the SSL encoder"
    # The released facebook/w2v-bert-2.0 is 2.3 GB; the recipe only needs a
    # model with its interface, so write a two-layer, 32-dimensional one with
    # random weights. The fbank front end (80 mels x 2 stacked frames = 160
    # inputs) is computed in the model, so no processor download is needed.
    python3 - <<PY
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

config = Wav2Vec2BertConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    output_hidden_size=32,
    conv_depthwise_kernel_size=31,
    feature_projection_input_dim=160,
    num_adapter_layers=1,
    adapter_kernel_size=3,
    adapter_stride=2,
    vocab_size=32,
)
Wav2Vec2BertModel(config).save_pretrained("downloads/tiny_w2v_bert2")
print("wrote downloads/tiny_w2v_bert2")
PY
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
