#!/usr/bin/env bash
# Build prefix rows with "so far" and whole-clip (<metric>_full) labels from
# whole clips. Input per split: data/<split>_whole/wav.scp (whole clips at any
# sampling rate), optionally ref_wav.scp (time-aligned clean references of some
# clips) and metric.scp (labels from elsewhere, e.g. listening-test MOS; rows
# keyed <utt> label the whole clip, rows keyed <utt>__p<ms> its prefix).
# See README.md for the sources and their preparation.
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
    local a=$1 b
    for b in "$@"; do [ "${b}" -lt "${a}" ] && a=${b}; done
    echo "${a}"
}
SECONDS=0

stage=1
stop_stage=100
splits="train valid dev"  # data/<split>_whole/ prepared beforehand
train_set=train           # split whose labels extend the token info
nj=16                     # parallel jobs for audio and VERSA
fs=16000
prefix_seconds="2 4 6 8"
versa_config=conf/versa_so_far.yaml        # labels measured on prefixes
whole_versa_config=conf/versa_so_far.yaml  # labels measured on whole clips
versa_opts=--gpu-only     # --gpu-only or --cpu-only for scripts/utils/versa_eval.sh
token_info=               # metric token info JSON to start from (see README.md)
init_model=               # optional model state dict extended like the token info
drop_metrics="sar sir"    # VERSA outputs the released model was not trained on
silence_sec=1.0           # silent reference clip given to every row

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# Merge labels for the keys of a wav.scp: <keys> <out> <input>...; an input is
# a VERSA score directory (result/result.*.txt) or a metric.scp, later inputs
# win, non-finite values are dropped and keys without labels get {}.
merge_labels() {
    python3 - "$@" << 'EOF'
import glob, json, math, os, sys


def read(src):
    if os.path.isdir(src):  # VERSA: one JSON object with a "key" per line
        for path in sorted(glob.glob(f"{src}/result/result.*.txt")):
            for line in filter(str.strip, open(path, encoding="utf-8")):
                row = json.loads(line)
                yield row.pop("key"), row
    elif os.path.isfile(src):  # metric.scp
        for line in filter(str.strip, open(src, encoding="utf-8")):
            key, value = line.split(maxsplit=1)
            yield key, json.loads(value)


rows = {line.split()[0]: {} for line in open(sys.argv[1]) if line.strip()}
for src in sys.argv[3:]:
    for key, row in read(src):
        if key in rows:
            rows[key].update(
                (m, v) for m, v in row.items()
                if isinstance(v, str) or isinstance(v, (int, float)) and math.isfinite(v)
            )
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for key in sorted(rows):
        f.write(f"{key} {json.dumps(rows[key], ensure_ascii=False)}\n")
EOF
}

# Cut prefixes of a wav.scp in parallel: <wav.scp> <out_dir>
cut_prefixes() {
    local scp=$1 dir=$2 n i split_scps=""
    n=$(min "${nj}" "$(wc -l < "${scp}")")
    mkdir -p "${dir}/log"
    for i in $(seq "${n}"); do split_scps+=" ${dir}/log/wav.${i}.scp"; done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${scp}" ${split_scps}
    # shellcheck disable=SC2086
    ${train_cmd} JOB=1:"${n}" "${dir}/log/prefix.JOB.log" \
        python3 pyscripts/utils/make_prefix_wav_scp.py "${dir}/log/wav.JOB.scp" \
            "${dir}/split.JOB" --prefix_seconds ${prefix_seconds} --audio_format flac
    for f in wav.scp utt2num_samples; do
        for i in $(seq "${n}"); do cat "${dir}/split.${i}/${f}"; done | sort > "${dir}/${f}"
    done
}

# Score a wav.scp with VERSA: <wav.scp> <ref_wav.scp or None> <config> <out_dir>
score() {
    local scp=$1 ref=$2 config=$3 dir=$4 part
    mkdir -p "${dir}"
    : > "${dir}/gt.scp"
    if [ "${ref}" != None ]; then
        utils/filter_scp.pl "${scp}" "${ref}" > "${dir}/gt.scp"
    fi
    utils/filter_scp.pl "${dir}/gt.scp" "${scp}" > "${dir}/with_ref.scp"
    utils/filter_scp.pl --exclude "${dir}/gt.scp" "${scp}" > "${dir}/no_ref.scp"
    for part in with_ref no_ref; do
        [ -s "${dir}/${part}.scp" ] || continue
        ref=None
        [ ${part} = with_ref ] && ref=${dir}/gt.scp
        # shellcheck disable=SC2086
        scripts/utils/versa_eval.sh "${dir}/${part}.scp" "${ref}" None "${dir}/${part}" \
            "$(min "${nj}" "$(wc -l < "${dir}/${part}.scp")")" "${config}" ${versa_opts}
    done
}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Resample whole clips and references to ${fs} Hz"
    for s in ${splits}; do
        src=data/${s}_whole
        if [ ! -f "${src}/wav.scp" ]; then
            log "Error: ${src}/wav.scp is missing; prepare the whole clips first (README.md)"
            exit 1
        fi
        for f in wav.scp ref_wav.scp; do
            [ -f "${src}/${f}" ] || continue
            sort -o "${src}/${f}" "${src}/${f}"
            _opts=()
            [ ${f} = ref_wav.scp ] && _opts=(--suffix .ref --out_filename ref_wav.scp)
            scripts/audio/format_wav_scp.sh --nj "$(min "${nj}" "$(wc -l < "${src}/${f}")")" \
                --cmd "${train_cmd}" --audio-format flac --fs "${fs}" ${_opts[@]+"${_opts[@]}"} \
                "${src}/${f}" "data/${s}_${fs}"
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Cut the ${prefix_seconds} s prefixes of clips and references"
    for s in ${splits}; do
        d=data/${s}_${fs}
        cut_prefixes "${d}/wav.scp" "${d}/prefix"
        if [ -f "${d}/ref_wav.scp" ]; then
            cut_prefixes "${d}/ref_wav.scp" "${d}/prefix_ref"
        fi
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Label whole clips and prefixes with VERSA"
    for s in ${splits}; do
        d=data/${s}_${fs}
        ref=None
        [ -f "${d}/ref_wav.scp" ] && ref=${d}/ref_wav.scp
        score "${d}/wav.scp" "${ref}" "${whole_versa_config}" "${d}/versa_whole"
        ref=None
        [ -f "${d}/prefix_ref/wav.scp" ] && ref=${d}/prefix_ref/wav.scp
        score "${d}/prefix/wav.scp" "${ref}" "${versa_config}" "${d}/versa_prefix"
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Add whole-clip <metric>_full targets and build data/<split>"
    mkdir -p data/local
    python3 -c 'import sys, numpy, soundfile; fs = int(sys.argv[2])
soundfile.write(sys.argv[3], numpy.zeros(round(float(sys.argv[1]) * fs), "int16"), fs)' \
        "${silence_sec}" "${fs}" data/local/silence.wav
    for s in ${splits}; do
        d=data/${s}_${fs}
        out=data/${s}
        mkdir -p "${out}"
        merge_labels "${d}/wav.scp" "${d}/whole_metric.scp" \
            "${d}/versa_whole/with_ref" "${d}/versa_whole/no_ref" "data/${s}_whole/metric.scp"
        merge_labels "${d}/prefix/wav.scp" "${d}/prefix_metric.scp" \
            "${d}/versa_prefix/with_ref" "${d}/versa_prefix/no_ref" "data/${s}_whole/metric.scp"
        python3 pyscripts/utils/prep_metric_id.py "${d}/whole_metric.scp" "${d}/whole_metrics"
        # shellcheck disable=SC2046
        python3 pyscripts/utils/add_full_metric_targets.py "${d}/prefix_metric.scp" \
            "${d}/whole_metric.scp" "${d}/prefix_full" --metrics $(cat "${d}/whole_metrics")
        sort "${d}/whole_metric.scp" "${d}/prefix_full/metric.scp" > "${out}/metric.scp"
        sort "${d}/wav.scp" "${d}/prefix/wav.scp" > "${out}/wav.scp"
        # Each row's "speaker" is its clip, so prefix keys sort after it.
        awk '{u = $1; sub(/__p[0-9]+$/, "", u); print $1, u}' "${out}/wav.scp" > "${out}/utt2spk"
        utils/utt2spk_to_spk2utt.pl "${out}/utt2spk" > "${out}/spk2utt"
        awk -v f="${PWD}/data/local/silence.wav" '{print $1, f}' "${out}/wav.scp" \
            > "${out}/ref_wav.scp"
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Extend ${token_info} with new metrics of ${train_set}; write metric2id"
    if [ -z "${token_info}" ]; then
        log "Error: give --token_info (README.md: token info)"
        exit 1
    fi
    # Unknown numerical metrics get new tokens; unknown categorical ones (e.g.
    # ASR hypotheses) are dropped below; add those with extend_metric_vocab.py.
    python3 - "${token_info}" "data/${train_set}/metric.scp" "${drop_metrics}" \
        > data/local/new_metrics << 'EOF'
import json, sys

known = json.load(open(sys.argv[1]))["tokenizer"]
skip = {n + s for n in sys.argv[3].split() for s in ("", "_full")}
new = set()
for line in open(sys.argv[2], encoding="utf-8"):
    for name, value in json.loads(line.split(maxsplit=1)[1]).items():
        if name not in known:
            (skip if isinstance(value, str) else new).add(name)
print(" ".join(sorted(new - skip)))
EOF
    read -r -a numerical < data/local/new_metrics
    _opts=()
    if [ ${#numerical[@]} -gt 0 ]; then
        _opts+=(--numerical "${numerical[@]}" --metric_scp "data/${train_set}/metric.scp")
    fi
    if [ -n "${init_model}" ]; then
        _opts+=(--checkpoint "${init_model}" --output_checkpoint data/init_model.pth)
    fi
    log "New numerical metrics: ${#numerical[@]}"
    python3 pyscripts/utils/extend_metric_vocab.py "${token_info}" data/token_info.json \
        ${_opts[@]+"${_opts[@]}"}
    # Keep the labels the token info encodes; metric2id lists those of train.
    # shellcheck disable=SC2086
    python3 - data/token_info.json "${drop_metrics}" "${train_set}" ${splits} << 'EOF'
import json, sys

known = json.load(open(sys.argv[1]))["tokenizer"]
drop = {n + s for n in sys.argv[2].split() for s in ("", "_full")}
for split in sys.argv[4:]:
    path, names = f"data/{split}/metric.scp", set()
    lines = [line.split(maxsplit=1) for line in open(path, encoding="utf-8")]
    with open(path, "w", encoding="utf-8") as f:
        for key, value in lines:
            row = {
                m: v
                for m, v in json.loads(value).items()
                if m in known and m not in drop and (not isinstance(v, str) or v in known[m])
            }
            names.update(row)
            f.write(f"{key} {json.dumps(row, ensure_ascii=False)}\n")
    if split == sys.argv[3]:
        open("data/metric2id", "w").writelines(f"{n}\n" for n in sorted(names))
EOF
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
