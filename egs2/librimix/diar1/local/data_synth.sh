#!/usr/bin/env bash
set -euo pipefail

. ./path.sh
. ./cmd.sh

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

manifest=            # CSV: split,rec_id,wav_path
rttm=                # Either a single RTTM file or a directory of *.rttm
fs=8k                # 8k or 16k
num_spk=3            # target max speakers per recording
outdir=data          # where to write Kaldi dirs

. utils/parse_options.sh || exit 1

if [ -z "${manifest}" ] || [ -z "${rttm}" ]; then
  echo "Usage: $0 --manifest synth_manifest.csv --rttm /path/to/all.rttm [--fs 8k|16k] [--num_spk 3] [--outdir data]"
  exit 2
fi

mkdir -p "${outdir}"

python3 local/prepare_synth_from_rttm.py \
  --manifest "${manifest}" \
  --rttm "${rttm}" \
  --fs "${fs}" \
  --num_spk "${num_spk}" \
  --outdir "${outdir}"

# clean & validate like your LibriMix script
for d in train dev test; do
  if [ -d "${outdir}/${d}" ]; then
    utils/fix_data_dir.sh "${outdir}/${d}"
  fi
done

log "local/data_synth.sh done."
