#!/usr/bin/env bash
set -euo pipefail

# Rebuild data/synth_{train,dev,test} from a folder of *.wav/*.rttm pairs.
# Must be run from: espnet/egs2/librimix/diar1

# Defaults
SRC=""                 # e.g. /work/nvme/bbjs/ttao3/datasets/synth_out_oct17_1
OUTDIR="data"
FS="8k"                # 8k | 16k
NUM_SPK=3
SEED=17
PREFIX="synth"         # final dirs: ${OUTDIR}/${PREFIX}_{train,dev,test}

usage() {
  cat <<USAGE
Usage: $0 --src /path/to/wav_rttm_dir [--outdir data] [--fs 8k|16k] [--num-spk 3] [--seed 17] [--prefix synth]

What it does:
  1) Removes existing ${OUTDIR}/{${PREFIX}_train,${PREFIX}_dev,${PREFIX}_test} and temporary ${OUTDIR}/{train,dev,test}
  2) Generates a manifest (train/dev/test 80/10/10, deterministic with --seed) that SKIPS wavs with no SPEAKER lines
  3) Calls ./local/data_synth.sh (which uses local/prepare_synth_from_rttm.py)
  4) Renames train/dev/test -> ${PREFIX}_train/${PREFIX}_dev/${PREFIX}_test
  5) Runs utils/fix_data_dir.sh and utils/validate_data_dir.sh on each split
USAGE
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="${2:?}"; shift 2 ;;
    --outdir) OUTDIR="${2:?}"; shift 2 ;;
    --fs) FS="${2:?}"; shift 2 ;;
    --num-spk) NUM_SPK="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --prefix) PREFIX="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# Basic checks
[[ -n "${SRC}" ]] || { echo "ERROR: --src is required"; usage; exit 2; }
[[ -d "${SRC}" ]] || { echo "ERROR: --src not found: ${SRC}"; exit 2; }
[[ -x "./local/data_synth.sh" ]] || { echo "ERROR: run from diar1/ (missing ./local/data_synth.sh)"; exit 2; }
command -v utils/fix_data_dir.sh >/dev/null || { echo "ERROR: missing utils/fix_data_dir.sh in PATH"; exit 2; }
command -v utils/validate_data_dir.sh >/dev/null || { echo "ERROR: missing utils/validate_data_dir.sh in PATH"; exit 2; }

echo "==> Cleaning old dirs in ${OUTDIR}"
rm -rf "${OUTDIR}/${PREFIX}_train" "${OUTDIR}/${PREFIX}_dev" "${OUTDIR}/${PREFIX}_test" \
       "${OUTDIR}/train" "${OUTDIR}/dev" "${OUTDIR}/test"

echo "==> Building ${OUTDIR}/{train,dev,test} from ${SRC}"
./local/data_synth.sh \
  --manifest <(python3 - <<PY
import glob, os, random, sys
random.seed(${SEED})
wavs = sorted(glob.glob('${SRC.rstrip("/")}/*.wav'))
print('split,rec_id,wav_path')
idx = list(range(len(wavs))); random.shuffle(idx)
N=len(idx); ntr=N*80//100; nd=N*10//100

def rec_from_rttm(w):
    try:
        with open(w[:-4]+'.rttm') as f:
            for ln in f:
                if ln.startswith('SPEAKER'):
                    return ln.split()[1]
    except Exception:
        pass
    return None

k=0
for j in idx:
    w = wavs[j]
    rec = rec_from_rttm(w)
    if rec is None:
        # Skip files with no SPEAKER lines to avoid "no RTTM entries" warnings
        continue
    split = 'train' if k<ntr else ('dev' if k<ntr+nd else 'test')
    print(f'{split},{rec},{w}')
    k+=1
PY
) \
  --rttm "${SRC}" \
  --fs "${FS}" \
  --num_spk "${NUM_SPK}" \
  --outdir "${OUTDIR}"

echo "==> Renaming splits to ${PREFIX}_*"
for s in train dev test; do
  if [[ -d "${OUTDIR}/${s}" ]]; then
    mv -f "${OUTDIR}/${s}" "${OUTDIR}/${PREFIX}_${s}"
  fi
done

echo "==> Fixing and validating"
for d in "${OUTDIR}/${PREFIX}_train" "${OUTDIR}/${PREFIX}_dev" "${OUTDIR}/${PREFIX}_test"; do
  [[ -d "$d" ]] || continue
  utils/fix_data_dir.sh "$d" >/dev/null
  utils/validate_data_dir.sh --no-feats "$d"
done

# Remove any leftover temporary train/dev/test if they still exist
rm -rf "${OUTDIR}/train" "${OUTDIR}/dev" "${OUTDIR}/test"

echo "==> Done. Wrote:"
echo "    ${OUTDIR}/${PREFIX}_train {wav.scp,segments,utt2spk,spk2utt,rttm}"
echo "    ${OUTDIR}/${PREFIX}_dev   {wav.scp,segments,utt2spk,spk2utt,rttm}"
echo "    ${OUTDIR}/${PREFIX}_test  {wav.scp,segments,utt2spk,spk2utt,rttm}"
