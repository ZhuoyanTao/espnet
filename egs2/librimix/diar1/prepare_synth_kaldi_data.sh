#!/usr/bin/env bash
set -euo pipefail

# Default options
SRC=""
PREFIX="synth"
TRAIN_PCT=80
DEV_PCT=10
TEST_PCT=10
SEED=17
OVERWRITE=false

usage() {
  cat <<USAGE
Usage:
  $0 --src /path/to/folder [--prefix synth] [--train 80 --dev 10 --test 10] [--seed 17] [--overwrite]

What it does:
  - Finds *.wav under --src and requires matching .rttm with same basename
  - Creates data/{PREFIX}_{train,dev,test} with wav.scp, segments, utt2spk
  - Deterministic shuffle with --seed, then 80/10/10 split by default
USAGE
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="${2:?}"; shift 2 ;;
    --prefix) PREFIX="${2:?}"; shift 2 ;;
    --train) TRAIN_PCT="${2:?}"; shift 2 ;;
    --dev) DEV_PCT="${2:?}"; shift 2 ;;
    --test) TEST_PCT="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --overwrite) OVERWRITE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

[[ -z "${SRC}" ]] && { echo "ERROR: --src is required"; usage; exit 2; }
[[ -d "${SRC}" ]] || { echo "ERROR: --src not found: ${SRC}"; exit 2; }

# Check split adds to 100
TOTAL=$((TRAIN_PCT + DEV_PCT + TEST_PCT))
[[ "${TOTAL}" -eq 100 ]] || { echo "ERROR: splits must sum to 100 (got ${TOTAL})"; exit 2; }

mkdir -p data

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

# Collect IDs with matching RTTM
mapfile -t WAVS < <(find "${SRC}" -maxdepth 1 -type f -name '*.wav' | sort)
IDS=()
SKIPPED=0
for w in "${WAVS[@]}"; do
  base="$(basename "${w}" .wav)"
  rttm="${SRC}/${base}.rttm"
  if [[ ! -f "${rttm}" ]]; then
    echo "WARN: Missing RTTM for ${base}.wav -> ${base}.rttm"
    SKIPPED=$((SKIPPED+1))
    continue
  fi
  IDS+=("${base}")
done
if [[ "${SKIPPED}" -gt 0 ]]; then
  echo "NOTE: ${SKIPPED} file(s) skipped due to missing RTTM."
fi
[[ "${#IDS[@]}" -gt 0 ]] || { echo "ERROR: No valid wav+rttm pairs found in ${SRC}"; exit 2; }

# Deterministic shuffle
printf '%s\n' "${IDS[@]}" > "${TMPDIR}/ids.txt"
SEED_ENV="${SEED}" IDS_FILE="${TMPDIR}/ids.txt" python3 - > "${TMPDIR}/all_ids.txt" <<'PY'
import os, random
ids_path = os.environ["IDS_FILE"]
seed = int(os.environ.get("SEED_ENV", "17"))
with open(ids_path) as f:
    ids = [line.strip() for line in f if line.strip()]
random.seed(seed)
random.shuffle(ids)
for x in ids:
    print(x)
PY


N="$(wc -l < "${TMPDIR}/all_ids.txt")"
n_train=$(( N * TRAIN_PCT / 100 ))
n_dev=$(( N * DEV_PCT / 100 ))
n_test=$(( N - n_train - n_dev ))

head -n "${n_train}" "${TMPDIR}/all_ids.txt" > "${TMPDIR}/train_ids.txt"
tail -n +"$((n_train+1))" "${TMPDIR}/all_ids.txt" | head -n "${n_dev}" > "${TMPDIR}/dev_ids.txt"
tail -n "${n_test}" "${TMPDIR}/all_ids.txt" > "${TMPDIR}/test_ids.txt"

make_dir() {
  local d="$1"
  if [[ -d "${d}" && "${OVERWRITE}" == "true" ]]; then
    rm -rf "${d}"
  fi
  mkdir -p "${d}"
}

build_split() {
  local split_ids="$1"
  local outdir="$2"
  make_dir "${outdir}"

  : > "${outdir}/wav.scp"
  : > "${outdir}/segments"
  : > "${outdir}/utt2spk"

  while IFS= read -r id; do
    [[ -z "${id}" ]] && continue
    local_wav="${SRC}/${id}.wav"
    local_rttm="${SRC}/${id}.rttm"

    # reco id = basename (no spaces)
    reco="${id}"

    # wav.scp
    echo "${reco} ${local_wav}" >> "${outdir}/wav.scp"

    # Parse RTTM -> segments + utt2spk
    # Format: SPEAKER <file-id> <chan> <tbeg> <tdur> <ortho> <stype> <name> <conf>
    # We take <name> as speaker, create utt: {reco}-{start_ms}-{end_ms}
python3 - "${local_rttm}" "${reco}" >> "${outdir}/segments" 2>> "${outdir}/utt2spk" <<'PY'
import sys, pathlib
from decimal import Decimal, ROUND_HALF_UP

rttm_path = pathlib.Path(sys.argv[1])
reco = sys.argv[2]

def ms(x):
    d = Decimal(x).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
    return int((d * 1000).to_integral_value(rounding=ROUND_HALF_UP))

with rttm_path.open() as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if parts[0] != "SPEAKER":
            continue
        tbeg = float(parts[3])
        tdur = float(parts[4])
        spk  = parts[7] if len(parts) > 7 else "spk"
        start = tbeg
        end = tbeg + tdur
        utt_id = f"{reco}-{ms(start)}-{ms(end)}"
        print(f"{utt_id} {reco} {start:.3f} {end:.3f}")
        print(f"{utt_id} {spk}", file=sys.stderr)
PY
if [[ $? -ne 0 ]]; then
  echo "WARN: failed to parse ${local_rttm}, skipping ${reco}" >&2
  continue
fi

  done < "${split_ids}"

  # Fix & validate (if Kaldi utils are present)
  if command -v utils/fix_data_dir.sh >/dev/null 2>&1; then
    utils/fix_data_dir.sh "${outdir}" >/dev/null
  fi
}

build_split "${TMPDIR}/train_ids.txt" "data/${PREFIX}_train"
build_split "${TMPDIR}/dev_ids.txt"   "data/${PREFIX}_dev"
build_split "${TMPDIR}/test_ids.txt"  "data/${PREFIX}_test"

echo "Done. Wrote:"
echo "  data/${PREFIX}_train/{wav.scp,segments,utt2spk}"
echo "  data/${PREFIX}_dev/{wav.scp,segments,utt2spk}"
echo "  data/${PREFIX}_test/{wav.scp,segments,utt2spk}"
