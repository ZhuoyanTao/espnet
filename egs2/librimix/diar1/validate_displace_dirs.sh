#!/usr/bin/env bash
# Validate & finalize ESPnet diarization data dirs for DISPLACE
# Usage: ./validate_displace_dirs.sh [RECIPE_DIR]
# If RECIPE_DIR is omitted, the script tries to infer it from its own location.

set -euo pipefail

# --- locate recipe dir (has utils/, conf/, data/, etc.) ---
if [[ $# -ge 1 ]]; then
  RECIPE_DIR="$(readlink -f "$1")"
else
  # infer from this script's path
  THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  RECIPE_DIR="${THIS_DIR}"
fi

UTILS_DIR="${RECIPE_DIR}/utils"
DATA_DIR="${RECIPE_DIR}/data"

if [[ ! -d "${UTILS_DIR}" ]]; then
  echo "ERROR: utils/ not found under ${RECIPE_DIR}. Pass the correct RECIPE_DIR."
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: data/ not found under ${RECIPE_DIR}."
  exit 1
fi

echo "Recipe dir  : ${RECIPE_DIR}"
echo "Data dir    : ${DATA_DIR}"
echo

# --- process the three splits ---
for s in displace_train displace_dev displace_test; do
  out="${DATA_DIR}/${s}"
  if [[ ! -d "${out}" ]]; then
    echo "WARN: ${out} does not exist; skipping."
    continue
  fi
  if [[ ! -f "${out}/wav.scp" ]]; then
    echo "ERROR: ${out}/wav.scp missing; skipping this split."
    continue
  fi

  # 1) utt2spk: make each utt its own speaker (one wav per utt)
  awk '{print $1, $1}' "${out}/wav.scp" | sort -k1,1 > "${out}/utt2spk"

  # 2) spk2utt
  "${UTILS_DIR}/utt2spk_to_spk2utt.pl" "${out}/utt2spk" > "${out}/spk2utt"

  # 3) RTTM symlink name that ESPnet expects
  if [[ -f "${out}/rttm" && ! -e "${out}/espnet_rttm" ]]; then
    (cd "${out}" && ln -s rttm espnet_rttm)
  fi

  echo "== ${s} =="
  echo "wav:  $(wc -l < "${out}/wav.scp")"
  if [[ -e "${out}/espnet_rttm" ]]; then
    echo "rttm: $(grep -vc '^[[:space:]]*$' "${out}/espnet_rttm")"
  else
    echo "rttm: (missing ${out}/espnet_rttm)"
  fi

  # 4) validate (no feats/text in diarization dirs)
  "${UTILS_DIR}/validate_data_dir.sh" --no-feats --no-text --no-spk-sort "${out}" || true

  # 5) fix any sorting/glitches quietly
  "${UTILS_DIR}/fix_data_dir.sh" "${out}" >/dev/null
  echo
done

echo "All done ✅"
