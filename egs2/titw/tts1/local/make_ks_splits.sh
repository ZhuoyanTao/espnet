#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   local/make_ks_from_master.sh \
#     data/titw_easy/bonafide_metadata_cfg_v3 \
#     ks_spk_train.txt ks_spk_dev.txt ks_spk_test.txt \
#     data/titw_easy_ks_train data/titw_easy_ks_dev data/titw_easy_ks_test
#
# Creates each OUT dir with: wav.scp, utt2spk, spk2utt, text, utt2num_samples

MASTER=${1:?}
SPK_TRAIN=${2:?}
SPK_DEV=${3:?}
SPK_TEST=${4:?}
OUT_TRAIN=${5:?}
OUT_DEV=${6:?}
OUT_TEST=${7:?}

for f in wav.scp utt2spk text; do
  [[ -f "${MASTER}/${f}" ]] || { echo "Missing ${MASTER}/${f}" >&2; exit 1; }
done

filter_split () {
  local MASTER=$1
  local SPKLIST=$2
  local OUT=$3
  mkdir -p "${OUT}"
  echo "[KS] Build ${OUT} from ${MASTER} using ${SPKLIST}"

  # Filter utt2spk by speaker list
  awk 'BEGIN{FS=OFS=" "}
       NR==FNR{allow[$1]=1; next}
       allow[$2]{print $0}' "${SPKLIST}" "${MASTER}/utt2spk" > "${OUT}/utt2spk"

  # Collect kept utts
  cut -d" " -f1 "${OUT}/utt2spk" | sort > "${OUT}/_kept"

  # Filter wav.scp and text
  utils/filter_scp.pl "${OUT}/_kept" "${MASTER}/wav.scp" > "${OUT}/wav.scp"
  utils/filter_scp.pl "${OUT}/_kept" "${MASTER}/text"    > "${OUT}/text"

  # Regenerate spk2utt
  utils/utt2spk_to_spk2utt.pl < "${OUT}/utt2spk" > "${OUT}/spk2utt"

  # utt2num_samples (optional, fast header read; skips pipes)
  if command -v python3 >/dev/null 2>&1; then
    python3 - << 'PY' "${OUT}/wav.scp" > "${OUT}/utt2num_samples" || true
import sys, soundfile as sf, os
scp=sys.argv[1]
with open(scp, 'r', encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line: continue
        if '|' in line:  # command; skip
            continue
        utt, path = line.split(maxsplit=1)
        try:
            info = sf.info(path)
            print(f"{utt} {info.frames}")
        except Exception:
            pass
PY
  fi

  rm -f "${OUT}/_kept"

  echo "[KS] Counts for ${OUT}:"
  echo -n "  utts:     "; wc -l < "${OUT}/utt2spk" || true
  echo -n "  speakers: "; awk '{print $2}' "${OUT}/utt2spk" | sort -u | wc -l || true
}

filter_split "${MASTER}" "${SPK_TRAIN}" "${OUT_TRAIN}"
filter_split "${MASTER}" "${SPK_DEV}"   "${OUT_DEV}"
filter_split "${MASTER}" "${SPK_TEST}"  "${OUT_TEST}"

echo "[KS] Done."
