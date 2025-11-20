#!/usr/bin/env bash
set -euo pipefail

DATADIR="data"
SPLITS=("synth_train" "synth_dev" "synth_test")

usage() {
  cat <<USAGE
Usage: $0 [--dir data] [--splits "synth_train synth_dev synth_test"]
Cleans + sorts wav.scp, segments, utt2spk and runs fix/validate when available.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DATADIR="${2:?}"; shift 2 ;;
    --splits) IFS=' ' read -r -a SPLITS <<< "${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

clean_file() {
  local in="$1" out="$2" kind="$3"
  if [[ ! -f "$in" ]]; then
    echo "  - Skip $kind (missing: $in)"
    return 0
  fi
  case "$kind" in
    utt2spk)
      tr -d '\r' < "$in" | awk 'NF>=2{print $1,$2}' | LC_ALL=C sort -k1,1 -u > "$out"
      ;;
    segments)
      tr -d '\r' < "$in" | awk 'NF>=4{print $1,$2,$3,$4}' | LC_ALL=C sort -k1,1 -u > "$out"
      ;;
    wav.scp)
      tr -d '\r' < "$in" | awk 'NF>=2{print $1,$2}' | LC_ALL=C sort -k1,1 -u > "$out"
      ;;
    *) cp -f "$in" "$out" ;;
  esac
  mv -f "$out" "$in"
  echo "  - Cleaned $kind"
}

clean_one() {
  local d="$1"
  [[ -d "$d" ]] || { echo "Skip $d (not found)"; return 0; }
  echo "==> Cleaning $d"

  clean_file "$d/utt2spk" "$d/.utt2spk.tmp" "utt2spk"
  clean_file "$d/segments" "$d/.segments.tmp" "segments"
  clean_file "$d/wav.scp"  "$d/.wav.scp.tmp" "wav.scp"

  if command -v utils/fix_data_dir.sh >/dev/null 2>&1; then
    utils/fix_data_dir.sh "$d" >/dev/null || true
  fi

  # (Re)build spk2utt if utt2spk exists
  if [[ -f "$d/utt2spk" ]] && command -v utils/utt2spk_to_spk2utt.pl >/dev/null 2>&1; then
    utils/utt2spk_to_spk2utt.pl "$d/utt2spk" > "$d/spk2utt" || true
  fi

  if command -v utils/validate_data_dir.sh >/dev/null 2>&1; then
    utils/validate_data_dir.sh --no-feats "$d" || true
  fi
}

for s in "${SPLITS[@]}"; do
  clean_one "${DATADIR}/${s}"
done

echo "Done."
