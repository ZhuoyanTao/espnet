#!/usr/bin/env bash
set -Eeuo pipefail

DATA_DIR="data"
TRAIN_SET="synth_train"
DEV_SET="synth_dev"
TEST_SETS="synth_test"
EXP_DIR="synth_exp"
NUM_SPK=3
COLLAR=0.0
NGPU=1
DIAR_CFG="conf/train_diar.yaml"
INF_CFG="conf/decode_diar.yaml"
RERUN=false
WIPE_DUMP=true
FORCE_RERUN=false   # run stages even if validation fails

usage() {
  cat <<USAGE
Usage:
  $0 [--data data] [--train_set synth_train] [--valid_set synth_dev] [--test_sets "synth_test"]
     [--expdir synth_exp] [--num_spk 3] [--collar 0.0] [--ngpu 1]
     [--diar_config conf/train_diar.yaml] [--inference_config conf/decode_diar.yaml]
     [--rerun] [--no-wipe-dump] [--force-rerun]

What it does:
  - Cleans utt2spk/segments/wav.scp in data/{synth_train,synth_dev,synth_test}
  - Rebuilds spk2utt and validates each split
  - If --rerun, removes copied dump (unless --no-wipe-dump) and runs diar.sh stage 2..7
  - Won't stop on first validation error; prints a summary at the end.
USAGE
}

trap 'echo "[ERROR] line \$LINENO: command \"\$BASH_COMMAND\" failed." >&2' ERR
log() { echo -e "$(date '+%Y-%m-%dT%H:%M:%S') ($0) $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_DIR="${2:?}"; shift 2 ;;
    --train_set) TRAIN_SET="${2:?}"; shift 2 ;;
    --valid_set) DEV_SET="${2:?}"; shift 2 ;;
    --test_sets) TEST_SETS="${2:?}"; shift 2 ;;
    --expdir) EXP_DIR="${2:?}"; shift 2 ;;
    --num_spk) NUM_SPK="${2:?}"; shift 2 ;;
    --collar) COLLAR="${2:?}"; shift 2 ;;
    --ngpu) NGPU="${2:?}"; shift 2 ;;
    --diar_config) DIAR_CFG="${2:?}"; shift 2 ;;
    --inference_config) INF_CFG="${2:?}"; shift 2 ;;
    --rerun) RERUN=true; shift ;;
    --no-wipe-dump) WIPE_DUMP=false; shift ;;
    --force-rerun) FORCE_RERUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

clean_split() {
  local d="$1"
  [[ -d "$d" ]] || { log "SKIP: $d not found"; return 2; }

  log "Cleaning $d"

  # Normalize utt2spk to two columns, drop junk, sort unique by utt id
  if [[ -f "$d/utt2spk" ]]; then
    awk 'NF>=2{print $1, $2}' "$d/utt2spk" | LC_ALL=C sort -u > "$d/.utt2spk.tmp" || true
    mv -f "$d/.utt2spk.tmp" "$d/utt2spk"
  else
    log "WARN: $d/utt2spk missing"; return 1
  fi

  [[ -f "$d/segments" ]] && LC_ALL=C sort -u "$d/segments" -o "$d/segments"
  [[ -f "$d/wav.scp"   ]] && LC_ALL=C sort -u "$d/wav.scp"   -o "$d/wav.scp"

  # Align utt ids between segments and utt2spk (drop orphans)
  if [[ -f "$d/segments" ]]; then
    comm -23 <(awk '{print $1}' "$d/segments" | LC_ALL=C sort) <(awk '{print $1}' "$d/utt2spk" | LC_ALL=C sort) \
      | xargs -r -I{} sed -i -e '/^{}[[:space:]]/d' "$d/segments"
    comm -23 <(awk '{print $1}' "$d/utt2spk" | LC_ALL=C sort) <(awk '{print $1}' "$d/segments" | LC_ALL=C sort) \
      | xargs -r -I{} sed -i -e '/^{}[[:space:]]/d' "$d/utt2spk"
  fi

  # Rebuild spk2utt + fix + validate (capture status)
  utils/fix_data_dir.sh "$d" >/dev/null || true
  if utils/validate_data_dir.sh --no-feats "$d"; then
    log "VALID: $d"
    return 0
  else
    log "INVALID: $d (see messages above)"
    return 1
  fi
}

# Clean all splits and record status
declare -A STATUS
for split in "$TRAIN_SET" "$DEV_SET" $TEST_SETS; do
  if clean_split "${DATA_DIR}/${split}"; then
    STATUS["$split"]="OK"
  else
    STATUS["$split"]="FAIL"
  fi
done

log "Summary:"
for k in "${!STATUS[@]}"; do
  echo "  $k: ${STATUS[$k]}"
done

any_fail=false
for v in "${STATUS[@]}"; do
  [[ "$v" == "FAIL" ]] && any_fail=true
done

if "$RERUN"; then
  if $any_fail && ! "$FORCE_RERUN"; then
    log "Not re-running diar.sh because some splits failed validation. Use --force-rerun to override."
    exit 1
  fi
  if "$WIPE_DUMP"; then
    for s in "$TRAIN_SET" "$DEV_SET" $TEST_SETS; do
      rm -rf "dump/raw/org/${s}"
    done
  fi
  log "Re-running diar.sh stage 2..7"
  ./diar.sh \
    --collar "${COLLAR}" \
    --stage 2 --stop_stage 7 \
    --train_set "${TRAIN_SET}" \
    --valid_set "${DEV_SET}" \
    --test_sets "${TEST_SETS}" \
    --expdir "${EXP_DIR}" \
    --ngpu "${NGPU}" \
    --diar_config "${DIAR_CFG}" \
    --inference_config "${INF_CFG}" \
    --inference_nj 5 \
    --num_spk "${NUM_SPK}"
fi

log "Done."
