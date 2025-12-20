#!/usr/bin/env bash
# Sync UniVERSA-related code from old messy repo into this clean repo.
# Safe to run multiple times.

set -euo pipefail

# ---- Paths (change if your layout ever changes) ----
OLD_REPO="/work/nvme/bbjs/ttao3/espnet"
NEW_REPO="/work/nvme/bbjs/ttao3/universa/espnet"

echo "[INFO] OLD_REPO = ${OLD_REPO}"
echo "[INFO] NEW_REPO = ${NEW_REPO}"

# Sanity checks
if [ ! -d "${OLD_REPO}" ]; then
  echo "[ERROR] OLD_REPO does not exist: ${OLD_REPO}" >&2
  exit 1
fi

if [ ! -d "${NEW_REPO}" ]; then
  echo "[ERROR] NEW_REPO does not exist: ${NEW_REPO}" >&2
  exit 1
fi

# Move into NEW_REPO so relative paths are correct
cd "${NEW_REPO}"

# Ensure target directories exist
mkdir -p egs2/universa_unite
mkdir -p espnet2/universa
mkdir -p espnet2/bin
mkdir -p espnet2/fileio
mkdir -p espnet2/tasks

echo "[INFO] Syncing UniVERSA recipe (uni_versa1)..."
rsync -av \
  "${OLD_REPO}/egs2/universa_unite/uni_versa1/" \
  "egs2/universa_unite/uni_versa1/"

echo "[INFO] Syncing espnet2/universa modules..."
rsync -av \
  "${OLD_REPO}/espnet2/universa/" \
  "espnet2/universa/"

echo "[INFO] Syncing train/inference entry points..."
rsync -av \
  "${OLD_REPO}/espnet2/bin/universa_train.py" \
  "${OLD_REPO}/espnet2/bin/universa_inference.py" \
  "espnet2/bin/"

echo "[INFO] Syncing metric_scp reader..."
rsync -av \
  "${OLD_REPO}/espnet2/fileio/metric_scp.py" \
  "espnet2/fileio/"

echo "[INFO] Syncing universa task definition..."
rsync -av \
  "${OLD_REPO}/espnet2/tasks/universa.py" \
  "espnet2/tasks/"

# -------------------------------------------------------------------
# OPTIONAL: reuse old EXP / DUMP via symlinks (commented out by default)
# -------------------------------------------------------------------
# If you want to reuse your heavy experiment folders from the old repo
# without copying them (and without tracking them in git), uncomment:

#: <<'SYMLINK_BLOCK'
# echo "[INFO] Creating symlinks to old exp/ and dump/ (if they exist)..."
#
# OLD_RECIPE_DIR="${OLD_REPO}/egs2/universa_unite/uni_versa1"
# NEW_RECIPE_DIR="${NEW_REPO}/egs2/universa_unite/uni_versa1"
#
# cd "${NEW_RECIPE_DIR}"
#
# if [ -d "${OLD_RECIPE_DIR}/exp" ] && [ ! -e "exp" ]; then
#   ln -s "${OLD_RECIPE_DIR}/exp" exp
#   echo "[INFO] Created symlink: exp -> ${OLD_RECIPE_DIR}/exp"
# fi
#
# if [ -d "${OLD_RECIPE_DIR}/dump" ] && [ ! -e "dump" ]; then
#   ln -s "${OLD_RECIPE_DIR}/dump" dump
#   echo "[INFO] Created symlink: dump -> ${OLD_RECIPE_DIR}/dump"
# fi
# SYMLINK_BLOCK

echo "[INFO] Sync completed successfully."
