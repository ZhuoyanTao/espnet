#!/usr/bin/env bash
# =============================================================================
# run_distortion_inference.sbatch
#
# CPU-only SLURM job: wraps uni_versa.sh (stages 9-10) to run inference on
# all distorted and clean prefix datasets for one model checkpoint.
#
# Inference is CPU-parallelised internally by uni_versa.sh via --inference_nj.
# Do NOT request a GPU partition.  Do NOT set CUDA_VISIBLE_DEVICES.
# Do NOT pass --gpu_inference true.
#
# Submit one job per model:
#   sbatch --export=ALL,MODEL=ANCHOR run_distortion_inference.sbatch
#   sbatch --export=ALL,MODEL=ARECHO run_distortion_inference.sbatch
#
# Or edit MODEL_TAG below and submit twice with different copies.
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM directives  (CPU partition — no GPU requested)
# ---------------------------------------------------------------------------
#SBATCH --job-name=dist_infer_cpu
#SBATCH --account=bbvf-delta-cpu           # <-- your Delta CPU allocation
#SBATCH --partition=cpu                    # Delta CPU partition (no GPU)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128               # matches --inference_nj 128
#SBATCH --mem=256G                        # ~2 GB per parallel inference job
#SBATCH --time=04:00:00                   # CPU inference is slower; 4 h ceiling
#SBATCH --output=logs/%x_%j.out          # create logs/ dir before submitting
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# !! Edit these variables to match your experiment !!
# ---------------------------------------------------------------------------

# Root of the ESPnet recipe (where uni_versa.sh lives)
RECIPE_DIR="/project/shrikann_35/terryt"

# -- Model A: ANCHOR --------------------------------------------------------
# Duplicate this block (or use --export=MODEL=ANCHOR) for the second model.
MODEL_TAG="${MODEL:-ANCHOR}"    # override with: sbatch --export=ALL,MODEL=ARECHO ...

case "${MODEL_TAG}" in
  ANCHOR)
    TRAIN_CONFIG="conf/train_aruniversa_prefix_full.yaml"
    UNIVERSA_EXP="exp/universa_train_aruniversa_prefix_full_raw_fs16000_defer_full_metatrue.bak.bak.bak.bak"
    TRAIN_ARGS="--defer_full_meta true"
    INFERENCE_ARGS="--defer_full_meta true"
    ;;
  ARECHO)
    # Fill in ARECHO's config and exp dir here
    TRAIN_CONFIG="conf/train_aruniversa_prefix_full.yaml"        # adjust if different
    UNIVERSA_EXP="exp/arecho_YOUR_EXP_DIR_HERE"                  # <-- set this
    TRAIN_ARGS="--defer_full_meta true"
    INFERENCE_ARGS="--defer_full_meta true"
    ;;
  *)
    echo "ERROR: Unknown MODEL_TAG '${MODEL_TAG}'. Use ANCHOR or ARECHO."
    exit 1
    ;;
esac

# Shared inference settings — do NOT change these for this experiment
GPU_INFERENCE="false"
INFERENCE_NJ=128
USE_REF_WAV="true"
USE_REF_TEXT="false"

# Data root (where the symlinked dataset directories live)
DATA_ROOT="${RECIPE_DIR}"

# ---------------------------------------------------------------------------
# Test sets (must be symlinked under DATA_ROOT before submitting)
#
# Naming convention matches dataset_builder.py output:
#   prefix_distorted_overall_dev_<N>s
#   prefix_clean_overall_dev_<N>s
# ---------------------------------------------------------------------------

DISTORTED_TEST_SETS=(
    "prefix_distorted_overall_dev_2s"
    "prefix_distorted_overall_dev_4s"
    "prefix_distorted_overall_dev_6s"
    "prefix_distorted_overall_dev_8s"
)

CLEAN_TEST_SETS=(
    "prefix_clean_overall_dev_2s"
    "prefix_clean_overall_dev_4s"
    "prefix_clean_overall_dev_6s"
    "prefix_clean_overall_dev_8s"
)

ALL_TEST_SETS=("${DISTORTED_TEST_SETS[@]}" "${CLEAN_TEST_SETS[@]}")

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

echo "=============================================="
echo " SLURM Job Start"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $SLURM_NODELIST"
echo " CPUs     : $SLURM_CPUS_PER_TASK"
echo " Model    : $MODEL_TAG"
echo " Exp dir  : $UNIVERSA_EXP"
echo " Time     : $(date)"
echo "=============================================="

# Load modules (adjust module names to your cluster's module system)
module purge
module load anaconda3/23.3.1     # or whichever Anaconda module is available

# Activate the conda environment used for your ESPnet recipe
conda activate espnet            # replace with your actual environment name

# Confirm no GPU is being used (belt-and-suspenders)
unset CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES unset — running CPU-only inference."
echo "Python: $(which python)  |  Conda env: $CONDA_DEFAULT_ENV"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if [[ ! -f "${RECIPE_DIR}/uni_versa.sh" ]]; then
    echo "ERROR: uni_versa.sh not found in ${RECIPE_DIR}"
    exit 1
fi

if [[ ! -d "${RECIPE_DIR}/${UNIVERSA_EXP}" ]]; then
    echo "ERROR: UNIVERSA_EXP directory not found: ${RECIPE_DIR}/${UNIVERSA_EXP}"
    exit 1
fi

MISSING_DATASETS=()
for test_set in "${ALL_TEST_SETS[@]}"; do
    dir="${DATA_ROOT}/${test_set}"
    if [[ ! -d "$dir" ]]; then
        echo "WARNING: dataset directory not found: $dir"
        MISSING_DATASETS+=("$test_set")
    elif [[ ! -f "${dir}/wav.scp" ]]; then
        echo "WARNING: wav.scp missing in: $dir"
        MISSING_DATASETS+=("$test_set [no wav.scp]")
    fi
done

if [[ ${#MISSING_DATASETS[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: ${#MISSING_DATASETS[@]} dataset(s) are missing or incomplete."
    echo "Run dataset_builder.py and symlink before submitting this job."
    for ds in "${MISSING_DATASETS[@]}"; do
        echo "  - $ds"
    done
    exit 1
fi

echo "All ${#ALL_TEST_SETS[@]} dataset directories verified OK."

# ---------------------------------------------------------------------------
# Run inference on each test set sequentially
#
# Each call to uni_versa.sh launches up to --inference_nj=128 internal
# parallel processes by itself.  We run test sets sequentially here to
# avoid exceeding the node's 128-CPU budget.
# ---------------------------------------------------------------------------

cd "${RECIPE_DIR}" || { echo "Cannot cd to ${RECIPE_DIR}"; exit 1; }
mkdir -p logs    # ensure log directory exists for any inner ESPnet logging

FAILED_SETS=()
PASS_SETS=()

for test_set in "${ALL_TEST_SETS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo " Test set : ${test_set}"
    echo " Start    : $(date)"
    echo "--------------------------------------------------"

    ./uni_versa.sh \
        --stage           9 \
        --stop_stage      10 \
        --train_config    "${TRAIN_CONFIG}" \
        --universa_exp    "${UNIVERSA_EXP}" \
        --test_sets       "${test_set}" \
        --use_ref_wav     "${USE_REF_WAV}" \
        --use_ref_text    "${USE_REF_TEXT}" \
        --gpu_inference   "${GPU_INFERENCE}" \
        --inference_nj    "${INFERENCE_NJ}" \
        --train_args      "${TRAIN_ARGS}" \
        --inference_args  "${INFERENCE_ARGS}"

    exit_code=$?
    echo " End      : $(date)  (exit ${exit_code})"

    if [[ $exit_code -ne 0 ]]; then
        echo " FAILED: ${test_set}"
        FAILED_SETS+=("${test_set}")
    else
        echo " OK: ${test_set}"
        PASSED_SETS+=("${test_set}")
    fi
done

# ---------------------------------------------------------------------------
# Job summary
# ---------------------------------------------------------------------------

echo ""
echo "=============================================="
echo " SLURM Job End: $(date)"
echo " Model         : $MODEL_TAG"
echo " Passed        : ${#PASSED_SETS[@]} / ${#ALL_TEST_SETS[@]}"
echo " Failed        : ${#FAILED_SETS[@]}"
for f in "${FAILED_SETS[@]}"; do
    echo "   - $f"
done
echo "=============================================="

if [[ ${#FAILED_SETS[@]} -gt 0 ]]; then
    exit 1
fi
exit 0
