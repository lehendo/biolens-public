#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SAE training — secondary (4h cap, H100 available)
#
# Trains on pre-cached activations from extract_activations.sh.
# Run finalize first:
#   python scripts/extract_activations.py --finalize --output-dir $CACHE_DIR
#
# Usage:
#   sbatch scripts/slurm/train_sae.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_train_sae
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/arjunc4/biolens/logs/train_sae_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/train_sae_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="esm2_8m"
LAYER=5
VARIANT="topk"
K=32
EXPANSION=8
N_STEPS=10000
BATCH_SIZE=4096
LR="2e-4"
CACHE_DIR="/scratch/arjunc4/biolens/data/activations/${MODEL}_layer${LAYER}_mean"
CHECKPOINT_DIR="/scratch/arjunc4/biolens/checkpoints/${MODEL}_layer${LAYER}_${VARIANT}"
REPO="/u/arjunc4/biolens"
WANDB_PROJECT="biolens"

# ── Environment ───────────────────────────────────────────────────────────────
# set -e/pipefail deliberately NOT active yet: `source ~/.bashrc` and `conda
# activate` are known-fragile on this cluster under strict mode — confirmed
# 2026-07-03 via `bash -c 'set -e; source ~/.bashrc; conda activate biolens;
# echo SUCCESS'` failing silently (no output, exit 1). Something in the
# cluster's own bashrc/conda chain returns nonzero as part of normal,
# harmless control flow — fine under bash's default lenient mode, fatal
# under -e. Enabling strict mode only AFTER activation succeeds protects
# against real script bugs (see below) instead of dotfile quirks.
source ~/.bashrc
conda activate biolens

set -eo pipefail    # -e: without this, a failing python command doesn't stop
                     # the script or produce a nonzero job exit code — a real
                     # Experiment 4 run hit exactly this: crashed on a missing
                     # import, but the trailing echo still printed "complete"
                     # and sacct showed ExitCode 0:0 (2026-07-03).

export HF_HOME="/scratch/arjunc4/.cache/huggingface"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "/scratch/arjunc4/biolens/logs"

echo "Training on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Cache: ${CACHE_DIR}"

# ── Train ─────────────────────────────────────────────────────────────────────
python "${REPO}/scripts/train_sae.py" \
    --model             "${MODEL}"            \
    --layer             "${LAYER}"            \
    --variant           "${VARIANT}"          \
    --k                 "${K}"                \
    --expansion-factor  "${EXPANSION}"        \
    --n-steps           "${N_STEPS}"          \
    --batch-size        "${BATCH_SIZE}"       \
    --lr                "${LR}"              \
    --cache             "${CACHE_DIR}"        \
    --output-dir        "${CHECKPOINT_DIR}"   \
    --wandb-project     "${WANDB_PROJECT}"    \
    --device            cuda

echo "Training complete."
