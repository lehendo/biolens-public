#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SAE training — parameterized for multi-scale (ESM2 35M/150M/650M) and the
# expansion-factor / k-value sub-experiments.
#
# Trains on pre-cached, finalized activations from
# extract_activations_multiscale.sh.
#
# Usage:
#   sbatch --export=MODEL=esm2_35m,LAYER=5,EXPANSION=8,K=32 \
#       scripts/slurm/train_sae_multiscale.sh
#
#   # Expansion-factor sweep (Experiment 3, new 2026-07-01):
#   sbatch --export=MODEL=esm2_8m,LAYER=5,EXPANSION=4,K=32  scripts/slurm/train_sae_multiscale.sh
#   sbatch --export=MODEL=esm2_8m,LAYER=5,EXPANSION=16,K=32 scripts/slurm/train_sae_multiscale.sh
#
#   # Sparsity (k) sweep:
#   sbatch --export=MODEL=esm2_8m,LAYER=5,EXPANSION=8,K=16  scripts/slurm/train_sae_multiscale.sh
#   sbatch --export=MODEL=esm2_8m,LAYER=5,EXPANSION=8,K=64  scripts/slurm/train_sae_multiscale.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_train_sae_ms
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/arjunc4/biolens/logs/train_sae_ms_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/train_sae_ms_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="${MODEL:?Set MODEL via --export=MODEL=esm2_35m}"
LAYER="${LAYER:?Set LAYER via --export=LAYER=<layer index>}"
VARIANT="topk"
K="${K:-32}"
EXPANSION="${EXPANSION:-8}"
N_STEPS=10000
BATCH_SIZE=4096
LR="2e-4"
CACHE_DIR="/scratch/arjunc4/biolens/data/activations/${MODEL}_layer${LAYER}_mean"
CHECKPOINT_DIR="/scratch/arjunc4/biolens/checkpoints/${MODEL}_layer${LAYER}_${VARIANT}_k${K}_x${EXPANSION}"
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
                     # and sacct showed ExitCode 0:0 (2026-07-03). Especially
                     # important here: two sequential python steps (finalize,
                     # then train) — without this, a failed finalize wouldn't
                     # stop the training step from running against a
                     # non-finalized cache.

export HF_HOME="/scratch/arjunc4/.cache/huggingface"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "/scratch/arjunc4/biolens/logs"

echo "Training on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model=${MODEL} Layer=${LAYER} k=${K} expansion=${EXPANSION}"
echo "Cache: ${CACHE_DIR}"

# ── Finalize the cache (compute norm stats) ─────────────────────────────────────
# Safe to run here even though it wasn't run right after extraction: this job
# only starts once the extraction array job's --dependency=afterok has been
# satisfied, so every shard already exists on disk by this point. Idempotent —
# safe to re-run if a previous training job already finalized this same cache
# (e.g. the k/expansion-factor variants on ESM2 8M, whose cache was already
# finalized back in Phase 0).
python "${REPO}/scripts/extract_activations.py" \
    --model      "${MODEL}"     \
    --layer      "${LAYER}"     \
    --finalize                  \
    --output-dir "${CACHE_DIR}"

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
    --run-name          "${MODEL}_L${LAYER}_k${K}_x${EXPANSION}" \
    --device            cuda

echo "Training complete: ${CHECKPOINT_DIR}"
