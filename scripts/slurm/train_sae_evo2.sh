#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Evo 2 SAE training.
#
# Reuses scripts/train_sae.py unchanged — it's already fully generic, driven
# entirely by configs/models.yaml's hidden_dim for whichever --model is
# given (evo2_7b -> d_model=4096, vs. ESM2's 320-1280). This script is just
# the appropriately-resourced SLURM wrapper: H200 + more memory than the
# ESM2 training jobs, since d_sae = d_model * expansion_factor is much
# larger here (4096 * 8 = 32768 at the default expansion factor).
#
# Finalizes the cache (norm stats) itself, at the top of this script, rather
# than relying on a separate manual step someone has to remember to run
# between extraction and training — that gap is exactly the bug caught and
# fixed in train_sae_multiscale.sh (2026-07-03): a cache that's never had
# finalize() called on it makes train_sae.py fail immediately with a
# RuntimeError from ActivationCache.get_norm_stats().
#
# Usage:
#   sbatch --export=LAYER=16 scripts/slurm/train_sae_evo2.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_train_sae_evo2
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00                       # secondary's hard MaxTime
                                                # (scontrol show partition secondary,
                                                # 2026-07-03) — using the full cap;
                                                # SAE training on already-extracted
                                                # activations should be far lighter
                                                # than the 7B forward passes anyway.
#SBATCH --output=/scratch/arjunc4/biolens/logs/train_sae_evo2_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/train_sae_evo2_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="evo2_7b"
LAYER="${LAYER:-16}"
VARIANT="topk"
K=32
EXPANSION=8
N_STEPS=10000
BATCH_SIZE=2048            # smaller than ESM2's 4096 — d_model=4096 activations are
                            # 12.8x larger per-vector than ESM2 8M's d_model=320
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
echo "Cache: ${CACHE_DIR}"

# ── Finalize the cache (compute norm stats) ─────────────────────────────────────
# Idempotent — safe even if already finalized.
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
    --device            cuda

echo "Training complete: ${CHECKPOINT_DIR}"
