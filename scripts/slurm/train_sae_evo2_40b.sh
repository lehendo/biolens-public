#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Evo 2 40B SAE training.
#
# Reuses scripts/train_sae.py unchanged, same as the 7B training script —
# it's fully generic, driven by configs/models.yaml's hidden_dim for
# whichever --model is given. IMPORTANT, unlike the extraction script:
# train_sae.py never loads the Evo 2 model itself (grep-confirmed
# 2026-07-25 — it only calls registry.get_config() to read hidden_dim from
# YAML), so this script does NOT need transformer_engine and is NOT
# blocked by the same thing extract_activations_evo2_40b.sh's banner is.
# Its only real dependency is DATA, not environment: it needs that
# extraction script to have actually produced and finalized a real
# activation cache first. Safe to leave staged/ready regardless of when
# transformer_engine gets installed.
#
# H100/secondary, not H200/scavenger (unlike the extraction script) —
# training operates purely on already-extracted, on-disk activation
# vectors (d=8192) plus the SAE's own weights, nowhere near the >80GB the
# full 40B model needs for a forward pass. The SAE itself is bigger than
# 7B's though: d_sae = d_model * expansion_factor = 8192 * 8 = 65536 (vs
# 7B's 32768) — W_enc/W_dec are each ~537M params here (8192 x 65536)
# vs. ~134M for 7B (4096 x 32768), a 4x jump in the encoder/decoder
# matrices' own parameter count (plus Adam's 2x state on top of that) —
# --mem and --batch-size below are adjusted for that, as a first-guess
# (not empirically calibrated, same caveat as the extraction script);
# recalibrate from this script's own first real run if it OOMs or has
# obvious slack.
#
# Usage:
#   sbatch scripts/slurm/train_sae_evo2_40b.sh
#   (run only after extract_activations_evo2_40b.sh has completed AND been
#   finalized for LAYER=24 — this script finalizes it again itself at the
#   top, idempotently, same pattern as the 7B script, but the cache must
#   exist at all first.)
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_train_sae_evo2_40b
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G                            # double the 7B script's 128G —
                                                # 4x larger W_enc/W_dec plus
                                                # Adam state; first-guess, see
                                                # header note.
#SBATCH --time=04:00:00                       # secondary's hard MaxTime —
                                                # training on already-extracted
                                                # activations should still be
                                                # far lighter than a forward-
                                                # pass-heavy job, same
                                                # reasoning as the 7B script.
#SBATCH --output=/scratch/arjunc4/biolens/logs/train_sae_evo2_40b_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/train_sae_evo2_40b_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="evo2_40b"
LAYER="${LAYER:-24}"          # matches extract_activations_evo2_40b.sh's default —
                                # see that script's header for why 24, not 16 or 25
VARIANT="topk"
K=32
EXPANSION=8
N_STEPS=10000
BATCH_SIZE=512              # down from 7B's 2048 — see header note on the ~4x
                              # larger encoder/decoder matrices; first-guess
LR="2e-4"
CACHE_DIR="/scratch/arjunc4/biolens/data/activations/${MODEL}_layer${LAYER}_mean"
CHECKPOINT_DIR="/scratch/arjunc4/biolens/checkpoints/${MODEL}_layer${LAYER}_${VARIANT}"
REPO="/u/arjunc4/biolens"
WANDB_PROJECT="biolens"

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate biolens

set -eo pipefail

export HF_HOME="/scratch/arjunc4/.cache/huggingface"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "/scratch/arjunc4/biolens/logs"

if [[ ! -d "${CACHE_DIR}" ]]; then
    echo "ERROR: activation cache not found at ${CACHE_DIR}" >&2
    echo "Run extract_activations_evo2_40b.sh first (requires transformer_engine" >&2
    echo "to be installed — see that script's header banner)." >&2
    exit 1
fi

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
