#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Activation extraction — ESM2 35M/150M/650M (8M already done, Phase 0)
#
# Parameterized version of extract_activations.sh: pass MODEL and LAYER as
# environment variables via --export, so the same script covers all three
# remaining ESM2 sizes without triplicating the file.
#
# Usage (submit one job per model size):
#   sbatch --export=MODEL=esm2_35m,LAYER=5   scripts/slurm/extract_activations_multiscale.sh
#   sbatch --export=MODEL=esm2_150m,LAYER=14 scripts/slurm/extract_activations_multiscale.sh
#   sbatch --export=MODEL=esm2_650m,LAYER=16 scripts/slurm/extract_activations_multiscale.sh
# (LAYER defaults to a mid-depth layer per configs/eval.yaml::probe_layers if
# unset — see the fallback logic below. For the layer-ablation sub-experiment,
# submit one job per layer in configs/eval.yaml::probe_layers.<model>.)
#
# After all array tasks for a given (MODEL, LAYER) complete:
#   python scripts/extract_activations.py --model $MODEL --layer $LAYER \
#       --finalize --output-dir $OUTPUT_DIR
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_extract_ms
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/arjunc4/biolens/logs/extract_ms_%A_%a.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/extract_ms_%A_%a.err
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="${MODEL:?Set MODEL via --export=MODEL=esm2_35m (or _150m / _650m)}"
LAYER="${LAYER:?Set LAYER via --export=LAYER=<layer index> — see configs/eval.yaml::probe_layers}"
POOLING="mean"
N_SHARDS=10
OUTPUT_DIR="/scratch/arjunc4/biolens/data/activations/${MODEL}_layer${LAYER}_${POOLING}"
REPO="/u/arjunc4/biolens"

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
export BIOLENS_DATA_DIR="/scratch/arjunc4/biolens/data/annotations"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "/scratch/arjunc4/biolens/logs"

echo "Model=${MODEL} Layer=${LAYER} Task ${SLURM_ARRAY_TASK_ID}/${N_SHARDS} on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Extract ───────────────────────────────────────────────────────────────────
python "${REPO}/scripts/extract_activations.py" \
    --model         "${MODEL}"            \
    --layer         "${LAYER}"            \
    --pooling       "${POOLING}"          \
    --output-dir    "${OUTPUT_DIR}"       \
    --data-source   swissprot             \
    --max-seq-len   1024                  \
    --batch-size    64                    \
    --shard-id      "${SLURM_ARRAY_TASK_ID}" \
    --n-shards      "${N_SHARDS}"         \
    --device        cuda

echo "Task ${SLURM_ARRAY_TASK_ID} complete (model=${MODEL}, layer=${LAYER})."
