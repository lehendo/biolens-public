#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Activation extraction — SLURM array job (secondary, 4h cap)
#
# Runs N_SHARDS independent tasks, each processing 1/N_SHARDS of Swiss-Prot.
# Array tasks write disjoint shards; filelock protects the manifest.
#
# Usage:
#   Edit MODEL, LAYER, POOLING, N_SHARDS, OUTPUT_DIR below, then:
#   sbatch scripts/slurm/extract_activations.sh
#
# After all tasks complete:
#   python scripts/extract_activations.py --finalize --output-dir $OUTPUT_DIR
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_extract
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-9                           # 10 shards → each handles ~57K of 570K seqs
#SBATCH --output=/scratch/arjunc4/biolens/logs/extract_%A_%a.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/extract_%A_%a.err
#SBATCH --requeue                             # requeue on preemption (secondary is non-preemptible
                                              # but --requeue is good practice)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="esm2_8m"
LAYER=5                   # last layer for ESM2 8M (num_layers - 1)
POOLING="mean"
N_SHARDS=10               # must match --array upper bound + 1
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

echo "Task ${SLURM_ARRAY_TASK_ID}/${N_SHARDS} on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

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

echo "Task ${SLURM_ARRAY_TASK_ID} complete."
