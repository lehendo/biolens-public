#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1b — genomic-annotation (ENCODE cCRE) linear probing for a
# trained Evo 2 SAE.
#
# A SINGLE job, not an array: scripts/run_experiment1b_evo2_genomic.py
# extracts Evo 2 activations for a fixed eval window set ONCE, then runs the
# probing sweep — same pattern as run_experiment4.sh (Gemma Scope), not
# run_sweep_eval.sh's per-min_positives array pattern (that pattern exists to
# parallelize the sample-size SWEEP across many min_positives values in
# separate jobs; this script runs at a single min_positives value per
# invocation — resubmit with a different --min-positives export to sweep, or
# extend this script later if a full min_positives sweep over genomic
# annotations is needed).
#
# REQUIRES the reference genome already downloaded (see
# extract_activations_evo2.sh's usage comment — same file, same path,
# reused here) and a trained Evo 2 SAE checkpoint (train_sae_evo2.sh).
#
# secondary/H100, not IllinoisComputes-GPU: this job only extracts activations
# for a bounded eval window set (--max-eval-windows, default 10,000), not the
# full genome — a small fraction of extract_activations_evo2.sh's ~1M-window,
# 40-shard training extraction (each shard ~25K windows in under 4h on the
# same GPU type). 10,000 windows should comfortably fit inside secondary's
# 4h cap; switch to IllinoisComputes-GPU with a longer --time if a real run
# shows otherwise, the same way Experiment 4 needed to.
#
# Usage:
#   sbatch --export=SAE_CHECKPOINT=/scratch/arjunc4/biolens/checkpoints/evo2_7b_layer16_topk/evo2_7b_L16_topk_k32/final.pt \
#       scripts/slurm/run_experiment1b_evo2_genomic.sh
#
# Note the nested <run_dir>/<run_name>/final.pt layout (same pattern ESM2's SAE
# checkpoints use) — job 9531444 (2026-07-16) FAILED with FileNotFoundError
# from a flattened path that skipped the inner run_name directory.
#
# Replication check on an independent eval window set (e.g. the PLS
# confounding-estimate anomaly): set WINDOW_OFFSET
# (0 <= offset < WINDOW_STRIDE, default stride 200000) AND RUN_SUFFIX (so the
# output directory doesn't collide with and overwrite the primary run's
# results at the same LAYER):
#   sbatch --export=SAE_CHECKPOINT=...,WINDOW_OFFSET=100000,RUN_SUFFIX=_replicate1 \
#       scripts/slurm/run_experiment1b_evo2_genomic.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_evo2_genomic
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/arjunc4/biolens/logs/evo2_genomic_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/evo2_genomic_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="evo2_7b"
LAYER="${LAYER:-16}"
SAE_CHECKPOINT="${SAE_CHECKPOINT:?Set SAE_CHECKPOINT via --export}"
REFERENCE_GENOME="/scratch/arjunc4/biolens/data/reference/hg38.fa"
MIN_POSITIVES="${MIN_POSITIVES:-50}"
MAX_EVAL_WINDOWS="${MAX_EVAL_WINDOWS:-10000}"
WINDOW_STRIDE="${WINDOW_STRIDE:-200000}"
WINDOW_OFFSET="${WINDOW_OFFSET:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
OUTPUT_DIR="/scratch/arjunc4/biolens/results/experiment1b_evo2_genomic_L${LAYER}${RUN_SUFFIX}"
REPO="/u/arjunc4/biolens"

# ── Environment ───────────────────────────────────────────────────────────────
# set -e/pipefail deliberately NOT active yet — see extract_activations_evo2.sh
# for the full explanation of why this must come AFTER conda activate on this
# cluster, not before.
source ~/.bashrc
conda activate biolens

set -eo pipefail

export HF_HOME="/scratch/arjunc4/.cache/huggingface"
export BIOLENS_DATA_DIR="/scratch/arjunc4/biolens/data/annotations"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "/scratch/arjunc4/biolens/logs"

if [[ ! -f "${REFERENCE_GENOME}" ]]; then
    echo "ERROR: reference genome not found at ${REFERENCE_GENOME}" >&2
    echo "Download it first (see extract_activations_evo2.sh's usage comment)." >&2
    exit 1
fi

echo "Experiment 1b (Evo 2 genomic) on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Run ───────────────────────────────────────────────────────────────────────
python "${REPO}/scripts/run_experiment1b_evo2_genomic.py" \
    --model             "${MODEL}"            \
    --layer             "${LAYER}"            \
    --sae-checkpoint    "${SAE_CHECKPOINT}"   \
    --reference-genome  "${REFERENCE_GENOME}" \
    --window-stride     "${WINDOW_STRIDE}"    \
    --window-offset     "${WINDOW_OFFSET}"    \
    --max-eval-windows  "${MAX_EVAL_WINDOWS}" \
    --min-positives     "${MIN_POSITIVES}"    \
    --held-out-baselines                       \
    --held-out-fraction 0.5                    \
    --permuted-label-control                   \
    --output-dir        "${OUTPUT_DIR}"       \
    --device            cuda

echo "Experiment 1b complete: ${OUTPUT_DIR}"
