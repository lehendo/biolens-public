#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Phase 0 end-to-end pipeline (local dev / single-GPU)
#
# Runs the complete Phase 0 pipeline on a small slice of Swiss-Prot
# to verify the pipeline is working before submitting to the cluster.
#
# Exit criterion: FVE > 0.80 AND any GO term single_feature_auroc > 0.70
#
# Usage:
#   bash scripts/run_phase0.sh [--quick]   (--quick: 1000 seqs, 2000 steps)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then QUICK=true; fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${REPO}/data"   # local dev: store under repo data/; cluster: use /scratch

MODEL="esm2_8m"
LAYER=5
CACHE_DIR="${SCRATCH}/activations/${MODEL}_layer${LAYER}_mean"
CHECKPOINT_DIR="${SCRATCH}/checkpoints/${MODEL}_layer${LAYER}_topk"
EVAL_DIR="${SCRATCH}/eval/${MODEL}_layer${LAYER}_topk"

MAX_SEQS=20000
N_SHARDS=1
N_STEPS=5000

if $QUICK; then
    MAX_SEQS=1000
    N_STEPS=500
    echo "Quick mode: ${MAX_SEQS} sequences, ${N_STEPS} training steps"
fi

mkdir -p "${CACHE_DIR}" "${CHECKPOINT_DIR}" "${EVAL_DIR}"
mkdir -p "${SCRATCH}/annotations"

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export BIOLENS_DATA_DIR="${SCRATCH}/annotations"

echo "══════════════════════════════════════════════"
echo " BioLens Phase 0 pipeline"
echo " Model: ${MODEL}  Layer: ${LAYER}  Seqs: ${MAX_SEQS}"
echo "══════════════════════════════════════════════"

# ── 1. Extract activations ─────────────────────────────────────────────────
echo ""
echo "Step 1/4: Extracting activations..."
python "${REPO}/scripts/extract_activations.py" \
    --model "${MODEL}" --layer "${LAYER}" --pooling mean \
    --output-dir "${CACHE_DIR}" \
    --data-source swissprot \
    --max-seqs "${MAX_SEQS}" \
    --max-seq-len 1024 \
    --batch-size 32 \
    --shard-id 0 --n-shards 1

# ── 2. Finalize cache ─────────────────────────────────────────────────────
echo ""
echo "Step 2/4: Finalizing cache (computing normalization stats)..."
python "${REPO}/scripts/extract_activations.py" \
    --finalize --output-dir "${CACHE_DIR}"

# ── 3. Train SAE ──────────────────────────────────────────────────────────
echo ""
echo "Step 3/4: Training TopK SAE..."
python "${REPO}/scripts/train_sae.py" \
    --model "${MODEL}" --layer "${LAYER}" \
    --variant topk --k 32 --expansion-factor 8 \
    --n-steps "${N_STEPS}" --batch-size 1024 --lr 2e-4 \
    --cache "${CACHE_DIR}" \
    --output-dir "${CHECKPOINT_DIR}" \
    --no-wandb

# Find the final checkpoint
CHECKPOINT=$(ls -t "${CHECKPOINT_DIR}"/*/final.pt 2>/dev/null | head -1)
if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT=$(ls -t "${CHECKPOINT_DIR}"/*/*.pt 2>/dev/null | head -1)
fi

# ── 4. Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "Step 4/4: Running evaluation..."
python "${REPO}/scripts/run_eval.py" \
    --model "${MODEL}" --layer "${LAYER}" \
    --sae-checkpoint "${CHECKPOINT}" \
    --cache "${CACHE_DIR}" \
    --output-dir "${EVAL_DIR}" \
    --max-eval-seqs "${MAX_SEQS}" \
    --min-positives 20 \
    --max-go-terms 200 \
    --no-wandb

echo ""
echo "══════════════════════════════════════════════"
echo " Phase 0 complete.  Results: ${EVAL_DIR}"
echo "══════════════════════════════════════════════"
