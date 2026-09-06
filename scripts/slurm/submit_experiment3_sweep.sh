#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Orchestrates the full Experiment 3 build: for each ESM2 scale
# (35M/150M/650M — 8M's activations already exist from Phase 0), plus the
# expansion-factor and k-value variants on 8M, chains:
# extract -> finalize -> train -> sweep-eval, via
# `sbatch --dependency=afterok:<jobid>`.
#
# This is a driver script run directly on the login node (NOT itself an
# sbatch job) — it just submits the real jobs and prints their IDs.
#
# Usage:
#   bash scripts/slurm/submit_experiment3_sweep.sh
#
# Requires: extract_activations_multiscale.sh, train_sae_multiscale.sh,
# run_sweep_eval.sh (all in this directory), and the Phase 0 ESM2 8M
# activation cache already present (no extraction job submitted for 8M).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="/u/arjunc4/biolens"
SCRATCH="/scratch/arjunc4/biolens"

# ── (model, layer) pairs to cover — scale sweep across ESM2 sizes.
# Layers chosen to match configs/eval.yaml::probe_layers' mid-depth entry per
# model (the layer used for the main Experiment 3 cross-scale sweep; run the
# layer-ablation sub-experiment separately over the full probe_layers list).
declare -A MODEL_LAYERS=(
    [esm2_35m]=5
    [esm2_150m]=14
    [esm2_650m]=16
)

# ── (model, layer, k, expansion) variants for the 8M expansion-factor / k
# sweeps (Experiment 3, added 2026-07-01).
# 8M/layer5/k32/x8 is Phase 0's existing checkpoint — not resubmitted here.
EXTRA_VARIANTS=(
    "esm2_8m 5 32 4"    # expansion-factor sweep point
    "esm2_8m 5 32 16"   # expansion-factor sweep point
    "esm2_8m 5 16 8"    # sparsity (k) sweep point
    "esm2_8m 5 64 8"    # sparsity (k) sweep point
)

SWEEP_POINTS_JOB_IDS=()

submit_scale_job() {
    local model=$1 layer=$2 k=$3 expansion=$4 needs_extraction=$5

    local cache_dir="${SCRATCH}/data/activations/${model}_layer${layer}_mean"
    local checkpoint_dir="${SCRATCH}/checkpoints/${model}_layer${layer}_topk_k${k}_x${expansion}"
    local run_name="${model}_L${layer}_k${k}_x${expansion}"
    local checkpoint="${checkpoint_dir}/${run_name}/final.pt"
    local sweep_out="${SCRATCH}/eval/sweep_${run_name}"

    local train_dependency=""
    if [[ "${needs_extraction}" == "yes" ]]; then
        echo "Submitting extraction array job for ${model} (layer ${layer})..."
        local extract_jid
        extract_jid=$(sbatch --parsable --export="MODEL=${model},LAYER=${layer}" \
            "${SLURM_DIR}/extract_activations_multiscale.sh")
        echo "  extract job: ${extract_jid} (array 0-9)"
        train_dependency="--dependency=afterok:${extract_jid}"
    fi

    echo "Submitting training job for ${run_name}..."
    local train_jid
    # shellcheck disable=SC2086  # $train_dependency must word-split when empty
    train_jid=$(sbatch --parsable ${train_dependency} \
        --export="MODEL=${model},LAYER=${layer},K=${k},EXPANSION=${expansion}" \
        "${SLURM_DIR}/train_sae_multiscale.sh")
    echo "  train job: ${train_jid}"

    echo "Submitting sweep-eval array job for ${run_name}..."
    local sweep_jid
    sweep_jid=$(sbatch --parsable --dependency="afterok:${train_jid}" \
        --export="MODEL=${model},LAYER=${layer},CHECKPOINT=${checkpoint},CACHE_DIR=${cache_dir},OUTPUT_ROOT=${sweep_out}" \
        "${SLURM_DIR}/run_sweep_eval.sh")
    echo "  sweep-eval job: ${sweep_jid} (array 0-6) -> ${sweep_out}"

    SWEEP_POINTS_JOB_IDS+=("${sweep_jid}")
}

echo "── Scale sweep: ESM2 35M/150M/650M (extraction + train + sweep-eval) ──"
for model in "${!MODEL_LAYERS[@]}"; do
    layer="${MODEL_LAYERS[$model]}"
    submit_scale_job "${model}" "${layer}" 32 8 yes
done

echo ""
echo "── Expansion-factor / k variants on ESM2 8M (train + sweep-eval only, ──"
echo "── activations already extracted in Phase 0) ──"
for variant in "${EXTRA_VARIANTS[@]}"; do
    read -r model layer k expansion <<< "${variant}"
    submit_scale_job "${model}" "${layer}" "${k}" "${expansion}" no
done

echo ""
echo "All jobs submitted. Sweep-eval array job IDs: ${SWEEP_POINTS_JOB_IDS[*]}"
echo ""
echo "Once every sweep-eval array job shows COMPLETED (check with 'sacct -j <jobid>'),"
echo "aggregate each run with scripts/fit_sweep.py, e.g.:"
echo "  python ${REPO}/scripts/fit_sweep.py \\"
echo "      --sweep-dir ${SCRATCH}/eval/sweep_esm2_35m_L5_k32_x8 \\"
echo "      --verified-features configs/verified_features/esm2_35m_layer5_topk_k32.yaml \\"
echo "      --output docs/experiment3_sweep_fit_esm2_35m.json"
