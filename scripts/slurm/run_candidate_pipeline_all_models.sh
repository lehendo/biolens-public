#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Registry-growth pipeline (Round 8+) for ALL FOUR ESM2 models in one job.
#
# Automates the purely mechanical steps of the verified-features workflow —
# NOT the verification judgment itself. For each model, in sequence:
#   1. fit_sweep.py   — carryover check: how many of the EXISTING registry's
#                        entries land on a real top-hit in the 100K sweep
#                        (informational only, never fatal — see note below)
#   2. select_fresh_candidates.py — pull up to N_PER_MP new candidates per
#                        min_positives bucket, excluding anything already in
#                        the registry
#   3. inspect_features.py        — dump top-activating real sequences for
#                        every fresh candidate, so they can be checked against
#                        real UniProt records
#
# What this script deliberately does NOT do: decide whether a candidate's
# top-activating sequences actually match its claimed GO term. That step
# requires reading real protein records and making a judgment call — the
# entire reason this registry exists is that this judgment is NOT reliable
# to skip (7/8 of Phase 0's naive top-feature AUROC "findings" turned out
# spurious on inspection). Automating it away here would silently reopen
# that exact failure mode. Run this script to generate the raw material,
# then verify each candidate manually (via UniProt) before it goes in the
# registry.
#
# No GPU needed — every step here reads pre-cached activations + JSON, pure
# CPU/IO work. Submitted as a SLURM job anyway (not run on a login node)
# because inspect_features.py scans the full activation cache per model,
# which is real memory/CPU work for the larger models (esm2_650m: d_sae=10240).
#
# Usage:
#   sbatch scripts/slurm/run_candidate_pipeline_all_models.sh
#
# Output (one directory per model):
#   /scratch/arjunc4/biolens/eval/round8_candidates/<model>/carryover.log
#   /scratch/arjunc4/biolens/eval/round8_candidates/<model>/candidates.txt
#   /scratch/arjunc4/biolens/eval/round8_candidates/<model>/inspect.log
#
# After the job completes, cat each model's candidates.txt + inspect.log
# (or the whole round8_candidates/ tree) back for verification against
# real UniProt records before updating configs/verified_features/*.yaml.
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_candidate_pipeline
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/arjunc4/biolens/logs/candidate_pipeline_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/candidate_pipeline_%j.err

REPO="/u/arjunc4/biolens"
DATA_ROOT="/scratch/arjunc4/biolens"
OUTPUT_ROOT="${DATA_ROOT}/eval/round8_candidates"
N_PER_MP=30       # same value already used for esm2_8m's Round 8 pull — kept
                  # uniform across all four models per "consistent and
                  # rigorous" (2026-07-21)
TOP_K=10

source ~/.bashrc
conda activate biolens
set -eo pipefail   # see run_sweep_eval.sh for why this is enabled only after
                   # conda activate, not before

export HF_HOME="/scratch/arjunc4/.cache/huggingface"
export BIOLENS_DATA_DIR="/scratch/arjunc4/biolens/data/annotations"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "/scratch/arjunc4/biolens/logs"

# ── Per-model config — every path verified via `find`/shell history this
# session, not guessed.
# MODEL_KEY:CHECKPOINT:CACHE_DIR:SWEEP_DIR_100K:REGISTRY:EXPANSION_FACTOR
CONFIGS=(
  "esm2_8m:${DATA_ROOT}/checkpoints/esm2_8m_layer5_topk/esm2_8m_L5_topk_k32/final.pt:${DATA_ROOT}/data/activations/esm2_8m_layer5_mean:${DATA_ROOT}/eval/sweep_esm2_8m_L5_k32_x8_100k:${REPO}/configs/verified_features/esm2_8m_layer5_topk_k32.yaml:8"
  "esm2_35m:${DATA_ROOT}/checkpoints/esm2_35m_layer5_topk_k32_x8/esm2_35m_L5_k32_x8/final.pt:${DATA_ROOT}/data/activations/esm2_35m_layer5_mean:${DATA_ROOT}/eval/sweep_esm2_35m_L5_k32_x8_100k:${REPO}/configs/verified_features/esm2_35m_layer5_topk_k32.yaml:"
  "esm2_150m:${DATA_ROOT}/checkpoints/esm2_150m_layer14_topk_k32_x8/esm2_150m_L14_k32_x8/final.pt:${DATA_ROOT}/data/activations/esm2_150m_layer14_mean:${DATA_ROOT}/eval/sweep_esm2_150m_L14_k32_x8_100k:${REPO}/configs/verified_features/esm2_150m_layer14_topk_k32.yaml:"
  "esm2_650m:${DATA_ROOT}/checkpoints/esm2_650m_layer16_topk_k32_x8/esm2_650m_L16_k32_x8/final.pt:${DATA_ROOT}/data/activations/esm2_650m_layer16_mean:${DATA_ROOT}/eval/sweep_esm2_650m_L16_k32_x8_100k:${REPO}/configs/verified_features/esm2_650m_layer16_topk_k32.yaml:"
)

for entry in "${CONFIGS[@]}"; do
  IFS=':' read -r MODEL CHECKPOINT CACHE_DIR SWEEP_DIR REGISTRY EXPANSION_FACTOR <<< "${entry}"

  MODEL_OUT="${OUTPUT_ROOT}/${MODEL}"
  mkdir -p "${MODEL_OUT}"
  echo "════════════════════════════════════════════════════════════"
  echo "${MODEL}"
  echo "════════════════════════════════════════════════════════════"

  EXPANSION_FACTOR_ARGS=()
  if [[ -n "${EXPANSION_FACTOR}" ]]; then
    EXPANSION_FACTOR_ARGS=(--expansion-factor "${EXPANSION_FACTOR}" --d-model 320 --variant topk)
  fi

  # Step 1 — carryover check. Informational only: fit_sweep.py only hard-exits
  # if ZERO registry entries match a real top-hit in this sweep at all; a low
  # n_clusters (bootstrap needs >=10) just means the printed CI is skipped,
  # not a script failure. `|| true` guards the (unlikely but possible) total-
  # zero case so one model's edge case can't kill the other three.
  echo "--- Step 1: carryover check (${MODEL}) ---"
  python "${REPO}/scripts/fit_sweep.py" \
      --sweep-dir "${SWEEP_DIR}" \
      --verified-features "${REGISTRY}" \
      2>&1 | tee "${MODEL_OUT}/carryover.log" || true

  # Step 2 — pull fresh candidates.
  echo "--- Step 2: select fresh candidates (${MODEL}) ---"
  python "${REPO}/scripts/select_fresh_candidates.py" \
      --sweep-dir "${SWEEP_DIR}" \
      --verified-features "${REGISTRY}" \
      --n-per-mp "${N_PER_MP}" \
      2>&1 | tee "${MODEL_OUT}/candidates.txt"

  # Parse the feature-index line select_fresh_candidates.py prints last.
  FEATURE_INDICES=$(grep -A1 "Feature indices for inspect_features.py" "${MODEL_OUT}/candidates.txt" | tail -1)

  if [[ -z "${FEATURE_INDICES// /}" ]]; then
    echo "${MODEL}: no fresh candidates this round — skipping inspect_features.py"
    continue
  fi

  # Prefer mp10's results file for GO-term cross-reference annotation (purely
  # informational in inspect_features.py's printed report — select_fresh_candidates.py's
  # own output above already carries the authoritative go_id/auroc per candidate);
  # fall back to whichever mp subdir exists first if mp10 is absent.
  GO_PROBING_RESULTS="${SWEEP_DIR}/mp10/go_probing_results.json"
  if [[ ! -f "${GO_PROBING_RESULTS}" ]]; then
    GO_PROBING_RESULTS=$(find "${SWEEP_DIR}" -name go_probing_results.json | sort | head -1)
  fi

  echo "--- Step 3: inspect_features.py (${MODEL}, $(echo ${FEATURE_INDICES} | wc -w) features) ---"
  python "${REPO}/scripts/inspect_features.py" \
      --sae-checkpoint "${CHECKPOINT}" \
      --cache "${CACHE_DIR}" \
      --feature-idx ${FEATURE_INDICES} \
      --top-k "${TOP_K}" \
      --go-probing-results "${GO_PROBING_RESULTS}" \
      --verified-annotations "${REGISTRY}" \
      --device cpu \
      "${EXPANSION_FACTOR_ARGS[@]}" \
      2>&1 | tee "${MODEL_OUT}/inspect.log"

  echo "${MODEL} done. Output: ${MODEL_OUT}/"
done

echo "════════════════════════════════════════════════════════════"
echo "All models complete. Output tree: ${OUTPUT_ROOT}/"
echo "════════════════════════════════════════════════════════════"
