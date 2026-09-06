#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Seed configs/verified_features/gemma_scope_2b_layer12_experiment4.yaml with
# real candidates (see scripts/select_experiment4_candidates.py's module
# docstring for the full story). ONE forward pass over the Bias-in-Bios corpus, not the full
# min_positives sweep — scripts/slurm/run_experiment4.sh's completed runs
# (~4h15m for the full 7-point sweep) are NOT the right timing comparison;
# this should be a small fraction of that. Same partition/account/gres as
# that proven-working config, much shorter --time as a safety margin, not
# an expectation.
#
# Requires the same gated HuggingFace access as run_experiment4.sh (Gemma-2-2B
# + Gemma Scope license accepted) — see that script's header for the
# pre-flight check.
#
# Usage:
#   sbatch scripts/slurm/select_experiment4_candidates.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_exp4_candidates
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/arjunc4/biolens/logs/exp4_candidates_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/exp4_candidates_%j.err

source ~/.bashrc
conda activate biolens

set -eo pipefail

REPO="/u/arjunc4/biolens"
export HF_HOME="/scratch/arjunc4/.cache/huggingface"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

echo "Experiment 4 candidate selection on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python "${REPO}/scripts/select_experiment4_candidates.py" \
    --sweep-results "/scratch/arjunc4/biolens/results/experiment4_gemma_scope_L12/mp10/go_probing_results.json" \
    --registry      "${REPO}/configs/verified_features/gemma_scope_2b_layer12_experiment4.yaml" \
    --top-k         5 \
    --device        cuda

echo "Candidate selection complete."
