#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Controlled subsampling variant — scripts/run_controlled_subsampling.py.
#
# No GPU needed (SAE-encoding an already-extracted activation cache is cheap
# CPU work, same reasoning as run_candidate_pipeline_all_models.sh), but
# submitted as a real SLURM job anyway, NOT run on a login node — confirmed
# the hard way 2026-07-28: two attempts run directly on cc-login5 (one
# foreground, one properly nohup'd + disowned) both died at the exact same
# point, right after GO-DAG resolution, with no traceback in either log.
# Root cause (per the cluster's own process-control notification, not a
# guess): login nodes enforce a hard 30-MINUTE CPU-TIME LIMIT and auto-kill
# anything over it, regardless of nohup/disown — this job's real work (20
# repeats x 4 target sizes = 80 LogisticRegression fits on a 100000x2560
# matrix, plus GOA/GO-DAG parsing) comfortably exceeds that on a login node.
#
# Usage:
#   sbatch scripts/slurm/run_controlled_subsampling.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_controlled_subsampling
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/arjunc4/biolens/logs/controlled_subsampling_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/controlled_subsampling_%j.err

REPO="/u/arjunc4/biolens"

source ~/.bashrc
conda activate biolens

set -eo pipefail

export BIOLENS_DATA_DIR="/scratch/arjunc4/biolens/data/annotations"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "/scratch/arjunc4/biolens/logs"
mkdir -p "/scratch/arjunc4/biolens/eval/controlled_subsampling"

echo "Controlled subsampling on $(hostname)"

python "${REPO}/scripts/run_controlled_subsampling.py" \
    --model esm2_8m \
    --sae-checkpoint /scratch/arjunc4/biolens/checkpoints/esm2_8m_layer5_topk/esm2_8m_L5_topk_k32/final.pt \
    --expansion-factor 8 \
    --cache /scratch/arjunc4/biolens/data/activations/esm2_8m_layer5_mean \
    --n-repeats 20 \
    --output /scratch/arjunc4/biolens/eval/controlled_subsampling/esm2_8m.json \
    --device cpu

echo "Controlled subsampling complete."
