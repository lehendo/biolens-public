#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Evo 2 genomic-feature verification —
# scripts/verify_geuvadis_naive_features.py. Builds the first-ever Evo 2
# verified_features registry from job 9712776's naive_feature_results.json.
#
# No GPU needed (SAE-encoding an already-extracted whole-genome activation
# cache is cheap CPU work, same reasoning as run_controlled_subsampling.sh),
# but submitted as a real SLURM job anyway, NOT run on a login node — a real
# 30-minute login-node CPU-time cap has killed direct attempts at this class
# of job before (see run_controlled_subsampling.sh's header for the original
# incident). This job scans a larger cache (710,518 whole-genome windows vs.
# controlled_subsampling's single-GO-term activation set) plus a full GENCODE
# gene-overlap search, so it's very unlikely to fit under that cap either.
#
# Usage:
#   sbatch scripts/slurm/verify_geuvadis_naive_features.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_verify_geuvadis_naive
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/arjunc4/biolens/logs/verify_geuvadis_naive_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/verify_geuvadis_naive_%j.err

REPO="/u/arjunc4/biolens"

source ~/.bashrc
conda activate biolens

set -eo pipefail

export BIOLENS_DATA_DIR="/scratch/arjunc4/biolens/data/annotations"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "/scratch/arjunc4/biolens/logs"

echo "Verifying Geuvadis naive-feature results on $(hostname)"

python "${REPO}/scripts/verify_geuvadis_naive_features.py" \
    --naive-results /scratch/arjunc4/biolens/eval/geuvadis_naive_probing/naive_feature_results.json \
    --sae-checkpoint /scratch/arjunc4/biolens/checkpoints/evo2_7b_layer16_topk/evo2_7b_L16_topk_k32/final.pt \
    --activation-cache /scratch/arjunc4/biolens/data/activations/evo2_7b_layer16_mean \
    --top-k 20 \
    --output "${REPO}/configs/verified_features/evo2_7b_layer16_topk_k32_geuvadis_naive.yaml" \
    --evidence-output /scratch/arjunc4/biolens/eval/geuvadis_naive_probing/verification_evidence.json \
    --device cpu

echo "Verification complete."
