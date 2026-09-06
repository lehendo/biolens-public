#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Gene-locus-level naive-feature probing for the Geuvadis case study —
# produces the naive_feature_results.json that
# scripts/run_geuvadis_case_study.py's --naive-feature-results requires.
#
# NOT the same as run_experiment1b_evo2_genomic.py's cCRE-class probing —
# that script's genomic_probing_results.json is keyed by cCRE class
# (dELS/pELS/PLS/...), but the Geuvadis case study needs a naive best-AUROC
# feature per eQTL candidate GENE. This is a separate, gene-locus-scoped
# probing pass: for each candidate gene, tiles windows across its own gene
# body (+ promoter-proximal flank) rather than genome-wide tiling.
#
# A SINGLE job, not an array — same reasoning as run_experiment1b_evo2_genomic.sh.
#
# Real incident, 2026-07-28 (job 9701717): the first version of this script
# used WINDOW_STRIDE=512 with WINDOW_SIZE=4096 (8x overlap between
# consecutive windows) and computed multivariate_auroc by default — the
# combination produced 66,516 windows and TIMED OUT at the 4h wall-clock
# limit with the Evo 2 forward pass alone consuming 3.5h and probing barely
# started. Fixed two ways: WINDOW_STRIDE now defaults to WINDOW_SIZE
# (non-overlapping — also the more defensible design, not just faster,
# since >85%-overlapping windows mostly inflate the visible positive count
# per gene without adding real sequence diversity), and
# scripts/run_geuvadis_naive_probing.py now passes
# compute_multivariate_auroc=False (this script's only consumer,
# run_geuvadis_case_study.py's --naive-feature-results loader, never reads
# multivariate_auroc — it was pure wasted compute for this specific caller).
#
# REQUIRES: the reference genome (already on the server), a trained Evo 2 SAE
# checkpoint, and the Geuvadis eQTL candidate file (already on the server at
# /scratch/arjunc4/biolens/data/geuvadis/EUR373.gene.cis.FDR5.best.rs137.txt.gz).
# GENCODE gene annotations are downloaded automatically on first run (cached
# under BIOLENS_DATA_DIR) if not already present.
#
# Partition, 2026-07-29: the resubmitted job (9707872) sat PENDING for ~8h
# on `secondary` — every one of that partition's H100 nodes showed `mix` in
# `sinfo -N -o "%25P %15N %25G %10t" | grep -i gpu | grep -i mix`. First
# tried switching to `ic-express`, which had a coarsely `idle` H100 node —
# but `scontrol show node` revealed that "H100" was actually MIG-partitioned
# into 16 slices of 1g.20gb each (`gpu:nvidia_h100_80gb_hbm3_1g.20gb:16`),
# not a full 80GB card — sbatch rejected the plain `--gres=gpu:H100:1`
# request outright ("Requested node configuration is not available"), and
# even a corrected MIG-slice request would likely be too small for Evo 2 7B
# (the real successful run needed close to the full 80GB card). Reverted to
# `secondary` instead: `scontrol show node` on each of its H100 nodes found
# ccc0424 had only 1/8 GPUs allocated (7 genuinely free) despite showing
# `mix` in the coarse sinfo view — sinfo's idle/mix label is node-level, not
# GPU-level, so a partially-used multi-GPU node can still have real free
# capacity a plain single-GPU request will schedule onto. Lesson: for
# multi-GPU nodes, check `scontrol show node <name>` for
# AllocTRES gres/gpu=N vs the node's total Gres=gpu:...:N, not just the
# sinfo idle/mix label, before ruling a "mix" node out.
#
# Usage:
#   sbatch --export=SAE_CHECKPOINT=/scratch/arjunc4/biolens/checkpoints/evo2_7b_layer16_topk/evo2_7b_L16_topk_k32/final.pt \
#       scripts/slurm/run_geuvadis_naive_probing.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_geuvadis_naive_probing
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/arjunc4/biolens/logs/geuvadis_naive_probing_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/geuvadis_naive_probing_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="evo2_7b"
LAYER="${LAYER:-16}"
SAE_CHECKPOINT="${SAE_CHECKPOINT:?Set SAE_CHECKPOINT via --export}"
REFERENCE_GENOME="/scratch/arjunc4/biolens/data/reference/hg38.fa"
EQTL_CANDIDATES="/scratch/arjunc4/biolens/data/geuvadis/EUR373.gene.cis.FDR5.best.rs137.txt.gz"
MAX_CANDIDATE_GENES="${MAX_CANDIDATE_GENES:-500}"
WINDOW_SIZE="${WINDOW_SIZE:-4096}"
WINDOW_STRIDE="${WINDOW_STRIDE:-4096}"
FLANK_BP="${FLANK_BP:-2000}"
MIN_POSITIVES="${MIN_POSITIVES:-5}"
OUTPUT="/scratch/arjunc4/biolens/eval/geuvadis_naive_probing/naive_feature_results.json"
REPO="/u/arjunc4/biolens"

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate biolens

set -eo pipefail

export HF_HOME="/scratch/arjunc4/.cache/huggingface"
export BIOLENS_DATA_DIR="/scratch/arjunc4/biolens/data/annotations"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "$(dirname "${OUTPUT}")"
mkdir -p "/scratch/arjunc4/biolens/logs"

if [[ ! -f "${REFERENCE_GENOME}" ]]; then
    echo "ERROR: reference genome not found at ${REFERENCE_GENOME}" >&2
    exit 1
fi
if [[ ! -f "${EQTL_CANDIDATES}" ]]; then
    echo "ERROR: eQTL candidates file not found at ${EQTL_CANDIDATES}" >&2
    exit 1
fi

echo "Geuvadis naive-feature probing on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Run ───────────────────────────────────────────────────────────────────────
python "${REPO}/scripts/run_geuvadis_naive_probing.py" \
    --model               "${MODEL}"              \
    --layer               "${LAYER}"              \
    --sae-checkpoint      "${SAE_CHECKPOINT}"     \
    --reference-genome    "${REFERENCE_GENOME}"   \
    --eqtl-candidates     "${EQTL_CANDIDATES}"    \
    --max-candidate-genes "${MAX_CANDIDATE_GENES}" \
    --window-size         "${WINDOW_SIZE}"        \
    --window-stride       "${WINDOW_STRIDE}"      \
    --flank-bp             "${FLANK_BP}"           \
    --min-positives        "${MIN_POSITIVES}"      \
    --output               "${OUTPUT}"             \
    --device               cuda

echo "Geuvadis naive-feature probing complete: ${OUTPUT}"
