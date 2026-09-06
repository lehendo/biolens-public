#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Evo 2 activation extraction — genomic-annotation probing windows
# (used by Experiment 1b).
#
# Requires the `evo2` package (pip install evo2 — see src/biolens/models/evo2.py
# for the "untested on this machine, first real validation happens here" note)
# and a downloaded, indexed reference genome FASTA (GRCh38 primary assembly).
#
# H100, not A30 (unlike the ESM2 extraction scripts): Evo 2 7B's real memory
# footprint (7B params + activations, at the conservative 8192-token context
# this project uses per configs/models.yaml) needs more headroom than A30's
# 24GB gives. H100's 80GB is comfortably sufficient. NOT H200: as of the last
# `sinfo` check on this cluster (2026-07-03), H200 only exists under the
# `scavenger` partition (preemptible — jobs can be killed mid-run if reclaimed),
# while `secondary` has substantial reliable H100 capacity (5 nodes x 8 GPUs).
# Re-check `sinfo -N -o "%25P %15N %25G %10t" | grep -i gpu` before assuming
# this is still accurate — cluster GPU inventory changes over time. Also
# avoid V100 specifically on this cluster: a real compute-capability mismatch
# with this project's PyTorch build was already hit and fixed once (Phase 0).
#
# Usage:
#   # One-time: download + index the reference genome (run on a login node,
#   # not inside the SLURM job — this is a ~3GB download, do it once):
#   wget -P /scratch/arjunc4/biolens/data/reference \
#       https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
#   gunzip /scratch/arjunc4/biolens/data/reference/hg38.fa.gz
#
#   sbatch scripts/slurm/extract_activations_evo2.sh
#
# After all array tasks complete:
#   python scripts/extract_activations.py --model evo2_7b --layer 16 \
#       --finalize --output-dir $OUTPUT_DIR
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_extract_evo2
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00                       # secondary's hard MaxTime
                                                # (scontrol show partition secondary,
                                                # 2026-07-03) — using the full cap,
                                                # not less: requesting under the max
                                                # doesn't add safety margin, it just
                                                # removes usable compute time.
#SBATCH --array=0-39                          # 40 shards (doubled from 20) so each
                                                # task's workload comfortably fits
                                                # inside the 4h partition cap.
#SBATCH --output=/scratch/arjunc4/biolens/logs/extract_evo2_%A_%a.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/extract_evo2_%A_%a.err
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="evo2_7b"
LAYER="${LAYER:-16}"          # mid-depth, per configs/eval.yaml::probe_layers.evo2_7b
POOLING="mean"
N_SHARDS=40
WINDOW_SIZE=4096               # bp per DNA window
WINDOW_STRIDE=4096              # non-overlapping tiling
CHROMS="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX"
REFERENCE_GENOME="/scratch/arjunc4/biolens/data/reference/hg38.fa"
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

if [[ ! -f "${REFERENCE_GENOME}" ]]; then
    echo "ERROR: reference genome not found at ${REFERENCE_GENOME}" >&2
    echo "Download it first (see usage comment at top of this script)." >&2
    exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}/${N_SHARDS} on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Extract ───────────────────────────────────────────────────────────────────
python "${REPO}/scripts/extract_activations.py" \
    --model             "${MODEL}"            \
    --layer             "${LAYER}"            \
    --pooling           "${POOLING}"          \
    --output-dir        "${OUTPUT_DIR}"       \
    --data-source       genomic               \
    --reference-genome  "${REFERENCE_GENOME}" \
    --window-size       "${WINDOW_SIZE}"      \
    --window-stride     "${WINDOW_STRIDE}"    \
    --chroms            "${CHROMS}"           \
    --batch-size        4                     \
    --shard-id          "${SLURM_ARRAY_TASK_ID}" \
    --n-shards          "${N_SHARDS}"         \
    --device            cuda

echo "Task ${SLURM_ARRAY_TASK_ID} complete."
