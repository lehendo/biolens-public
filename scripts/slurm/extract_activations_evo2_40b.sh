#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Evo 2 40B activation extraction — trains a second Evo 2 SAE at 40B once
# the 7B pipeline works, using the same code against a different checkpoint.
#
# ██████  DO NOT SUBMIT YET  ██████
# `transformer_engine` is NOT currently installed in the shared `biolens`
# conda env (confirmed 2026-07-25 via direct import check on the cluster).
# Arc Institute/NVIDIA document this as a hard requirement for evo2_40b —
# FP8 via Transformer Engine on a Hopper GPU, needed for correct inference,
# not just a speed optimization (source: NVIDIA NIM Evo 2 prerequisites docs,
# https://docs.nvidia.com/nim/bionemo/evo2/2.1.0/prerequisites.html). The 7B
# "light install" this env currently has deliberately skips it (see
# src/biolens/models/evo2.py's module docstring). Installing it into the
# live shared env was deliberately deferred (2026-07-25) to avoid any risk
# to jobs already running against that same env — install once safe, then
# remove this banner.
#
# GPU choice: H200, not H100 — evo2_40b needs >80GB VRAM for inference.
# On H100 (80GB) that requires 2-GPU model parallelism; on H200 (141GB) it
# fits on a SINGLE GPU (source: same NVIDIA NIM docs above), which is why
# this script (unlike a 2-GPU alternative) needs no changes to the existing
# single-GPU Evo2Model adapter. `scavenger` is the confirmed-accessible
# source of H200:8 nodes for this account (AllowAccounts=ALL, verified via
# `scontrol show partition scavenger`, 2026-07-25) — `secondary` (used by
# the 7B scripts) has no H200 capacity. scavenger is preemptible; --requeue
# is set below for that reason, same as the 7B extraction script.
#
# Layer choice: 24, not 16 (7B's choice) — 7B's layer 16 was chosen as
# mid-depth out of 32 total blocks (exactly half). 40B has 50 blocks; half
# would be 25, and 24 is one of this checkpoint's own attn_layer_idxs
# ([3,10,17,24,31,35,42,49], read directly from the packaged
# evo2-40b-1m.yml config, 2026-07-25) — close to the proportional match and
# lands on a documented attention-block index rather than an arbitrary
# nearby number. Evo2Model's `blocks.{layer}.mlp.l3` tap point is confirmed
# to exist on both hyena and attention block types for evo2_7b via live
# introspection (src/biolens/models/evo2.py) but NOT yet re-confirmed for a
# loaded evo2_40b model — do that check (same named_modules() introspection)
# as part of this script's first real run, before trusting the activations
# it produces.
#
# Batch size / shard count are first-guess, NOT empirically calibrated
# (unlike the 7B script's tuned defaults) — 40B forward passes are both
# slower per-sequence and larger per-activation-vector (d_model=8192 vs
# 4096) than 7B's. Recalibrate --batch-size and N_SHARDS from this script's
# own first real per-shard wall-clock time, the same way the 7B script's "40
# shards, doubled from 20" comment records having done.
#
# Usage (once transformer_engine is installed and the banner above is removed):
#   sbatch scripts/slurm/extract_activations_evo2_40b.sh
#
# After all array tasks complete:
#   python scripts/extract_activations.py --model evo2_40b --layer 24 \
#       --finalize --output-dir $OUTPUT_DIR
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_extract_evo2_40b
#SBATCH --partition=scavenger
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G                            # double the 7B script's 128G —
                                                # 40B's own weights (~80GB in
                                                # mixed precision) plus host-side
                                                # buffers need more headroom;
                                                # first-guess, recalibrate if a
                                                # real run shows otherwise.
#SBATCH --time=24:00:00                       # scavenger's MaxTime (1-00:00:00,
                                                # scontrol show partition
                                                # scavenger, 2026-07-25) — far
                                                # more generous than secondary's
                                                # 4h cap, useful given 40B's
                                                # slower per-sequence forward pass.
#SBATCH --array=0-39                          # same shard count as the 7B script,
                                                # first-guess — recalibrate from
                                                # real per-shard timing (see banner).
#SBATCH --output=/scratch/arjunc4/biolens/logs/extract_evo2_40b_%A_%a.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/extract_evo2_40b_%A_%a.err
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="evo2_40b"
LAYER="${LAYER:-24}"          # see banner above for why 24, not a proportional 25
POOLING="mean"
N_SHARDS=40
WINDOW_SIZE=4096               # bp per DNA window — same as 7B, for direct comparability
WINDOW_STRIDE=4096              # non-overlapping tiling
CHROMS="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX"
REFERENCE_GENOME="/scratch/arjunc4/biolens/data/reference/hg38.fa"
OUTPUT_DIR="/scratch/arjunc4/biolens/data/activations/${MODEL}_layer${LAYER}_${POOLING}"
REPO="/u/arjunc4/biolens"

# ── Environment ───────────────────────────────────────────────────────────────
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
    exit 1
fi

python -c "import transformer_engine" 2>/dev/null || {
    echo "ERROR: transformer_engine is not installed in this env — required for" >&2
    echo "evo2_40b (see this script's header banner). Do not proceed." >&2
    exit 1
}

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
    --batch-size        2                     \
    --shard-id          "${SLURM_ARRAY_TASK_ID}" \
    --n-shards          "${N_SHARDS}"         \
    --device            cuda

echo "Task ${SLURM_ARRAY_TASK_ID} complete."
