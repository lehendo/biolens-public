#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Experiment 4 — non-biology domain replication.
#
# A SINGLE job, not an array: scripts/run_experiment4_nonbio.py extracts
# Gemma-2-2B + Gemma Scope SAE features ONCE, then sweeps min_positives
# internally over the same extracted features (see that script's docstring —
# looping the whole extraction per sweep point, mirroring run_sweep_eval.sh's
# array pattern unmodified, would redundantly reload the model and re-run the
# full forward pass per sweep point for no benefit).
#
# REQUIRES a valid HuggingFace auth token with Gemma's license ACCEPTED —
# unlike Evo 2 (public arcinstitute repo), google/gemma-2-2b and the Gemma
# Scope SAE release are gated: you must accept Google's usage license on
# huggingface.co under the HF account tied to your token before this can
# download anything. Verify BEFORE submitting (fails fast, cheap, avoids
# burning a GPU allocation on an auth error):
#   HF_HOME=/scratch/arjunc4/.cache/huggingface python -c \
#     "from huggingface_hub import model_info; \
#      print(model_info('google/gemma-2-2b').id)"
# If that raises GatedRepoError/401, request access at
# https://huggingface.co/google/gemma-2-2b and https://huggingface.co/google/gemma-scope-2b-pt-res
# and set HF_TOKEN (or run `huggingface-cli login`) before retrying.
#
# IllinoisComputes-GPU (A100:4), not secondary/ic-express: two real attempts
# already failed on the 4-hour `secondary` cap — job 9321451 (a MIG slice on
# ic-express, scancelled at 4h15m as a precaution) and job 9322709 (a full
# H100 on `secondary`, which genuinely TIMEOUT'd at exactly 4h even at full,
# non-MIG-sliced compute, 2026-07-04). The real fix is the --batch-size bump
# in run_experiment4_nonbio.py (16 -> 128, was leaving most of the GPU idle),
# but IllinoisComputes-GPU's 3-day MaxTime removes the throughput guesswork
# entirely rather than gambling on a second partition's shorter cap again.
# Confirmed usable via `groups $USER` (acc-IllinoisComputes membership) —
# re-verify with `scontrol show partition IllinoisComputes-GPU | grep -iE
# "allowgroups|allowaccounts"` if this ever gets rejected. --time was
# originally set to the full 3-day cap (2026-07-04, before any real timing
# data existed) as maximum safety margin after two timeouts on
# shorter-capped partitions. Lowered to 12h on 2026-07-08 after two real
# completed runs (job 9358678, job 9454852's predecessor) both finished in
# ~4h15m — the 3-day request started colliding with a 2026-07-15 cluster
# maintenance reservation (`ReqNodeNotAvail, Reserved for maintenance`),
# since Slurm won't place a job whose requested walltime could still be
# running when a reservation starts. 12h keeps a real 3x margin over
# observed runtime while staying clear of that reservation.
#
# Usage:
#   sbatch scripts/slurm/run_experiment4.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_experiment4
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00      # two real completed runs (2026-07-07, 2026-07-08)
                              # both took ~4h15m; 12h is a 3x margin without
                              # colliding with the 2026-07-15 maintenance
                              # reservation the original 3-day cap ran into
#SBATCH --output=/scratch/arjunc4/biolens/logs/experiment4_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/experiment4_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
SAE_RELEASE="gemma-scope-2b-pt-res"
SAE_ID="layer_12/width_16k/average_l0_82"
LAYER=12
MIN_POSITIVES_LIST="10,20,50,100,150,200,300"   # matches configs/eval.yaml::probing.sweep_min_positives
OUTPUT_DIR="/scratch/arjunc4/biolens/results/experiment4_gemma_scope_L${LAYER}"
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
                     # the script or produce a nonzero job exit code — exactly
                     # what happened on the first real run of this script:
                     # crashed on a missing sae_lens import, but the trailing
                     # echo still printed "complete" and sacct showed
                     # ExitCode 0:0 (2026-07-03). Caught only because the
                     # actual output files were checked instead of trusting
                     # that message.

export HF_HOME="/scratch/arjunc4/.cache/huggingface"
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "/scratch/arjunc4/biolens/logs"

echo "Experiment 4 (Gemma Scope) on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Run ───────────────────────────────────────────────────────────────────────
python "${REPO}/scripts/run_experiment4_nonbio.py" \
    --sae-release           "${SAE_RELEASE}"        \
    --sae-id                "${SAE_ID}"              \
    --layer                 "${LAYER}"               \
    --min-positives-list    "${MIN_POSITIVES_LIST}"  \
    --held-out-baselines                             \
    --held-out-fraction     0.5                      \
    --permuted-label-control                         \
    --output-dir            "${OUTPUT_DIR}"          \
    --device                cuda

echo "Experiment 4 complete: ${OUTPUT_DIR}"
