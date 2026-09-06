#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3 — sample-size sweep evaluation, one SLURM array task per
# min_positives value (pre-specified before the confirmatory run, informed
# by the Phase 0 pilot).
#
# Writes each sweep point's results to ${OUTPUT_ROOT}/mp<N>/go_probing_results.json
# — the exact directory convention scripts/fit_sweep.py expects.
#
# Usage:
#   sbatch --export=MODEL=esm2_8m,LAYER=5,CHECKPOINT=/path/to/final.pt,\
#CACHE_DIR=/path/to/cache,OUTPUT_ROOT=/path/to/sweep_results \
#       scripts/slurm/run_sweep_eval.sh
#
#   Add EXPANSION_FACTOR=<N> to the --export list ONLY for a checkpoint that
#   predates saved sae_cfg metadata (run_eval.py's build_sae_from_checkpoint
#   call will raise a clear ValueError naming this if one is needed and
#   missing) — e.g. the original Phase 0 esm2_8m_layer5_topk checkpoint.
#   Every checkpoint trained since (train_sae_multiscale.sh's outputs)
#   already carries its own sae_cfg and does not need this.
#
#   Add MAX_EVAL_SEQS=<N> to override run_eval.py's default (10000) —
#   needed because the default leaves too few GO terms eligible at high
#   min_positives to trust the mean_auroc_inflation estimate there: backfill
#   of the original 10000-seq sweeps (2026-07-14, all 9 ESM2 configs,
#   byte-identical row counts across every one, confirming this is a
#   property of the eval slice size, not any model/SAE config) found only
#   2 GO terms eligible at min_positives=300, down from 133 at
#   min_positives=10 — a cluster-bootstrap CI from 2 points is barely
#   informative. 100000 (10x) gives real headroom.
#
# After all 7 array tasks complete:
#   python scripts/fit_sweep.py --sweep-dir $OUTPUT_ROOT \
#       --verified-features configs/verified_features/<...>.yaml \
#       --output docs/experiment3_sweep_fit.json
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_sweep_eval
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-6                          # 7 sweep points: {10,20,50,100,150,200,300}
#SBATCH --output=/scratch/arjunc4/biolens/logs/sweep_eval_%A_%a.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/sweep_eval_%A_%a.err
#SBATCH --requeue

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="${MODEL:?Set MODEL via --export}"
LAYER="${LAYER:?Set LAYER via --export}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT via --export=CHECKPOINT=/path/to/final.pt}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR via --export}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT via --export}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-}"   # only needed for pre-sae_cfg checkpoints — see usage note above
MAX_EVAL_SEQS="${MAX_EVAL_SEQS:-}"         # optional override — see usage note above
N_PERMUTATIONS="${N_PERMUTATIONS:-}"       # optional override for run_eval.py's --n-permutations
                                            # (default 20) — added 2026-07-19 after the default
                                            # cost (1 real + 20 permuted-label probing passes,
                                            # ~21x a single pass) turned out prohibitive for
                                            # esm2_650m at 100K eval (d_sae=10240, the largest
                                            # SAE + eval-size combination of any model): a single
                                            # min_positives task projected to ~40 days at the
                                            # default. Lower this for large/expensive configs.
REPO="/u/arjunc4/biolens"

# Pre-specified sweep points (configs/eval.yaml::probing.sweep_min_positives) —
# must match this array exactly; the bash array index must line up with
# --array=0-6 above.
SWEEP_POINTS=(10 20 50 100 150 200 300)
MP="${SWEEP_POINTS[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_DIR="${OUTPUT_ROOT}/mp${MP}"

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

echo "min_positives=${MP} (task ${SLURM_ARRAY_TASK_ID}/7) on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Evaluate ──────────────────────────────────────────────────────────────────
EXPANSION_FACTOR_ARGS=()
if [[ -n "${EXPANSION_FACTOR}" ]]; then
    EXPANSION_FACTOR_ARGS=(--expansion-factor "${EXPANSION_FACTOR}")
fi

MAX_EVAL_SEQS_ARGS=()
if [[ -n "${MAX_EVAL_SEQS}" ]]; then
    MAX_EVAL_SEQS_ARGS=(--max-eval-seqs "${MAX_EVAL_SEQS}")
fi

N_PERMUTATIONS_ARGS=()
if [[ -n "${N_PERMUTATIONS}" ]]; then
    N_PERMUTATIONS_ARGS=(--n-permutations "${N_PERMUTATIONS}")
fi

# --held-out-baselines: standard + hard-negative held-out AUROC (round 4/5, W3)
# --permuted-label-control: noise-vs-confounding decomposition (round 4, Issue 2)
python "${REPO}/scripts/run_eval.py" \
    --model                 "${MODEL}"          \
    --layer                 "${LAYER}"          \
    --sae-checkpoint        "${CHECKPOINT}"     \
    --cache                 "${CACHE_DIR}"      \
    --output-dir            "${OUTPUT_DIR}"     \
    --min-positives         "${MP}"             \
    --held-out-baselines                        \
    --held-out-fraction     0.5                 \
    --permuted-label-control                    \
    --wandb-project         biolens             \
    --run-name              "sweep_${MODEL}_L${LAYER}_mp${MP}" \
    --device                cuda                 \
    "${N_PERMUTATIONS_ARGS[@]}"                  \
    "${EXPANSION_FACTOR_ARGS[@]}"                \
    "${MAX_EVAL_SEQS_ARGS[@]}"

echo "min_positives=${MP} complete: ${OUTPUT_DIR}"
