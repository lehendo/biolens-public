#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Geuvadis case study.
#
# Requires, all already produced by earlier steps:
#   - A trained Evo 2 SAE checkpoint (scripts/slurm/train_sae_evo2.sh)
#   - A gene-locus naive-probing run's naive_feature_results.json (naive
#     best-AUROC features per locus — scripts/run_geuvadis_naive_probing.py)
#   - A genomic verified_features registry for that checkpoint
#     (scripts/verify_geuvadis_naive_features.py — a real one exists as of
#     job 9724784, 2026-07-30: 0/75 features confirmed, see that script's
#     header. run_geuvadis_case_study.py handles that gracefully — runs the
#     naive arm alone with its verification verdict attached as metadata,
#     rather than skipping every locus.)
#   - Downloaded: 1000 Genomes VCF (Geuvadis genotyped subset), Geuvadis
#     expression matrix, Geuvadis' own published cis-eQTL candidate list —
#     all public, no dbGaP gate (unlike GTEx, Papers 2/3)
#   - A UCSC liftOver chain file, hg19ToHg38.over.chain.gz (REQUIRED, not
#     optional — confirmed 2026-07-31 that the eQTL candidate file and 1000
#     Genomes Phase 3 VCF are GRCh37-coordinate while the reference genome/
#     GENCODE/Borzoi are all GRCh38; see biolens.data.reference_genome.
#     load_liftover_chain's docstring for the exact discrepancy that caught
#     this — job 9727212 "completed" 6 loci that were silently using the
#     WRONG genomic position, passing the ref-base check by pure chance,
#     not because their coordinates were actually correct)
#   - 1000 Genomes' own published sample-to-population panel (e.g.
#     integrated_call_samples_v3.20130502.ALL.panel from
#     http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/) — REQUIRED,
#     not optional: ancestry-covariate adjustment for population
#     stratification, the biggest open item in the original design (see
#     biolens.data.geuvadis.build_ancestry_covariates's docstring).
#
# Rebuilt 2026-08 (see run_geuvadis_case_study.py's module docstring): the
# mediator is now each individual's own SAE feature activation on their real
# personal haplotype sequence, not a single locus-level scalar — real per-
# locus cost is up to ~2*n_samples Evo 2 forward passes (n~445), not 2. Use
# MAX_LOCI for a cheap validation run before committing the full candidate
# list to this partition's 4h cap.
#
# Usage:
#   sbatch --export=ALL,\
#SAE_CHECKPOINT=/path/to/final.pt,LAYER=16,\
#VCF=/path/to/geuvadis.vcf.gz,EXPRESSION=/path/to/expr.tsv,\
#EQTL_CANDIDATES=/path/to/eqtls.tsv,\
#NAIVE_RESULTS=/path/to/naive_feature_results.json,\
#GENOMIC_VERIFIED_FEATURES=configs/verified_features/evo2_7b_layer16_topk_k32_geuvadis_naive.yaml,\
#LIFTOVER_CHAIN=/path/to/hg19ToHg38.over.chain.gz,\
#POPULATION_LABELS=/path/to/integrated_call_samples_v3.20130502.ALL.panel,\
#MAX_LOCI=3,SKIP_BORZOI=1 \
#       scripts/slurm/run_geuvadis_case_study.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_geuvadis
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00                       # secondary's hard MaxTime
                                                # (scontrol show partition secondary,
                                                # 2026-07-03) — using the full cap;
                                                # chr22-scoped case study, expected
                                                # lighter than genome-wide extraction,
                                                # but revisit if it doesn't finish.
#SBATCH --output=/scratch/arjunc4/biolens/logs/geuvadis_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/geuvadis_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="${MODEL:-evo2_7b}"
LAYER="${LAYER:?Set LAYER via --export}"
SAE_CHECKPOINT="${SAE_CHECKPOINT:?Set SAE_CHECKPOINT via --export}"
REFERENCE_GENOME="${REFERENCE_GENOME:-/scratch/arjunc4/biolens/data/reference/hg38.fa}"
VCF="${VCF:?Set VCF via --export}"
EXPRESSION="${EXPRESSION:?Set EXPRESSION via --export}"
EQTL_CANDIDATES="${EQTL_CANDIDATES:?Set EQTL_CANDIDATES via --export}"
NAIVE_RESULTS="${NAIVE_RESULTS:?Set NAIVE_RESULTS via --export}"
GENOMIC_VERIFIED_FEATURES="${GENOMIC_VERIFIED_FEATURES:?Set GENOMIC_VERIFIED_FEATURES via --export}"
LIFTOVER_CHAIN="${LIFTOVER_CHAIN:-/scratch/arjunc4/biolens/data/reference/hg19ToHg38.over.chain.gz}"
# 1000 Genomes' own published sample-to-population panel — required for
# ancestry-covariate adjustment (see run_geuvadis_case_study.py's module
# docstring and --population-labels help).
POPULATION_LABELS="${POPULATION_LABELS:?Set POPULATION_LABELS via --export (e.g. integrated_call_samples_v3.20130502.ALL.panel from http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/)}"
CONTEXT_BP="${CONTEXT_BP:-4096}"
# Rebuilt pipeline (2026-08) does up to ~2*n_samples Evo 2 forward passes per
# locus (n~445 diploid samples), not 2 — see run_geuvadis_case_study.py's
# module docstring. ACTIVATION_BATCH_SIZE tunes the per-chunk size of that;
# MAX_LOCI lets a cheap validation run process only the first few loci
# before committing the full candidate list to this partition's 4h cap.
ACTIVATION_BATCH_SIZE="${ACTIVATION_BATCH_SIZE:-32}"
MAX_LOCI="${MAX_LOCI:-}"
# Set SKIP_BORZOI=1 via --export when borzoi-pytorch isn't installed on this
# environment yet (confirmed not installed as of 2026-07-30) — otherwise
# every locus would fail on the Borzoi baseline specifically while the
# naive/verified arms and other baselines above it are unaffected either way.
SKIP_BORZOI="${SKIP_BORZOI:-0}"
OUTPUT_DIR="/scratch/arjunc4/biolens/results/geuvadis_case_study_${MODEL}_L${LAYER}"
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

echo "Geuvadis case study: model=${MODEL} layer=${LAYER} on $(hostname) — GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Run ───────────────────────────────────────────────────────────────────────
BORZOI_FLAG=()
if [ "${SKIP_BORZOI}" = "1" ]; then
    BORZOI_FLAG=(--skip-borzoi)
fi

MAX_LOCI_FLAG=()
if [ -n "${MAX_LOCI}" ]; then
    MAX_LOCI_FLAG=(--max-loci "${MAX_LOCI}")
fi

python "${REPO}/scripts/run_geuvadis_case_study.py" \
    --sae-checkpoint         "${SAE_CHECKPOINT}"      \
    --model                  "${MODEL}"               \
    --layer                  "${LAYER}"               \
    --reference-genome       "${REFERENCE_GENOME}"    \
    --vcf                    "${VCF}"                 \
    --expression             "${EXPRESSION}"           \
    --eqtl-candidates        "${EQTL_CANDIDATES}"     \
    --naive-feature-results  "${NAIVE_RESULTS}"       \
    --genomic-verified-features "${GENOMIC_VERIFIED_FEATURES}" \
    --liftover-chain         "${LIFTOVER_CHAIN}"      \
    --population-labels      "${POPULATION_LABELS}"   \
    --context-bp             "${CONTEXT_BP}"          \
    --activation-batch-size  "${ACTIVATION_BATCH_SIZE}" \
    --output-dir             "${OUTPUT_DIR}"          \
    --device                 cuda                     \
    "${BORZOI_FLAG[@]}"                                \
    "${MAX_LOCI_FLAG[@]}"

echo "Geuvadis case study complete: ${OUTPUT_DIR}"
