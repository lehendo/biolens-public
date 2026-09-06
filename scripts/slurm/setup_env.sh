#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SLURM wrapper for scripts/setup_env.sh — runs the full environment
# setup/rebuild on a real GPU node instead of a login node.
#
# Required: login nodes on the Illinois Campus Cluster hard-kill any process
# at 30 minutes of CPU time, and the flash-attn from-source compile
# (scripts/setup_env.sh step 4) took 2h46m25s in the one real timed run of
# this exact build (job 9809994, 2026-08) — it WILL get killed partway
# through if run on a login node.
#
# Usage:
#   sbatch scripts/slurm/setup_env.sh [conda-env-name]
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=biolens_setup_env
#SBATCH --partition=secondary
#SBATCH --account=jimeng-ic
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/arjunc4/biolens/logs/setup_env_%j.out
#SBATCH --error=/scratch/arjunc4/biolens/logs/setup_env_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$REPO_ROOT/scripts/setup_env.sh" "$@"
