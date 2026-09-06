#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Full environment setup/rebuild, from nothing to a working biolens install.
#
# A plain `pip install -e '.[dev,borzoi]'` is NOT sufficient on its own — see
# pyproject.toml's `torch` and `flash-attn` entries and README.md's
# Installation section for the full story. Both gotchas are real, confirmed,
# and cost real time to diagnose the first two times (conda env corruption
# masquerading as the root cause, twice, before the actual issue -- wrong
# torch CUDA build + a prebuilt flash-attn wheel with a mismatched C++ ABI --
# was found). This script exists so nobody has to rediscover that by hand
# again: it runs the exact three-step sequence, in the right order, with the
# right flags.
#
# MUST run on a GPU node with a CUDA 12.8 toolkit available, not a login
# node: step 3 (flash-attn) is a genuine from-source CUDA kernel compile,
# observed taking 1-3+ hours of real wall time and 128G+ memory (job
# 9809994, 2026-08: 2h46m25s). On the Illinois Campus Cluster specifically,
# login nodes hard-kill any process at 30 minutes of CPU time regardless of
# nohup/disown -- this WILL get killed partway through the compile if run
# there. Use `sbatch scripts/slurm/setup_env.sh` (wraps this script with the
# right SBATCH directives), or run interactively on an already-allocated GPU
# node.
#
# Usage:
#   ./scripts/setup_env.sh [conda-env-name]     # default env name: biolens
#
# Env vars (optional overrides):
#   CUDA_MODULE     Module to `module load` for the CUDA toolkit (default: cuda/12.8)
#   CUDA_HOME_PATH  Value to export as CUDA_HOME (default: /sw/apps/cuda/12.8)
#   SKIP_CONDA      Set to 1 to skip conda env create/activate entirely (use
#                   the currently-active Python environment as-is -- for a
#                   non-conda setup, e.g. a plain venv already activated).
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_NAME="${1:-biolens}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
CUDA_HOME_PATH="${CUDA_HOME_PATH:-/sw/apps/cuda/12.8}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================================="
echo "biolens environment setup — repo root: $REPO_ROOT"
echo "=================================================================="

# ── Step 0: conda env ───────────────────────────────────────────────────────
if [[ "${SKIP_CONDA:-0}" == "1" ]]; then
    echo "[0/4] SKIP_CONDA=1 — using the currently-active Python environment as-is."
    python -c "import sys; print('Python:', sys.executable, sys.version)"
else
    if ! command -v conda &>/dev/null; then
        echo "ERROR: conda not found on PATH. Install conda/miniconda first, or set" >&2
        echo "SKIP_CONDA=1 to use an already-active environment instead." >&2
        exit 1
    fi

    # The root cause of two real conda-corruption incidents this project hit
    # (2026-08) was ~/.conda being a symlink into
    # /scratch, whose files get purged after 30 days of no access — a
    # structural trap, not something this script can silently work around.
    # Warn, don't fail: this script should stay portable to machines/clusters
    # without that specific purge policy.
    if [[ -L "$HOME/.conda" ]]; then
        real_path="$(readlink -f "$HOME/.conda" 2>/dev/null || true)"
        if [[ "$real_path" == /scratch/* ]]; then
            echo "WARNING: \$HOME/.conda is a symlink into $real_path." >&2
            echo "WARNING: on the Illinois Campus Cluster, /scratch has a 30-day" >&2
            echo "WARNING: access-time purge policy that WILL silently corrupt this" >&2
            echo "WARNING: environment again (this happened twice already)." >&2
            echo "WARNING: Move ~/.conda to real home storage" >&2
            echo "WARNING: before continuing: rm ~/.conda && mkdir -p ~/.conda" >&2
            echo "WARNING: Continuing anyway in 10s (Ctrl-C to abort)..." >&2
            sleep 10
        fi
    fi

    echo "[0/4] conda env: $ENV_NAME"
    if conda env list | grep -qE "^${ENV_NAME}\s"; then
        echo "      Env already exists — reusing it."
    else
        echo "      Creating fresh env with Python 3.10..."
        conda create -y -n "$ENV_NAME" python=3.10
    fi

    # `conda activate` inside a non-interactive script needs conda's shell
    # hook sourced first — plain `conda activate` fails otherwise.
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    echo "      Activated: $(python -c 'import sys; print(sys.executable)')"
fi

# ── Step 1: CUDA toolkit module (cluster-specific, best-effort) ────────────
echo "[1/4] CUDA toolkit"
if command -v module &>/dev/null; then
    if module load "$CUDA_MODULE" 2>/dev/null; then
        echo "      Loaded module: $CUDA_MODULE"
    else
        echo "      WARNING: 'module load $CUDA_MODULE' failed — continuing," >&2
        echo "      but the flash-attn build in step 3 needs a real CUDA 12.8" >&2
        echo "      toolkit on PATH/CUDA_HOME. Set CUDA_MODULE= to the correct" >&2
        echo "      module name for this machine if it differs." >&2
    fi
else
    echo "      No 'module' command found (not an Lmod/environment-modules" >&2
    echo "      cluster) — assuming a CUDA 12.8 toolkit is already on PATH." >&2
fi
export CUDA_HOME="$CUDA_HOME_PATH"
echo "      CUDA_HOME=$CUDA_HOME"

# ── Step 2: base install ────────────────────────────────────────────────────
echo "[2/4] pip install -e '.[dev,borzoi]'"
pip install -e ".[dev,borzoi]"

# ── Step 3: torch, forced to the correct CUDA build ────────────────────────
# Plain PyPI serves torch==2.11.0+cu130 under this exact version string —
# the +cu128 build this project needs only exists on PyTorch's own index.
# See pyproject.toml's `torch` entry for the full, confirmed story.
echo "[3/4] torch==2.11.0+cu128 (forcing the correct CUDA build)"
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# ── Step 4: flash-attn, forced to build from source ────────────────────────
# The prebuilt PyPI wheel is compiled against a different torch C++ ABI and
# fails at IMPORT time (not install time) with
# `undefined symbol: ...materialize_cow_storage...`. See pyproject.toml's
# `flash-attn` entry for the full, confirmed story. This is a real from-
# source CUDA kernel compile — slow (1-3+ hours) and memory-hungry (128G+).
echo "[4/4] flash-attn==2.8.3.post1 (building from source — this takes 1-3+ hours)"
pip install flash-attn==2.8.3.post1 \
    --no-build-isolation --no-binary flash-attn \
    --no-cache-dir --force-reinstall --no-deps

# ── Verify ───────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "Verifying the full chain..."
python -c "
import torch, evo2, biolens
print('IMPORT CHECK PASSED')
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
import flash_attn
print('flash_attn:', flash_attn.__version__)
"
echo "=================================================================="
echo "Setup complete."
