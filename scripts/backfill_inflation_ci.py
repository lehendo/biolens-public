"""
Retroactively compute mean_auroc_inflation_ci95 for already-completed sweep
runs, without rerunning anything on GPU.

The bootstrap CI added to run_eval.py/run_experiment4_nonbio.py's summary
output only needs (auroc_inflation, best_feature_idx) pairs, which were
ALREADY saved in every
go_probing_results.json produced by a --held-out-baselines run, past or
present — no new extraction, no new SAE forward pass, no GPU. This script
reads those existing files and writes the CI directly into the existing
go_probing_summary.json (adding new keys, not touching anything else already
there), so past runs get the same statistical treatment as future ones
without a costly resubmission.

Handles two directory layouts automatically:
  - A single eval output dir (go_probing_results.json directly inside it) —
    e.g. a Phase-0-style scripts/run_eval.py run without a sweep.
  - A sweep directory containing mp<N>/go_probing_results.json
    subdirectories — e.g. scripts/slurm/run_sweep_eval.sh or
    scripts/slurm/run_experiment4.sh output.

Usage:
  python scripts/backfill_inflation_ci.py --dir /scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k32_x8
  python scripts/backfill_inflation_ci.py --dir /scratch/arjunc4/biolens/results/experiment4_gemma_scope_L12
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.eval.statistics import bootstrap_mean_auroc_inflation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def backfill_one_dir(output_dir: Path) -> bool:
    """
    Backfill a single directory containing go_probing_results.json (and,
    if it exists, go_probing_summary.json to update in place). Returns True
    if a CI was computed and written, False if skipped (no results file, or
    no rows with auroc_inflation present — e.g. a run without
    --held-out-baselines).
    """
    results_path = output_dir / "go_probing_results.json"
    if not results_path.exists():
        return False

    with open(results_path) as f:
        rows = json.load(f)

    inflation_rows = [
        (row["auroc_inflation"], row["best_feature_idx"])
        for row in rows
        if row.get("auroc_inflation") is not None
    ]
    if not inflation_rows:
        logger.info("%s: no rows with auroc_inflation — skipping (was this run "
                    "with --held-out-baselines?)", output_dir)
        return False

    inflations, feature_idx = zip(*inflation_rows)
    ci_result = bootstrap_mean_auroc_inflation(np.array(inflations), np.array(feature_idx))

    summary_path = output_dir / "go_probing_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    summary["mean_auroc_inflation_ci95"] = list(ci_result.ci)
    summary["mean_auroc_inflation_ci_n_clusters"] = ci_result.n_clusters
    # Cross-check against whatever mean was already saved, if present — should
    # match ci_result.mean closely; a real mismatch would indicate the saved
    # summary and results.json have drifted apart (e.g. edited by hand,
    # regenerated separately), worth knowing about rather than silently
    # overwriting.
    existing_mean = summary.get("mean_auroc_inflation")
    if existing_mean is not None and abs(existing_mean - ci_result.mean) > 1e-6:
        logger.warning(
            "%s: existing mean_auroc_inflation=%.6f does not match recomputed "
            "mean=%.6f from go_probing_results.json — files may have drifted "
            "apart; recomputed mean NOT written over the existing value, only "
            "the new CI fields were added.",
            output_dir, existing_mean, ci_result.mean,
        )
    else:
        summary["mean_auroc_inflation"] = ci_result.mean

    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "%s: %s  (n_rows=%d)", output_dir, str(ci_result), len(inflation_rows)
    )
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="Single eval output dir OR a sweep dir "
                                                   "containing mp<N>/ subdirectories")
    args = p.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    # Sweep layout: mp<N>/go_probing_results.json subdirectories.
    mp_subdirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("mp"))
    valid_mp_subdirs = [d for d in mp_subdirs if (d / "go_probing_results.json").exists()]

    if valid_mp_subdirs:
        n_done = sum(backfill_one_dir(d) for d in valid_mp_subdirs)
        logger.info("Backfilled %d/%d sweep-point directories under %s",
                    n_done, len(valid_mp_subdirs), root)
    elif (root / "go_probing_results.json").exists():
        backfill_one_dir(root)
    else:
        raise SystemExit(
            f"No go_probing_results.json found directly in {root} or in any "
            f"mp<N> subdirectory — is this the right path?"
        )


if __name__ == "__main__":
    main()
