"""
Does the same sample-size-dependent pattern show up in multivariate_auroc,
not just single_feature_auroc?

multivariate_auroc is already a held-out estimate by construction (fit on an
80/20 StratifiedShuffleSplit's train fold, AUROC computed on the test fold —
see probe_go_terms in biolens/eval/probing.py), unlike single_feature_auroc's
original naive (in-sample) computation. So this isn't the same "naive vs.
held-out" comparison Experiment 3 already makes for single_feature_auroc —
it's a different, related question: does a logistic regression over the
full d_sae-dimensional feature set still show an overfitting-driven pattern
(few positives relative to d_sae -> unstable/poor held-out generalization)
that improves as min_positives grows, even though the estimate is already
out-of-sample?

Pure data analysis — reads already-complete go_probing_results.json files
from the original 10K sweeps, no new computation or GPU needed.

Usage:
  python scripts/check_multivariate_auroc_pattern.py \\
      --sweep-dir /scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k32_x8 \\
      --label esm2_8m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", required=True, help="Directory containing mp<N>/go_probing_results.json")
    p.add_argument("--label", default=None, help="Optional label for the printed table (e.g. model name)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    label = args.label or sweep_dir.name

    print(f"\n=== {label} ({sweep_dir}) ===")
    print(f"{'mp':>6} {'n_terms':>8} {'mean_single_auroc':>18} {'mean_multi_auroc':>17} {'mean_held_out':>14}")

    for subdir in sorted(sweep_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("mp"):
            continue
        try:
            mp = int(subdir.name[2:])
        except ValueError:
            continue

        results_path = subdir / "go_probing_results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            rows = json.load(f)
        if not rows:
            continue

        single = np.array([r["single_feature_auroc"] for r in rows if r["single_feature_auroc"] is not None])
        multi = np.array([r["multivariate_auroc"] for r in rows if r["multivariate_auroc"] is not None])
        held_out = [r.get("held_out_auroc") for r in rows if r.get("held_out_auroc") is not None]
        held_out_arr = np.array(held_out) if held_out else None

        mean_single = f"{single.mean():.4f}" if len(single) else "n/a"
        mean_multi = f"{multi.mean():.4f}" if len(multi) else "n/a"
        mean_held_out = f"{held_out_arr.mean():.4f}" if held_out_arr is not None and len(held_out_arr) else "n/a"

        print(f"{mp:>6} {len(rows):>8} {mean_single:>18} {mean_multi:>17} {mean_held_out:>14}")


if __name__ == "__main__":
    main()
