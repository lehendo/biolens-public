"""
Cluster-bootstrap CI for the hard-negative confounding gap:
mean(held_out_auroc - hard_negative_held_out_auroc), per model per
min_positives sweep point.

This is the actual confounding-quantification tool: held_out_auroc
already removes winner's-curse selection-bias inflation (mechanism 1), so a
persistently positive gap against hard negatives is mechanism 2 (structural
confounding) surviving that correction. Reuses
bootstrap_mean_auroc_inflation unchanged — a per-row gap
(held_out - hard_negative_held_out) is just another per-row scalar to
cluster-bootstrap on best_feature_idx, mathematically identical to what that
function already does for auroc_inflation.

Usage:
  python scripts/fit_hard_negative_gap.py \\
      --sweep-dir esm2_8m=/scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k32_x8 \\
      --sweep-dir esm2_35m=/scratch/arjunc4/biolens/eval/sweep_esm2_35m_L5_k32_x8 \\
      --sweep-dir esm2_150m=/scratch/arjunc4/biolens/eval/sweep_esm2_150m_L14_k32_x8 \\
      --sweep-dir esm2_650m=/scratch/arjunc4/biolens/eval/sweep_esm2_650m_L16_k32_x8 \\
      --output docs/experiment_hard_negative_gap.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.eval.statistics import bootstrap_mean_auroc_inflation

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _parse_labeled_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected label=path (e.g. esm2_8m=/path/to/sweep), got: {spec!r}"
        )
    label, _, path = spec.partition("=")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-dir", action="append", required=True, type=_parse_labeled_path,
        metavar="LABEL=PATH",
        help="model_scale=sweep_dir, repeatable. sweep_dir contains mp<N>/go_probing_results.json.",
    )
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output", default=None, help="Optional path to save results as JSON")
    return p.parse_args()


def load_sweep_results(sweep_dir: Path) -> dict[int, list[dict]]:
    sweep_results: dict[int, list[dict]] = {}
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
            sweep_results[mp] = json.load(f)
    return sweep_results


def main() -> None:
    args = parse_args()

    output: dict[str, dict] = {}

    for label, sweep_dir in args.sweep_dir:
        logger.info("Loading sweep for scale=%s from %s", label, sweep_dir)
        sweep_results = load_sweep_results(sweep_dir)
        if not sweep_results:
            raise SystemExit(f"No sweep results found under {sweep_dir} for scale={label}")

        output[label] = {}
        print(f"\n=== {label} ===")
        print(f"{'mp':>5} {'n_paired':>9} {'n_clusters':>10} {'mean_gap':>9} {'95% CI':>20}")
        for mp in sorted(sweep_results):
            rows = sweep_results[mp]
            paired = [
                (r["held_out_auroc"], r["hard_negative_held_out_auroc"], r["best_feature_idx"])
                for r in rows
                if r.get("held_out_auroc") is not None
                and r.get("hard_negative_held_out_auroc") is not None
            ]
            if not paired:
                logger.warning("No paired rows at %s mp=%d — skipping", label, mp)
                continue

            gap = np.array([p[0] - p[1] for p in paired], dtype=float)
            feature_idx = np.array([p[2] for p in paired])

            result = bootstrap_mean_auroc_inflation(
                gap, feature_idx,
                n_bootstrap=args.n_bootstrap, confidence=args.confidence,
                random_state=args.random_state,
            )
            lo, hi = result.ci
            print(f"{mp:>5} {result.n_rows:>9} {result.n_clusters:>10} {result.mean:>9.4f} [{lo:.4f}, {hi:.4f}]")

            output[label][str(mp)] = {
                "n_paired": result.n_rows,
                "n_clusters": result.n_clusters,
                "mean_gap": result.mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_bootstrap_successful": result.n_bootstrap_successful,
            }

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))
        logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
