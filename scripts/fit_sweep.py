"""
Fit the Experiment 3 confirmatory sweep model:
`verified_rate ~ log(min_positives)`, clustered on feature_idx.

Joins per-sweep-point go_probing_results.json files (produced by
scripts/run_eval.py --held-out-baselines, one per min_positives value) against
a verified-features registry (configs/verified_features/*.yaml), and fits the
pre-specified logistic regression with cluster-robust and cluster-bootstrap
confidence intervals (biolens.eval.statistics.fit_verified_rate_sweep).

Only rows where the SPECIFIC (go_id, feature_idx) claim was actually checked
against the registry are used — this deliberately does not extrapolate to
unverified top hits.

Expects sweep results laid out as:
  <sweep-dir>/mp<N>/go_probing_results.json    for each min_positives=N

Usage:
  python scripts/fit_sweep.py \\
      --sweep-dir /path/to/sweep_results \\
      --verified-features configs/verified_features/esm2_8m_layer5_topk_k32.yaml \\
      --output docs/experiment3_sweep_fit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.eval.feature_inspection import load_verified_annotations
from biolens.eval.statistics import build_sweep_rows, fit_verified_rate_sweep

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-dir", required=True,
        help="Directory containing mp<N>/go_probing_results.json subdirectories",
    )
    p.add_argument("--verified-features", required=True, help="Path to verified_features/*.yaml")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output", default=None, help="Optional path to save the fit result as JSON")
    return p.parse_args()


def load_sweep_results(sweep_dir: Path) -> dict[int, list[dict]]:
    """Load every mp<N>/go_probing_results.json under sweep_dir into
    {min_positives: [rows]}. Directories that don't match the mp<N> naming
    convention or don't contain results are skipped with a warning, not a
    crash — a partially-run sweep should still fit on what's available."""
    sweep_results: dict[int, list[dict]] = {}
    for subdir in sorted(sweep_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("mp"):
            continue
        try:
            mp = int(subdir.name[2:])
        except ValueError:
            logger.warning("Skipping directory with unparseable name: %s", subdir.name)
            continue

        results_path = subdir / "go_probing_results.json"
        if not results_path.exists():
            logger.warning("No go_probing_results.json in %s — skipping", subdir)
            continue

        with open(results_path) as f:
            sweep_results[mp] = json.load(f)
        logger.info("Loaded %d rows for min_positives=%d from %s", len(sweep_results[mp]), mp, subdir)

    return sweep_results


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)

    sweep_results = load_sweep_results(sweep_dir)
    if not sweep_results:
        raise SystemExit(
            f"No sweep results found under {sweep_dir} "
            f"(expected mp<N>/go_probing_results.json subdirectories)"
        )

    verified = load_verified_annotations(args.verified_features)
    logger.info("Loaded %d verified-feature registry entries", len(verified))

    min_positives, target, feature_idx = build_sweep_rows(sweep_results, verified)
    if len(min_positives) == 0:
        raise SystemExit(
            "No sweep rows matched the verified-features registry — nothing to fit. "
            "This means none of the top-hit (go_id, feature_idx) pairs across the "
            "sweep have been checked against the registry yet."
        )
    logger.info(
        "Built %d confirmatory rows (%d distinct features) from %d sweep points",
        len(min_positives), len(set(feature_idx.tolist())), len(sweep_results),
    )

    result = fit_verified_rate_sweep(
        min_positives, target, feature_idx,
        n_bootstrap=args.n_bootstrap, confidence=args.confidence, random_state=args.random_state,
    )
    print("\n" + str(result))

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "intercept": result.intercept,
                    "slope": result.slope,
                    "intercept_se_cluster": result.intercept_se_cluster,
                    "slope_se_cluster": result.slope_se_cluster,
                    "slope_ci_cluster": list(result.slope_ci_cluster),
                    "slope_p_value_cluster": result.slope_p_value_cluster,
                    "slope_ci_bootstrap": (
                        list(result.slope_ci_bootstrap) if result.slope_ci_bootstrap else None
                    ),
                    "n_bootstrap_successful": result.n_bootstrap_successful,
                    "n_rows": result.n_rows,
                    "n_clusters": result.n_clusters,
                    "effect_per_doubling": result.effect_per_doubling(),
                },
                indent=2,
            )
        )
        logger.info("Saved fit result to %s", args.output)


if __name__ == "__main__":
    main()
