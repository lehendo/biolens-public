"""
Power calculation for the confirmatory sweep model: justifies the 50-100
verified-feature ground-truth registry target via simulation, on EFFECTIVE
CLUSTER COUNT — not naive row count — since the confirmatory model
(biolens.eval.statistics.fit_verified_rate_sweep) clusters on `feature_idx`:
one SAE feature can be the top hit for multiple GO terms/sweep points, so a
naive N-based power calculation would overstate how well-powered the fit
actually is.

Method: Monte Carlo simulation using the REAL, production `fit_verified_rate_sweep`
code path (not a separate closed-form power formula, which doesn't have a
simple textbook form for a clustered-GLM-plus-bootstrap design anyway) — for
each candidate registry size (n_clusters), repeatedly simulate synthetic
sweep data under an assumed true effect size, fit the real model, and measure
the fraction of simulations where the 95% cluster-robust CI excludes zero.
That fraction IS the statistical power at that registry size.

Cluster-size assumption (rows per feature): calibrated from the real,
already-verified registry (configs/verified_features/esm2_8m_layer5_topk_k32.yaml,
29 entries as of 2026-07-14): mean 1.38 claimed_go terms per feature entry
(mostly 1, up to 5). Each claimed (feature, GO-term) pair can appear as a row
at multiple min_positives sweep points if that feature remains the top hit
across thresholds — not guaranteed at every point, so --rows-per-cluster
defaults to a small integer informed by, but not identical to, that raw
claimed_go count; swept across a range rather than asserting one number, per
this project's own "state assumptions explicitly" standard.

Usage:
  python scripts/power_calculation.py \\
      --n-clusters-values 20 30 50 75 100 150 \\
      --effect-sizes 0.1 0.2 0.3 \\
      --n-sims 300 --output docs/power_calculation_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.eval.statistics import fit_verified_rate_sweep  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# Matches configs/eval.yaml::probing.sweep_min_positives — the real sweep points.
SWEEP_POINTS = (10, 20, 50, 100, 150, 200, 300)


def simulate_one_dataset(
    n_clusters: int,
    slope: float,
    rows_per_cluster: int,
    intercept: float,
    cluster_sd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate one synthetic confirmatory-sweep dataset under an assumed true
    logistic relationship verified_rate ~ log(min_positives), with a
    cluster-level random effect (intra-cluster correlation — the same
    structure the real registry has: a feature's own "how verifiable is
    this feature in general" quality is shared across every GO term it's
    the top hit for, not independent per row).

    Args:
        n_clusters: Number of distinct simulated features (registry size).
        slope: True population-level slope on log(min_positives).
        rows_per_cluster: Rows (GO-term/sweep-point hits) per feature.
        intercept: True population-level intercept.
        cluster_sd: SD of each cluster's random intercept offset — controls
            intra-cluster correlation strength; 0 = no clustering effect
            (rows within a feature are conditionally independent given
            min_positives), larger = stronger within-feature correlation.
        rng: NumPy Generator.

    Returns:
        (min_positives, verified, feature_idx) arrays, ready for
        fit_verified_rate_sweep — same shape/semantics as build_sweep_rows'
        real output.
    """
    cluster_offsets = rng.normal(0, cluster_sd, size=n_clusters)

    min_positives = rng.choice(SWEEP_POINTS, size=n_clusters * rows_per_cluster)
    feature_idx = np.repeat(np.arange(n_clusters), rows_per_cluster)
    offsets = np.repeat(cluster_offsets, rows_per_cluster)

    eta = intercept + offsets + slope * np.log(min_positives)
    p = 1.0 / (1.0 + np.exp(-eta))
    verified = (rng.random(len(p)) < p).astype(float)

    return min_positives.astype(float), verified, feature_idx


def compute_power(
    n_clusters: int,
    slope: float,
    rows_per_cluster: int,
    n_sims: int,
    rng: np.random.Generator,
    intercept: float = -1.0,
    cluster_sd: float = 0.5,
) -> tuple[float, int]:
    """
    Empirical power at a given registry size and true effect size: fraction
    of n_sims simulated datasets where the real fit_verified_rate_sweep's
    95% cluster-robust CI excludes zero (correctly detects the true effect).

    Returns:
        (power, n_fit_failures) — n_fit_failures counts simulations where the
        GLM failed to converge or a degenerate resample occurred (e.g. all-0
        or all-1 outcomes), excluded from the power denominator rather than
        silently counted as either a detection or a miss.
    """
    detections = 0
    n_valid = 0
    n_failures = 0

    for _ in range(n_sims):
        mp, verified, feat = simulate_one_dataset(
            n_clusters, slope, rows_per_cluster, intercept, cluster_sd, rng
        )
        if verified.min() == verified.max():
            n_failures += 1  # degenerate — no outcome variance, GLM undefined
            continue
        try:
            # n_bootstrap=0-equivalent: use the analytic cluster-robust CI
            # only (fast enough to skip the bootstrap for a power sweep that
            # needs thousands of fits; the real confirmatory analysis still
            # reports both, this just isn't re-deriving the bootstrap's own
            # validity, only using the model's primary analytic inference).
            result = fit_verified_rate_sweep(mp, verified, feat, n_bootstrap=0, random_state=0)
        except Exception as exc:  # noqa: BLE001 — a failed fit must not abort the whole power sweep
            logger.debug("Fit failed during power simulation: %s", exc)
            n_failures += 1
            continue
        lo, hi = result.slope_ci_cluster
        if not (np.isnan(lo) or np.isnan(hi)) and (lo > 0 or hi < 0):
            detections += 1
        n_valid += 1

    power = detections / n_valid if n_valid > 0 else float("nan")
    return power, n_failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-clusters-values", type=int, nargs="+",
        default=[20, 30, 50, 75, 100, 150],
        help="Candidate registry sizes (distinct verified features) to compute power at",
    )
    p.add_argument(
        "--effect-sizes", type=float, nargs="+", default=[0.1, 0.2, 0.3],
        help="Candidate true slopes on log(min_positives) — see module docstring; "
             "swept across a range rather than asserting one 'true' effect size",
    )
    p.add_argument(
        "--rows-per-cluster", type=int, default=3,
        help="Rows (GO-term-hit x sweep-point) per simulated feature — calibrated "
             "from the real registry's mean 1.38 claimed_go per entry, adjusted up "
             "for partial persistence across sweep points (see module docstring)",
    )
    p.add_argument("--n-sims", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    results = []

    for slope in args.effect_sizes:
        for n_clusters in args.n_clusters_values:
            t0 = time.time()
            power, n_failures = compute_power(
                n_clusters, slope, args.rows_per_cluster, args.n_sims, rng
            )
            results.append({
                "n_clusters": n_clusters,
                "rows_per_cluster": args.rows_per_cluster,
                "n_rows": n_clusters * args.rows_per_cluster,
                "true_slope": slope,
                "power": power,
                "n_sims": args.n_sims,
                "n_failures": n_failures,
            })
            logger.info(
                "slope=%.2f  n_clusters=%4d  power=%.3f  (%d/%d sims valid)  [%.1fs]",
                slope, n_clusters, power, args.n_sims - n_failures, args.n_sims,
                time.time() - t0,
            )

    print("\n" + "=" * 78)
    print("POWER CALCULATION — verified_rate ~ log(min_positives), clustered on feature_idx")
    print(f"(rows_per_cluster={args.rows_per_cluster}, n_sims={args.n_sims} per cell)")
    print("=" * 78)
    for slope in args.effect_sizes:
        print(f"\ntrue slope={slope}:")
        print(f"  {'n_clusters':>10}  {'n_rows':>8}  {'power':>7}")
        for r in results:
            if r["true_slope"] == slope:
                print(f"  {r['n_clusters']:>10}  {r['n_rows']:>8}  {r['power']:>7.3f}")
    print("=" * 78)

    for slope in args.effect_sizes:
        slope_results = [r for r in results if r["true_slope"] == slope]
        adequate = [r for r in slope_results if r["power"] >= 0.8]
        if adequate:
            min_n = min(r["n_clusters"] for r in adequate)
            print(f"Minimum n_clusters for >=80% power at slope={slope}: {min_n}")
        else:
            print(f"No candidate n_clusters value reached 80% power at slope={slope} "
                  f"(max tried: {max(args.n_clusters_values)})")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
