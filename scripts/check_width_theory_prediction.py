"""
Does the winner's-curse lemma predict a publishable dictionary-width effect,
and if so, how much data would a sweep need to detect it?

The lemma's own scaling (already in the codebase: `gaussian_max_inflation_
prediction`, `E[max_i eps_i] ~ sigma * sqrt(2 ln d)`) makes the width-vs-
inflation prediction fully computable WITHOUT running the width sweep at
all -- sigma cancels in the ratio between two widths, so the predicted
FRACTIONAL change in inflation depends only on d1 and d2, not on the
sample-size regime. This script makes that derivation explicit and
reproducible (independently verified before being adopted) rather than
leaving it as an unauditable one-off calculation in prose.

It also answers the power question directly: given the existing pilot's own
per-cluster inflation variance (from docs/experiment_auroc_inflation_vs_n.json,
esm2_8m mp=10), what power does the pilot's actual cluster counts have to
detect the theory-predicted effect, and how many clusters per arm would a
properly-powered version need?

No GPU, no new data -- pure derivation plus a re-analysis of numbers already
on disk.

Usage:
  python scripts/check_width_theory_prediction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scipy import stats  # noqa: E402

from biolens.eval.statistics import gaussian_max_inflation_prediction  # noqa: E402

_REPO_ROOT = Path(__file__).parent.parent


def predicted_inflation_ratio(d1: float, d2: float) -> float:
    """Predicted E[max inflation at d2] / E[max inflation at d1], per the
    lemma's scaling. sigma is arbitrary and cancels exactly in the ratio
    (passed as 1.0 below), so this doesn't depend on sample size /
    AUROC-variance regime at all.

    Must use folded=True: `_vectorized_auroc` always reflects before the
    max-over-d step (see gaussian_max_inflation_prediction's docstring), so
    the width-ratio prediction should describe what the pipeline actually
    computes. A real bug (external codebase audit, 2026-08-27, caught AFTER
    the sigma-rescaling fix landed): this used to call folded=False here,
    on the reasoning that "the fold factor is a constant multiplicative
    factor on sigma, so it cancels in the ratio regardless." That reasoning
    was correct under the SUPERSEDED formula (sigma rescaling truly is a
    constant multiplier, cancels in any ratio) but became silently wrong
    once folded=True started changing the EFFECTIVE CANDIDATE COUNT
    (n=2*d) inside the nonlinear sqrt(log) asymptotic instead -- that does
    NOT cancel in a ratio between two different d values. Confirmed the
    two give different numbers (folded=False systematically overstates
    every width-ratio percentage relative to folded=True, by a few tenths
    of a point at the widths this project actually considers).
    """
    return gaussian_max_inflation_prediction(1.0, d2, folded=True) / gaussian_max_inflation_prediction(1.0, d1, folded=True)


def required_clusters_per_arm(effect: float, per_cluster_sd: float, power_target: float, alpha: float = 0.05) -> float:
    """Clusters needed per arm (two-arm comparison, equal n) to detect
    `effect` at `power_target`, given each arm's mean-inflation SE scales as
    per_cluster_sd / sqrt(n) (the standard CLT approximation this project's
    own bootstrap SEs are consistent with -- see the reference-SE check
    below)."""
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power_target)
    se_diff_needed = effect / (z_crit + z_beta)
    return 2 * (per_cluster_sd / se_diff_needed) ** 2


def power_at_n(effect: float, per_cluster_sd: float, n_per_arm: float, alpha: float = 0.05) -> float:
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se_diff = per_cluster_sd * (2 / n_per_arm) ** 0.5
    z_effect = effect / se_diff
    return (1 - stats.norm.cdf(z_crit - z_effect)) + stats.norm.cdf(-z_crit - z_effect)


def main() -> None:
    print("Predicted inflation ratio from the lemma's sqrt(2 ln d) scaling (no data needed):")
    contrasts = [
        ("4x -> 16x (ESM2-8M pilot: 1280 -> 5120)", 1280, 5120),
        ("8x -> 16x", 2560, 5120),
        ("Gemma Scope 16k -> 1M (64x)", 16_000, 16_000 * 64),
        ("doubling from 16k", 16_000, 32_000),
        ("1000x from 16k", 16_000, 16_000_000),
    ]
    for label, d1, d2 in contrasts:
        ratio = predicted_inflation_ratio(d1, d2)
        print(f"  {label}: ratio={ratio:.4f} ({(ratio-1)*100:+.1f}%)")

    # d needed from d0=1280 for a 50% increase
    from scipy.optimize import brentq
    d0 = 1280
    target = 1.5
    d_needed = brentq(lambda d: predicted_inflation_ratio(d0, d) - target, d0, 1e15)
    print(f"\n  d needed from d0={d0} for +50% inflation: {d_needed:.3e} ({d_needed/d0:.0f}x wider)")

    # ── Power check against the reference precision from the real 100K sweep ──
    inflation_path = _REPO_ROOT / "docs/experiment_auroc_inflation_vs_n.json"
    ref = json.loads(inflation_path.read_text())["per_model_points"]["esm2_8m"][0]
    assert ref["min_positives"] == 10
    ref_se = (ref["ci95"][1] - ref["ci95"][0]) / 2 / 1.96
    per_cluster_sd = ref_se * ref["n_clusters"] ** 0.5
    print(f"\nReference (esm2_8m, mp=10): inflation={ref['mean_auroc_inflation']:.4f}, "
          f"SE={ref_se:.4f} at n_clusters={ref['n_clusters']} -> per-cluster SD={per_cluster_sd:.4f}")

    baseline_inflation = 0.1815  # k32_x4 pilot, mp=10 -- the actual comparison baseline
    ratio_4to16 = predicted_inflation_ratio(1280, 5120)
    effect = (ratio_4to16 - 1) * baseline_inflation
    print(f"\nTheory-predicted absolute effect for 4x->16x at baseline {baseline_inflation}: {effect:.4f}")

    for n in (20, 40, 81):
        p = power_at_n(effect, per_cluster_sd, n)
        print(f"  power at n_per_arm={n}: {p:.3f}")

    n80 = required_clusters_per_arm(effect, per_cluster_sd, 0.80)
    n90 = required_clusters_per_arm(effect, per_cluster_sd, 0.90)
    print(f"\nClusters/arm needed for 80% power: {n80:.0f}   for 90% power: {n90:.0f}")


if __name__ == "__main__":
    main()
