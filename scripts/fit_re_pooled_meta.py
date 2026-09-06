"""
Random-effects meta-analysis of the four ESM2 scales' independent AUROC-
inflation-vs-n and permuted-inflation-vs-n slopes (the two real-data
confirmations of the winner's-curse simulation's `n^(-1/2)`-ish scaling
lemma).

Replaces the naive "pool every model's raw rows into one regression"
approach already in fit_auroc_inflation_curve.py/fit_permuted_inflation_
curve.py's `pooled_fit` field, which implicitly weights each model by its
row count rather than by the precision of its own independent slope
estimate -- a real problem: the real-label pooled slope (-0.722) is the
only estimator among every reasonable pooling choice that EXCLUDES the
theoretical -0.5, purely because of unequal row counts across models, not
because the effect disagrees with theory. See RandomEffectsMetaResult's
docstring in src/biolens/eval/statistics.py for the full argument.

Each model's own per-scale fit already reports a 95% CI from an OLS
log-log regression on n_points=6 points (t-distributed, df=n_points-2=4)
-- this script back-derives each model's SE from that CI (rather than
re-fitting), then pools via DerSimonian-Laird.

Usage:
  python scripts/fit_re_pooled_meta.py \\
      --route1 docs/experiment_auroc_inflation_vs_n.json \\
      --route2 docs/experiment_permuted_inflation_vs_n.json \\
      --output docs/experiment_re_pooled_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scipy import stats as scipy_stats

from biolens.eval.statistics import fit_random_effects_meta


def _se_from_ci(ci: tuple[float, float], df: int) -> float:
    """Back out a study's own SE from its reported 95% CI half-width,
    inverting the same t(0.975, df) critical value the per-model OLS fit
    used to build that CI in the first place."""
    half_width = (ci[1] - ci[0]) / 2.0
    return half_width / scipy_stats.t.ppf(0.975, df)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route1", default="docs/experiment_auroc_inflation_vs_n.json",
                    help="fit_auroc_inflation_curve.py output (real-label AUROC inflation)")
    p.add_argument("--route2", default="docs/experiment_permuted_inflation_vs_n.json",
                    help="fit_permuted_inflation_curve.py output (permuted-label inflation)")
    p.add_argument("--output", default=None, help="Optional path to save the result as JSON")
    return p.parse_args()


def pool_route(per_model_fits: dict, label: str) -> dict:
    models = sorted(per_model_fits.keys())
    thetas = [per_model_fits[m]["slope"] for m in models]
    # each per-model fit's own n_points fixes its own OLS df = n_points - 2
    ses = [
        _se_from_ci(tuple(per_model_fits[m]["slope_ci95"]), per_model_fits[m]["n_points"] - 2)
        for m in models
    ]
    result = fit_random_effects_meta(thetas, ses)
    print(f"\n{label}:")
    print(f"  models (input order): {models}")
    print(f"  per-model slopes: {[round(t, 4) for t in thetas]}")
    print(f"  per-model SEs (back-derived from CI): {[round(s, 4) for s in ses]}")
    print(f"  {result}")
    return {
        "models": models,
        "per_model_slope": dict(zip(models, thetas)),
        "per_model_se": dict(zip(models, ses)),
        "theta_re": result.theta_re,
        "se_re": result.se_re,
        "ci": list(result.ci),
        "se_kh": result.se_kh,
        "ci_kh": list(result.ci_kh),  # Knapp-Hartung -- report this one, see statistics.py docstring
        "theta_fixed_effect": result.theta_fixed_effect,
        "tau_squared": result.tau_squared,
        "q_statistic": result.q_statistic,
        "q_p_value": result.q_p_value,
        "i_squared": result.i_squared,
        "k": result.k,
        "row_level_naive_pooled_slope": None,  # filled in by caller from the source file
        "row_level_naive_pooled_ci": None,
    }


def main() -> None:
    args = parse_args()

    route1_data = json.loads(Path(args.route1).read_text())
    route2_data = json.loads(Path(args.route2).read_text())

    out = {
        "route1_auroc_real_label": pool_route(route1_data["per_model_fits"], "Route 1 (AUROC, real-label)"),
        "route2_permuted_label": pool_route(route2_data["per_model_fits"], "Route 2 (permuted-label)"),
    }
    out["route1_auroc_real_label"]["row_level_naive_pooled_slope"] = route1_data["pooled_fit"]["slope"]
    out["route1_auroc_real_label"]["row_level_naive_pooled_ci"] = route1_data["pooled_fit"]["slope_ci95"]
    out["route2_permuted_label"]["row_level_naive_pooled_slope"] = route2_data["pooled_fit"]["slope"]
    out["route2_permuted_label"]["row_level_naive_pooled_ci"] = route2_data["pooled_fit"]["slope_ci95"]

    print("\nBoth routes' random-effects pooled CI vs. the theoretical -0.5 "
          "(Knapp-Hartung -- report this one; normal-quantile shown for reference):")
    for key in ("route1_auroc_real_label", "route2_permuted_label"):
        lo_kh, hi_kh = out[key]["ci_kh"]
        lo_z, hi_z = out[key]["ci"]
        covers_kh = lo_kh <= -0.5 <= hi_kh
        print(f"  {key}: KH=[{lo_kh:.4f}, {hi_kh:.4f}] (covers -0.5: {covers_kh})"
              f"  normal=[{lo_z:.4f}, {hi_z:.4f}]")

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
