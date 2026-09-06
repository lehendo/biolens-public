"""
Real-data AUROC-inflation-vs-n curve.

The theory section's lemma predicts AUROC-inflation MAGNITUDE decays
approximately as `n^(-1/2)` (a power law, via the Hanley-McNeil AUROC-
variance relationship) — a different quantity, and a different functional
form, than the headline `verified_rate ~ log(min_positives)` fit (a bounded
dose-response outcome, atheoretical choice of form). This script computes
and fits the quantity the lemma actually predicts: `(naive max AUROC) -
(held-out AUROC)`, i.e. `auroc_inflation`, already saved per-row in every
go_probing_results.json produced with --held-out-baselines, at every
min_positives sweep point, for all four ESM2 scales' 100K sweeps (the
higher-powered, current sweeps).

No new GPU compute needed — this is a pure re-analysis of already-completed
sweep output sitting on disk (confirmed real and populated, 2026-07-30 —
correcting a stale doc note that this "needs a resubmitted sweep").

Usage:
  python scripts/fit_auroc_inflation_curve.py \\
      --sweep-dir sweep_esm2_8m_L5_k32_x8_100k=/scratch/.../sweep_esm2_8m_L5_k32_x8_100k \\
      --sweep-dir sweep_esm2_35m_L5_k32_x8_100k=/scratch/.../sweep_esm2_35m_L5_k32_x8_100k \\
      --sweep-dir sweep_esm2_150m_L14_k32_x8_100k=/scratch/.../sweep_esm2_150m_L14_k32_x8_100k \\
      --sweep-dir sweep_esm2_650m_L16_k32_x8_100k=/scratch/.../sweep_esm2_650m_L16_k32_x8_100k \\
      --output docs/experiment_auroc_inflation_vs_n.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from biolens.eval.statistics import bootstrap_mean_auroc_inflation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_MIN_POSITIVES_POINTS = [10, 20, 50, 100, 150, 200, 300]
_THEORETICAL_SLOPE = -0.5  # lemma's n^(-1/2)-ish prediction, see module docstring


def _load_inflation_rows(
    sweep_dir: Path, min_positives: int, exclude_go_ids: frozenset[str] = frozenset()
) -> tuple[np.ndarray, np.ndarray]:
    path = sweep_dir / f"mp{min_positives}" / "go_probing_results.json"
    with open(path) as f:
        rows = json.load(f)
    rows = [r for r in rows if r.get("go_id") not in exclude_go_ids]
    inflation = np.array(
        [r["auroc_inflation"] for r in rows if r.get("auroc_inflation") is not None]
    )
    feat_idx = np.array(
        [r["best_feature_idx"] for r in rows if r.get("auroc_inflation") is not None]
    )
    return inflation, feat_idx


def _row_identity_key(sweep_dir: Path, min_positives: int) -> frozenset:
    """(feature_idx, go_id, auroc_inflation) triples for a sweep point —
    used to detect when two min_positives points' held-out-AUROC-eligible
    GO-term sets are literally identical (a real, confirmed artifact: a
    fixed max_go_terms cap can bind before min_positives does at the low
    end of a sweep, e.g. mp10 and mp20 both landing on the exact same
    top-500-by-positive-count term set — confirmed 2026-07-30 via direct
    row-level diff, not assumed from matching summary stats alone)."""
    path = sweep_dir / f"mp{min_positives}" / "go_probing_results.json"
    with open(path) as f:
        rows = json.load(f)
    return frozenset(
        (r["best_feature_idx"], r["go_id"], r["auroc_inflation"])
        for r in rows
        if r.get("auroc_inflation") is not None
    )


def fit_inflation_curve(
    sweep_dirs: dict[str, Path], exclude_go_ids: frozenset[str] = frozenset()
) -> dict:
    per_model: dict[str, list[dict]] = {}
    all_log_mp: list[float] = []
    all_log_inflation: list[float] = []
    all_model_labels: list[str] = []

    for model_name, sweep_dir in sweep_dirs.items():
        points = []
        seen_row_keys: dict[frozenset, int] = {}
        for mp in _MIN_POSITIVES_POINTS:
            results_path = sweep_dir / f"mp{mp}" / "go_probing_results.json"
            if not results_path.exists():
                logger.warning("%s: missing %s — skipping mp=%d", model_name, results_path, mp)
                continue

            row_key = _row_identity_key(sweep_dir, mp)
            duplicate_of = seen_row_keys.get(row_key)
            seen_row_keys.setdefault(row_key, mp)

            inflation, feat_idx = _load_inflation_rows(sweep_dir, mp, exclude_go_ids)
            if len(inflation) == 0:
                logger.warning("%s mp=%d: no rows with auroc_inflation — skipping", model_name, mp)
                continue

            ci_result = bootstrap_mean_auroc_inflation(inflation, feat_idx)
            point = {
                "min_positives": mp,
                "mean_auroc_inflation": ci_result.mean,
                "ci95": list(ci_result.ci),
                "n_rows": ci_result.n_rows,
                "n_clusters": ci_result.n_clusters,
                "duplicate_of_min_positives": duplicate_of,
            }
            points.append(point)
            logger.info(
                "%s mp=%-4d  mean_inflation=%.4f  ci95=[%.4f, %.4f]  n_clusters=%d%s",
                model_name, mp, ci_result.mean, *ci_result.ci, ci_result.n_clusters,
                f"  [DUPLICATE of mp={duplicate_of}, excluded from fit]" if duplicate_of else "",
            )

            # Only the FIRST point in an identical-row-set group contributes
            # to the log-log fit — a duplicate is the same measurement
            # again, not an independent data point (would silently
            # double-weight it and understate the fit's true uncertainty).
            if duplicate_of is None and ci_result.mean > 0:
                all_log_mp.append(float(np.log(mp)))
                all_log_inflation.append(float(np.log(ci_result.mean)))
                all_model_labels.append(model_name)

        per_model[model_name] = points

    # ── Per-model log-log OLS fits ────────────────────────────────────────────
    fits: dict[str, dict] = {}
    for model_name in sweep_dirs:
        model_log_mp = [lm for lm, lbl in zip(all_log_mp, all_model_labels) if lbl == model_name]
        model_log_inf = [
            li for li, lbl in zip(all_log_inflation, all_model_labels) if lbl == model_name
        ]
        if len(model_log_mp) < 3:
            fits[model_name] = {"error": f"only {len(model_log_mp)} usable points, need >=3"}
            continue
        X = sm.add_constant(np.array(model_log_mp))
        ols = sm.OLS(np.array(model_log_inf), X).fit()
        slope = float(ols.params[1])
        ci_lo, ci_hi = ols.conf_int(alpha=0.05)[1]
        fits[model_name] = {
            "slope": slope,
            "slope_ci95": [float(ci_lo), float(ci_hi)],
            "intercept": float(ols.params[0]),
            "r_squared": float(ols.rsquared),
            "n_points": len(model_log_mp),
            "p_value": float(ols.pvalues[1]),
            "consistent_with_theoretical_slope": bool(ci_lo <= _THEORETICAL_SLOPE <= ci_hi),
        }

    # ── Pooled fit (all models' points together, no model fixed effect —
    # a simple, honest check of the aggregate shape, not a claim about
    # between-model differences, which the per-model fits above already
    # cover individually) ──────────────────────────────────────────────────
    X_pooled = sm.add_constant(np.array(all_log_mp))
    ols_pooled = sm.OLS(np.array(all_log_inflation), X_pooled).fit()
    slope_pooled = float(ols_pooled.params[1])
    ci_lo_p, ci_hi_p = ols_pooled.conf_int(alpha=0.05)[1]
    pooled_fit = {
        "slope": slope_pooled,
        "slope_ci95": [float(ci_lo_p), float(ci_hi_p)],
        "intercept": float(ols_pooled.params[0]),
        "r_squared": float(ols_pooled.rsquared),
        "n_points": len(all_log_mp),
        "p_value": float(ols_pooled.pvalues[1]),
        "consistent_with_theoretical_slope": bool(ci_lo_p <= _THEORETICAL_SLOPE <= ci_hi_p),
    }

    return {
        "theoretical_slope": _THEORETICAL_SLOPE,
        "per_model_points": per_model,
        "per_model_fits": fits,
        "pooled_fit": pooled_fit,
        "note": (
            "Real-label curve — subject to a correlation-attenuation caveat: "
            "real per-feature AUROC estimates are correlated, unlike the "
            "lemma's i.i.d. assumption; correlation attenuates the expected "
            "max relative to theory. A permuted-label version (closer to the "
            "i.i.d. assumption) does not yet exist for the protein/ESM2 "
            "domain and would need a new sweep — not computed here."
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-dir", action="append", required=True, dest="sweep_dirs",
        metavar="MODEL_NAME=PATH",
        help="Repeatable. e.g. esm2_8m=/scratch/.../sweep_esm2_8m_L5_k32_x8_100k",
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--exclude-go-id", action="append", default=[], dest="exclude_go_ids",
        metavar="GO:XXXXXXX",
        help="Repeatable. Exclude this GO term from every model/min_positives point before "
             "fitting -- for leverage checks (2026-08-28: GO:0005515 was found "
             "to single-handedly drive a spurious significant result in a related regression; "
             "this flag tests whether the same term has leverage on THIS curve's pooled slope).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dirs = {}
    for entry in args.sweep_dirs:
        name, path = entry.split("=", 1)
        sweep_dirs[name] = Path(path)

    result = fit_inflation_curve(sweep_dirs, exclude_go_ids=frozenset(args.exclude_go_ids))
    if args.exclude_go_ids:
        result["excluded_go_ids"] = args.exclude_go_ids

    logger.info("=" * 70)
    for model_name, fit in result["per_model_fits"].items():
        if "error" in fit:
            logger.info("%s: %s", model_name, fit["error"])
        else:
            logger.info(
                "%s: slope=%.3f  CI95=[%.3f, %.3f]  R^2=%.3f  "
                "consistent with theoretical -0.5: %s",
                model_name, fit["slope"], *fit["slope_ci95"], fit["r_squared"],
                fit["consistent_with_theoretical_slope"],
            )
    logger.info(
        "POOLED: slope=%.3f  CI95=[%.3f, %.3f]  R^2=%.3f  consistent with theoretical -0.5: %s",
        result["pooled_fit"]["slope"], *result["pooled_fit"]["slope_ci95"],
        result["pooled_fit"]["r_squared"], result["pooled_fit"]["consistent_with_theoretical_slope"],
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
