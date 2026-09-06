"""
Permuted-label AUROC-inflation-vs-n curve — a follow-up to
fit_auroc_inflation_curve.py addressing its correlation-attenuation caveat.

fit_auroc_inflation_curve.py's real-label curve is subject to a caveat the
lemma's `sigma*sqrt(2 ln d)` prediction doesn't account for: real per-feature
AUROC estimates are correlated (shared eval proteins, correlated SAE feature
directions), which attenuates the expected max relative to the lemma's i.i.d.
assumption. Permuted (label-shuffled) inflation isolates the pure sampling-
noise component specifically -- real feature activations against a shuffled
label have no real relationship to that label by construction, so ANY
per-feature correlation from shared activations does not translate into
label-conditional correlation the way it does for the real-label curve. This
is the version that should be checked against the theoretical scaling most
directly, since the theory's i.i.d. assumption is closest to true there.

No new GPU compute needed: `noise_confounding_decomposition.json` (produced
by run_eval.py --permuted-label-control, already run for all four ESM2
scales' 100K sweeps for the "permuted-label subtraction doesn't work" finding)
already has `mean_permuted_inflation` and
`permuted_inflation_ci95` per GO term at every min_positives point, sitting on
disk -- this is a pure re-analysis, joined against the matching
go_probing_results.json (same directory) by go_id to attach best_feature_idx
for cluster-robust bootstrap CIs, exactly as fit_auroc_inflation_curve.py does
for the real-label curve.

Usage:
  python scripts/fit_permuted_inflation_curve.py \\
      --sweep-dir sweep_esm2_8m_L5_k32_x8_100k=/scratch/.../sweep_esm2_8m_L5_k32_x8_100k \\
      --sweep-dir sweep_esm2_35m_L5_k32_x8_100k=/scratch/.../sweep_esm2_35m_L5_k32_x8_100k \\
      --sweep-dir sweep_esm2_150m_L14_k32_x8_100k=/scratch/.../sweep_esm2_150m_L14_k32_x8_100k \\
      --sweep-dir sweep_esm2_650m_L16_k32_x8_100k=/scratch/.../sweep_esm2_650m_L16_k32_x8_100k \\
      --output docs/experiment_permuted_inflation_vs_n.json
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


def _load_permuted_inflation_rows(sweep_dir: Path, min_positives: int) -> tuple[np.ndarray, np.ndarray]:
    """Join go_probing_results.json (has best_feature_idx, keyed by go_id)
    against noise_confounding_decomposition.json (has mean_permuted_inflation,
    also keyed by go_id) -- the two files are produced by the same run_eval.py
    invocation over the same GO-term set, so every decomposition row's go_id
    must resolve against the probing results (asserted, not assumed)."""
    mp_dir = sweep_dir / f"mp{min_positives}"
    with open(mp_dir / "go_probing_results.json") as f:
        probing_rows = json.load(f)
    with open(mp_dir / "noise_confounding_decomposition.json") as f:
        decomposition_rows = json.load(f)

    feat_idx_by_go_id = {r["go_id"]: r["best_feature_idx"] for r in probing_rows}

    inflation, feat_idx = [], []
    for r in decomposition_rows:
        if r.get("mean_permuted_inflation") is None:
            continue
        go_id = r["go_id"]
        if go_id not in feat_idx_by_go_id:
            raise ValueError(
                f"{mp_dir}: go_id {go_id!r} in noise_confounding_decomposition.json "
                f"has no matching row in go_probing_results.json -- the two files "
                f"should be produced by the same run and share every go_id"
            )
        inflation.append(r["mean_permuted_inflation"])
        feat_idx.append(feat_idx_by_go_id[go_id])

    return np.array(inflation), np.array(feat_idx)


def _row_identity_key(sweep_dir: Path, min_positives: int) -> frozenset:
    """Same duplicate-sweep-point detection as fit_auroc_inflation_curve.py's
    _row_identity_key, keyed on the permuted value instead of the real one --
    a fixed max_go_terms cap can bind before min_positives does at the low
    end of a sweep, producing byte-identical row sets at e.g. mp10 and mp20."""
    inflation, feat_idx = _load_permuted_inflation_rows(sweep_dir, min_positives)
    return frozenset(zip(feat_idx.tolist(), inflation.tolist()))


def fit_permuted_inflation_curve(sweep_dirs: dict[str, Path]) -> dict:
    per_model: dict[str, list[dict]] = {}
    all_log_mp: list[float] = []
    all_log_inflation: list[float] = []
    all_model_labels: list[str] = []

    for model_name, sweep_dir in sweep_dirs.items():
        points = []
        seen_row_keys: dict[frozenset, int] = {}
        for mp in _MIN_POSITIVES_POINTS:
            decomp_path = sweep_dir / f"mp{mp}" / "noise_confounding_decomposition.json"
            if not decomp_path.exists():
                logger.warning("%s: missing %s — skipping mp=%d", model_name, decomp_path, mp)
                continue

            row_key = _row_identity_key(sweep_dir, mp)
            duplicate_of = seen_row_keys.get(row_key)
            seen_row_keys.setdefault(row_key, mp)

            inflation, feat_idx = _load_permuted_inflation_rows(sweep_dir, mp)
            if len(inflation) == 0:
                logger.warning(
                    "%s mp=%d: no rows with mean_permuted_inflation — skipping", model_name, mp,
                )
                continue

            ci_result = bootstrap_mean_auroc_inflation(inflation, feat_idx)
            point = {
                "min_positives": mp,
                "mean_permuted_inflation": ci_result.mean,
                "ci95": list(ci_result.ci),
                "n_rows": ci_result.n_rows,
                "n_clusters": ci_result.n_clusters,
                "duplicate_of_min_positives": duplicate_of,
            }
            points.append(point)
            logger.info(
                "%s mp=%-4d  mean_permuted_inflation=%.4f  ci95=[%.4f, %.4f]  n_clusters=%d%s",
                model_name, mp, ci_result.mean, *ci_result.ci, ci_result.n_clusters,
                f"  [DUPLICATE of mp={duplicate_of}, excluded from fit]" if duplicate_of else "",
            )

            # Same rule as fit_auroc_inflation_curve.py: only the first point
            # in an identical-row-set group, and only positive means (log
            # undefined otherwise -- a genuine possibility here since permuted
            # inflation can be slightly negative at some points, unlike the
            # real-label curve's held-out gap).
            if duplicate_of is None and ci_result.mean > 0:
                all_log_mp.append(float(np.log(mp)))
                all_log_inflation.append(float(np.log(ci_result.mean)))
                all_model_labels.append(model_name)

        per_model[model_name] = points

    # ── Per-model log-log OLS fits ────────────────────────────────────────
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

    # ── Pooled fit ──────────────────────────────────────────────────────
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
            "Permuted-label curve -- isolates the pure sampling-noise "
            "component (real feature activations against a shuffled label "
            "have no real relationship to that label by construction), the "
            "closest real-data analog to the lemma's i.i.d.-noise assumption. "
            "Compare against fit_auroc_inflation_curve.py's real-label curve "
            "and docs/winners_curse_theory_check.json's synthetic prediction."
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dirs = {}
    for entry in args.sweep_dirs:
        name, path = entry.split("=", 1)
        sweep_dirs[name] = Path(path)

    result = fit_permuted_inflation_curve(sweep_dirs)

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
