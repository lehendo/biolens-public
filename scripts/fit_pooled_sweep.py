"""
Fit the confirmatory sweep model ACROSS multiple ESM2 scales in one pooled
regression: `verified_rate ~ log(min_positives) * model_scale`, clustered
on (model_scale, feature_idx).

This is the cross-model comparison itself — not four separate
scripts/fit_sweep.py runs eyeballed side by side. Comparing significance
across models with unequal registry sizes that way is a known statistical
mistake (the difference between significant and not significant is not
itself significant), and is the exact failure mode the GWAS winner's-curse
replication-variability literature attributes anomalous replication patterns
to. The pooled model's joint interaction test answers "does the slope
actually differ by scale" directly, and each model still contributes its
own cluster count — no matched sample size required.

Usage:
  python scripts/fit_pooled_sweep.py \\
      --sweep-dir esm2_8m=/scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k32_x8 \\
      --sweep-dir esm2_35m=/scratch/arjunc4/biolens/eval/sweep_esm2_35m_L5_k32_x8 \\
      --sweep-dir esm2_150m=/scratch/arjunc4/biolens/eval/sweep_esm2_150m_L14_k32_x8 \\
      --sweep-dir esm2_650m=/scratch/arjunc4/biolens/eval/sweep_esm2_650m_L16_k32_x8 \\
      --verified-features esm2_8m=configs/verified_features/esm2_8m_layer5_topk_k32.yaml \\
      --verified-features esm2_35m=configs/verified_features/esm2_35m_layer5_topk_k32.yaml \\
      --verified-features esm2_150m=configs/verified_features/esm2_150m_layer14_topk_k32.yaml \\
      --verified-features esm2_650m=configs/verified_features/esm2_650m_layer16_topk_k32.yaml \\
      --scale-param-counts esm2_8m=8e6 --scale-param-counts esm2_35m=35e6 \\
      --scale-param-counts esm2_150m=150e6 --scale-param-counts esm2_650m=650e6 \\
      --output docs/experiment3_pooled_fit_with_trend.json

NOTE on --output: don't point this at docs/experiment3_pooled_fit.json, the
existing baseline every downstream reported number was checked against — the
write is in-place, and overwriting it means a later "does this look off"
question can't distinguish "the trend code is wrong" from "the /scratch
sweep inputs changed". Write to a new path, diff every pre-existing field
against the baseline for exact reproducibility (same
random_state=42/n_bootstrap=2000 means every non-trend field must match
exactly), then promote once confirmed.

--scale-param-counts is optional (repeatable label=count, same pattern as
--sweep-dir/--verified-features). If given for every scale, also fits a
cross-scale trend (does the slope decline with model size?) via
fit_pooled_scale_interaction's scale_param_counts argument — see
PooledScaleFitResult's docstring for why this is computed via the existing
stratified cluster bootstrap rather than an ordinary regression on the four
already-extracted per-scale slopes (they are correlated, not independent).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.eval.feature_inspection import load_verified_annotations
from biolens.eval.statistics import build_pooled_sweep_rows, fit_pooled_scale_interaction

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
    p.add_argument(
        "--verified-features", action="append", required=True, type=_parse_labeled_path,
        metavar="LABEL=PATH",
        help="model_scale=verified_features/*.yaml, repeatable. Labels must match --sweep-dir.",
    )
    p.add_argument(
        "--scale-param-counts", action="append", default=[], metavar="LABEL=COUNT",
        help="model_scale=parameter_count (e.g. esm2_8m=8e6), repeatable, OPTIONAL. If given "
             "for every scale in --sweep-dir, also fits a cross-scale trend against log10(count).",
    )
    p.add_argument("--reference-scale", default=None, help="Which scale's slope is the base coefficient (default: first alphabetically)")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output", default=None, help="Optional path to save the fit result as JSON")
    return p.parse_args()


def load_sweep_results(sweep_dir: Path) -> dict[int, list[dict]]:
    """Same mp<N>/go_probing_results.json loader as scripts/fit_sweep.py."""
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

    sweep_labels = {label for label, _ in args.sweep_dir}
    verified_labels = {label for label, _ in args.verified_features}
    if sweep_labels != verified_labels:
        raise SystemExit(
            f"--sweep-dir labels {sweep_labels} and --verified-features labels "
            f"{verified_labels} must match exactly"
        )

    per_model_sweep_results = {}
    for label, path in args.sweep_dir:
        logger.info("Loading sweep for scale=%s from %s", label, path)
        results = load_sweep_results(path)
        if not results:
            raise SystemExit(f"No sweep results found under {path} for scale={label}")
        per_model_sweep_results[label] = results

    per_model_verified = {}
    for label, path in args.verified_features:
        verified = load_verified_annotations(path)
        logger.info("Loaded %d verified-feature registry entries for scale=%s", len(verified), label)
        per_model_verified[label] = verified

    min_positives, target, feature_idx, model_scale = build_pooled_sweep_rows(
        per_model_sweep_results, per_model_verified
    )
    if len(min_positives) == 0:
        raise SystemExit(
            "No sweep rows matched any registry across any scale — nothing to fit."
        )
    logger.info(
        "Built %d pooled confirmatory rows across %d scales",
        len(min_positives), len(set(model_scale.tolist())),
    )

    scale_param_counts = None
    if args.scale_param_counts:
        scale_param_counts = {}
        for spec in args.scale_param_counts:
            if "=" not in spec:
                raise SystemExit(f"--scale-param-counts expects label=count, got: {spec!r}")
            label, _, count_str = spec.partition("=")
            scale_param_counts[label] = float(count_str)
        missing = sweep_labels - set(scale_param_counts)
        if missing:
            raise SystemExit(
                f"--scale-param-counts is missing entries for {missing} — "
                f"provide one for every --sweep-dir scale, or omit it entirely"
            )
        logger.info("Fitting cross-scale trend against log10(param count): %s", scale_param_counts)

    result = fit_pooled_scale_interaction(
        min_positives, target, feature_idx, model_scale,
        reference_scale=args.reference_scale,
        n_bootstrap=args.n_bootstrap, confidence=args.confidence, random_state=args.random_state,
        scale_param_counts=scale_param_counts,
    )
    print("\n" + str(result))

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "reference_scale": result.reference_scale,
                    "scales": result.scales,
                    "slope": result.slope,
                    "slope_se_cluster": result.slope_se_cluster,
                    "slope_ci_cluster": {k: list(v) for k, v in result.slope_ci_cluster.items()},
                    "slope_p_value_cluster": result.slope_p_value_cluster,
                    "slope_ci_bootstrap": (
                        {k: list(v) for k, v in result.slope_ci_bootstrap.items()}
                        if result.slope_ci_bootstrap else None
                    ),
                    "interaction_wald_stat": result.interaction_wald_stat,
                    "interaction_p_value": result.interaction_p_value,
                    "n_rows": result.n_rows,
                    "n_clusters_total": result.n_clusters_total,
                    "n_clusters_per_scale": result.n_clusters_per_scale,
                    "n_bootstrap_successful": result.n_bootstrap_successful,
                    "scale_param_counts": scale_param_counts,
                    "scale_trend_slope": result.scale_trend_slope,
                    "scale_trend_ci_bootstrap": (
                        list(result.scale_trend_ci_bootstrap)
                        if result.scale_trend_ci_bootstrap else None
                    ),
                    "scale_trend_p_bootstrap": result.scale_trend_p_bootstrap,
                    "scale_trend_bootstrap_draws": result.scale_trend_bootstrap_draws,
                },
                indent=2,
            )
        )
        logger.info("Saved fit result to %s", args.output)


if __name__ == "__main__":
    main()
