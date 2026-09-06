"""
Does AUROC-inflation (not verified RATE) scale with SAE dictionary width?

Pre-flight for a dictionary-width sweep: the originally-planned outcome
variable, verified RATE, is the wrong one for this hypothesis. The claim
under test is about SELECTION-INDUCED INFLATION (max over d_sae noisy
candidates), not the base per-feature verification rate — a wider
dictionary offers more candidates to max over, so inflation can rise
even as verified rate also rises (which the existing registries' verified
rate in fact does: 0.161 -> 0.333 going k32 4x -> 16x, opposite the
hypothesis a rate-based sweep was written to test).

This script instead computes `auroc_inflation` (the SAME quantity and SAME
estimator as scripts/fit_auroc_inflation_curve.py's Route 1 -- reusing
bootstrap_mean_auroc_inflation directly, no new statistics) across the four
existing ESM2-8M width/k pilot sweeps (k32_x4, k32_x16, k16_x8, k64_x8) at
every matched min_positives value. These sweeps already have
--held-out-baselines populated (confirmed 2026-08-15 via direct field check
on /scratch) -- NO NEW GPU COMPUTE NEEDED to answer this pre-flight
question, exactly as fit_auroc_inflation_curve.py needed none for the
cross-scale curve.

This does NOT replace the width sweep itself if that comparison shows a real
effect worth confirming at higher power -- it answers a narrower, cheaper
question first: does the existing (underpowered for verified-rate, but
sufficient for inflation, since inflation is a per-row continuous quantity
with cluster-bootstrap CIs rather than a small-n binomial proportion)
pilot data already show a width-inflation relationship worth a bigger run?

Usage:
  python scripts/fit_width_inflation_curve.py \\
      --arm k32_x4=/scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k32_x4 \\
      --arm k32_x16=/scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k32_x16 \\
      --arm k16_x8=/scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k16_x8 \\
      --arm k64_x8=/scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k64_x8 \\
      --output docs/experiment_width_inflation_pilot.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402

from biolens.eval.statistics import bootstrap_mean_auroc_inflation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_MIN_POSITIVES_POINTS = [10, 20, 50, 100, 150, 200, 300]

# (k, expansion) per arm -- matches configs/verified_features/esm2_8m_layer5_topk_*.yaml naming
_ARM_SPECS = {
    "k32_x4": {"k": 32, "expansion": 4},
    "k32_x16": {"k": 32, "expansion": 16},
    "k16_x8": {"k": 16, "expansion": 8},
    "k64_x8": {"k": 64, "expansion": 8},
}


def _load_inflation_rows(sweep_dir: Path, min_positives: int) -> tuple[np.ndarray, np.ndarray]:
    path = sweep_dir / f"mp{min_positives}" / "go_probing_results.json"
    with open(path) as f:
        rows = json.load(f)
    inflation = np.array(
        [r["auroc_inflation"] for r in rows if r.get("auroc_inflation") is not None]
    )
    feat_idx = np.array(
        [r["best_feature_idx"] for r in rows if r.get("auroc_inflation") is not None]
    )
    return inflation, feat_idx


def compute_per_arm_inflation(arm_dirs: dict[str, Path]) -> dict:
    per_arm: dict[str, list[dict]] = {}
    for arm_name, sweep_dir in arm_dirs.items():
        points = []
        for mp in _MIN_POSITIVES_POINTS:
            results_path = sweep_dir / f"mp{mp}" / "go_probing_results.json"
            if not results_path.exists():
                logger.warning("%s: missing %s — skipping mp=%d", arm_name, results_path, mp)
                continue
            inflation, feat_idx = _load_inflation_rows(sweep_dir, mp)
            if len(inflation) == 0:
                logger.warning("%s mp=%d: no rows with auroc_inflation — skipping", arm_name, mp)
                continue
            ci_result = bootstrap_mean_auroc_inflation(inflation, feat_idx)
            points.append({
                "min_positives": mp,
                "mean_auroc_inflation": ci_result.mean,
                "ci95": list(ci_result.ci),
                "n_rows": ci_result.n_rows,
                "n_clusters": ci_result.n_clusters,
            })
            logger.info(
                "%s mp=%-4d  mean_inflation=%.4f  ci95=[%.4f, %.4f]  n_clusters=%d",
                arm_name, mp, ci_result.mean, *ci_result.ci, ci_result.n_clusters,
            )
        per_arm[arm_name] = points
    return per_arm


def compare_pairs(per_arm: dict) -> dict:
    """At each matched min_positives, compare the two width-varying arms
    (k32_x4 vs k32_x16, fixed k=32) and the two k-varying arms (k16_x8 vs
    k64_x8, fixed 8x width) — CI overlap is a quick, honest signal, not a
    substitute for a real two-sample test, but it's what a pre-flight check
    needs before deciding whether a bigger run is worth it."""
    by_mp = {name: {p["min_positives"]: p for p in points} for name, points in per_arm.items()}
    comparisons = {"width_at_k32 (x4 vs x16)": [], "k_at_x8 (k16 vs k64)": []}
    for mp in _MIN_POSITIVES_POINTS:
        if mp in by_mp.get("k32_x4", {}) and mp in by_mp.get("k32_x16", {}):
            a, b = by_mp["k32_x4"][mp], by_mp["k32_x16"][mp]
            overlap = not (a["ci95"][1] < b["ci95"][0] or b["ci95"][1] < a["ci95"][0])
            comparisons["width_at_k32 (x4 vs x16)"].append({
                "min_positives": mp,
                "x4_inflation": a["mean_auroc_inflation"], "x4_ci95": a["ci95"],
                "x16_inflation": b["mean_auroc_inflation"], "x16_ci95": b["ci95"],
                "ci_overlap": overlap,
                "x16_higher": b["mean_auroc_inflation"] > a["mean_auroc_inflation"],
            })
        if mp in by_mp.get("k16_x8", {}) and mp in by_mp.get("k64_x8", {}):
            a, b = by_mp["k16_x8"][mp], by_mp["k64_x8"][mp]
            overlap = not (a["ci95"][1] < b["ci95"][0] or b["ci95"][1] < a["ci95"][0])
            comparisons["k_at_x8 (k16 vs k64)"].append({
                "min_positives": mp,
                "k16_inflation": a["mean_auroc_inflation"], "k16_ci95": a["ci95"],
                "k64_inflation": b["mean_auroc_inflation"], "k64_ci95": b["ci95"],
                "ci_overlap": overlap,
                "k64_higher": b["mean_auroc_inflation"] > a["mean_auroc_inflation"],
            })
    return comparisons


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm", action="append", required=True, dest="arms",
        metavar="ARM_NAME=PATH",
        help="Repeatable, ARM_NAME one of k32_x4/k32_x16/k16_x8/k64_x8. "
             "e.g. k32_x4=/scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k32_x4",
    )
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arm_dirs = {}
    for entry in args.arms:
        name, path = entry.split("=", 1)
        if name not in _ARM_SPECS:
            raise SystemExit(f"Unknown arm name {name!r}, expected one of {list(_ARM_SPECS)}")
        arm_dirs[name] = Path(path)

    per_arm = compute_per_arm_inflation(arm_dirs)
    comparisons = compare_pairs(per_arm)

    logger.info("=" * 70)
    logger.info("Width comparison at k=32 (does inflation rise with dictionary width?):")
    for row in comparisons["width_at_k32 (x4 vs x16)"]:
        logger.info(
            "  mp=%-4d  x4=%.4f %s  x16=%.4f %s  overlap=%s  x16_higher=%s",
            row["min_positives"], row["x4_inflation"], row["x4_ci95"],
            row["x16_inflation"], row["x16_ci95"], row["ci_overlap"], row["x16_higher"],
        )
    logger.info("k comparison at 8x width (does inflation depend on sparsity k?):")
    for row in comparisons["k_at_x8 (k16 vs k64)"]:
        logger.info(
            "  mp=%-4d  k16=%.4f %s  k64=%.4f %s  overlap=%s  k64_higher=%s",
            row["min_positives"], row["k16_inflation"], row["k16_ci95"],
            row["k64_inflation"], row["k64_ci95"], row["ci_overlap"], row["k64_higher"],
        )

    result = {"arm_specs": _ARM_SPECS, "per_arm_points": per_arm, "comparisons": comparisons}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
