"""
Winner's-curse simulation.

Demonstrates the theoretical mechanism behind the sample-size artifact found in early
GO probing runs: reporting
max_i AUROC_i over d_sae candidate features is a biased-high estimator of the true best
feature's AUROC when each per-feature AUROC estimate is itself noisy — classical winner's
curse / selection bias in order statistics. The bias should shrink as n_positive grows
(each per-feature AUROC estimate gets less noisy) and should NOT require any real signal:
here every synthetic feature is drawn with scores independent of labels by construction,
so any max AUROC measured above the reflected null (see below) is entirely a sampling
artifact.

Reuses biolens.eval.probing._vectorized_auroc — the exact same AUROC computation used by
the real GO-probing pipeline — so this simulation is methodologically identical to
production, not a separate reimplementation that could silently diverge. That function
REFLECTS every per-feature AUROC below 0.5 up via `max(auroc, 1-auroc)` (see its own
docstring), which is a real, second, DIFFERENT selection effect layered on top of the
max-over-features selection this simulation is designed to isolate. Reflection is itself
a maximization — of a statistic against its own complement — so the reflected AUROC's
null-hypothesis expectation is NOT 0.5. For a random variable A symmetric around 0.5,
E[max(A, 1-A)] = 0.5 + E[|A - 0.5|] > 0.5 whenever A has any sampling variance at all,
which every finite-sample AUROC estimate does. This gap is largest exactly where
per-feature AUROC is noisiest — i.e. at small n_positive — and shrinks toward 0 as
n_positive grows and each AUROC estimate's variance shrinks. Real bug fixed 2026-08-05:
an earlier version of this script's saved `inflation_above_null` used a hardcoded `- 0.5`
baseline instead of the reflected statistic's actual (larger) measured null, overstating
inflation by ~28% at every sweep point. This script's OWN inflation calculation below
(`max_arr.mean() - mean_arr.mean()`) was already correct; only the stale, unregenerated
saved file and the surrounding prose were wrong.

Metric-robustness extension: AUROC and AP are
threshold-free (rank-based, no true-signal case needed by construction), computed here
with the identical selection-under-noise setup so the pattern's presence or absence in
each metric is directly comparable. F1 and MCC need a decision threshold — none of these
metrics are used anywhere in the real production probing pipeline (which is AUROC-only),
so there's no "production threshold" to match; a fixed threshold of 0 (the true median of
the standard-normal score distribution used throughout this simulation) is used,
deliberately NOT optimized per feature, so the max-over-d_sae selection step is the only
source of selection bias being measured — optimizing the threshold per feature would
introduce a second, confounding selection-bias source (best-threshold-over-many, not just
best-feature-over-many) that this specific check isn't designed to isolate.

Usage:
  python scripts/winners_curse_simulation.py \\
      --d-sae 2560 --n-total 10000 \\
      --n-positive-values 10 20 50 100 150 200 300 500 1000 2000 \\
      --n-trials 200 --output docs/winners_curse_results.json
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

from biolens.eval.probing import _vectorized_auroc  # noqa: E402 (path insert above)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

METRICS = ("auroc", "ap", "f1", "mcc")


def _vectorized_ap(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Average precision for every column of scores independently — threshold-free,
    same rank-based spirit as _vectorized_auroc, fully vectorized across all d_sae
    columns at once (no per-feature sklearn.metrics.average_precision_score loop,
    which would be far too slow at real d_sae/n_trials scale).

    AP = sum_k precision@k * [rank k is a true positive] / n_positive, walking down
    each column's scores in descending order.

    Args:
        scores: (N, d_sae) float array.
        labels: (N,) binary int array.

    Returns:
        (d_sae,) float array of average-precision scores.
    """
    n_pos = int(labels.sum())
    if n_pos == 0:
        return np.zeros(scores.shape[1])

    order = np.argsort(-scores, axis=0)  # (N, d_sae), descending per column
    sorted_labels = np.take_along_axis(labels[:, None], order, axis=0)  # (N, d_sae)

    cum_tp = np.cumsum(sorted_labels, axis=0)  # (N, d_sae)
    ranks = np.arange(1, scores.shape[0] + 1, dtype=np.float64)[:, None]  # (N, 1)
    precision_at_k = cum_tp / ranks

    return (precision_at_k * sorted_labels).sum(axis=0) / n_pos


def _vectorized_f1_mcc(
    scores: np.ndarray, labels: np.ndarray, threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    F1 and Matthews correlation coefficient for every column of scores
    independently, at a FIXED threshold (see module docstring for why this is
    deliberately not optimized per feature). Fully vectorized via boolean
    confusion-matrix counts, not a per-feature sklearn loop.

    Args:
        scores: (N, d_sae) float array.
        labels: (N,) binary int array.
        threshold: Decision threshold — predict positive iff score > threshold.

    Returns:
        (f1, mcc) — each a (d_sae,) float array. A feature with zero predicted
        positives or zero predicted negatives gets F1=0.0 and MCC=0.0 (the
        standard convention for an undefined/degenerate confusion matrix,
        rather than propagating NaN into downstream aggregate statistics).
    """
    pred_pos = scores > threshold  # (N, d_sae)
    labels_col = labels[:, None].astype(bool)  # (N, 1), broadcasts against pred_pos

    tp = (pred_pos & labels_col).sum(axis=0).astype(np.float64)
    fp = (pred_pos & ~labels_col).sum(axis=0).astype(np.float64)
    fn = (~pred_pos & labels_col).sum(axis=0).astype(np.float64)
    tn = (~pred_pos & ~labels_col).sum(axis=0).astype(np.float64)

    f1_denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, f1_denom, out=np.zeros_like(tp), where=f1_denom > 0)

    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn, mcc_denom, out=np.zeros_like(tp), where=mcc_denom > 0
    )

    return f1, mcc


def run_trial(
    n_total: int, n_positive: int, d_sae: int, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    """
    One Monte Carlo trial: generate d_sae features with NO real signal (scores drawn
    independently of labels), compute all four per-feature metrics via vectorized
    production-equivalent code paths, and return {metric: (max, mean)} across features.

    True value for every feature/metric is its own null-hypothesis baseline in
    expectation by construction — labels are assigned independently of scores. For
    AUROC specifically, that baseline is NOT 0.5: _vectorized_auroc reflects every
    per-feature value below 0.5 up via max(auroc, 1-auroc), and reflection is itself a
    (smaller, per-feature) maximization, which pushes the null strictly above 0.5 by an
    amount that shrinks as n_positive grows (see module docstring). AP's null depends on
    prevalence (~n_positive/n_total); F1/MCC are ~0 for a random classifier at this
    threshold. Any max() measured above the metric's OWN measured mean-of-mean baseline
    is entirely a sampling artifact, for every metric checked here — the run() loop below
    reports inflation against that measured baseline, not a hardcoded theoretical
    constant, for exactly this reason.
    """
    scores = rng.standard_normal((n_total, d_sae)).astype(np.float32)
    labels = np.zeros(n_total, dtype=int)
    positive_idx = rng.choice(n_total, size=n_positive, replace=False)
    labels[positive_idx] = 1

    aurocs = _vectorized_auroc(scores, labels)
    aps = _vectorized_ap(scores, labels)
    f1s, mccs = _vectorized_f1_mcc(scores, labels)

    return {
        "auroc": (float(aurocs.max()), float(aurocs.mean())),
        "ap": (float(aps.max()), float(aps.mean())),
        "f1": (float(f1s.max()), float(f1s.mean())),
        "mcc": (float(mccs.max()), float(mccs.mean())),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d-sae", type=int, default=2560, help="Matches ESM2 8M's real d_sae")
    p.add_argument("--n-total", type=int, default=10000, help="Matches Phase 0's eval slice size")
    p.add_argument(
        "--n-positive-values", type=int, nargs="+",
        default=[10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000],
        help="Pre-registered sweep points — must match Experiment 3's real sweep for direct comparison",
    )
    p.add_argument("--n-trials", type=int, default=200, help="Monte Carlo repeats per sweep point")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Optional path to save results as JSON")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    results = []
    for n_positive in args.n_positive_values:
        if n_positive >= args.n_total:
            logger.warning("Skipping n_positive=%d >= n_total=%d", n_positive, args.n_total)
            continue

        t0 = time.time()
        max_vals: dict[str, list[float]] = {m: [] for m in METRICS}
        mean_vals: dict[str, list[float]] = {m: [] for m in METRICS}
        for _ in range(args.n_trials):
            trial = run_trial(args.n_total, n_positive, args.d_sae, rng)
            for m in METRICS:
                max_a, mean_a = trial[m]
                max_vals[m].append(max_a)
                mean_vals[m].append(mean_a)

        result = {"n_positive": n_positive, "n_trials": args.n_trials,
                   "d_sae": args.d_sae, "n_total": args.n_total}
        for m in METRICS:
            max_arr = np.array(max_vals[m])
            mean_arr = np.array(mean_vals[m])

            # Bootstrap CI on the mean of the max-metric distribution across trials.
            boot_means = [
                rng.choice(max_arr, size=len(max_arr), replace=True).mean()
                for _ in range(2000)
            ]
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

            # "Null baseline" is the empirically-measured mean-across-features
            # value, not a hardcoded constant. This matters for every metric
            # here, AUROC included: AP's null depends on prevalence
            # (~n_positive/n_total for random scores), F1's null depends on
            # both prevalence AND the fixed threshold's implied predicted-
            # positive rate, MCC's null is ~0 regardless of prevalence — and
            # AUROC's null is NOT simply 0.5, because _vectorized_auroc
            # reflects every per-feature value below 0.5 up via
            # max(auroc, 1-auroc) (see module docstring), which is itself a
            # maximization that pushes the true null strictly above 0.5.
            # REAL BUG, fixed 2026-08-05:
            # an earlier version of this comment claimed using mean_of_mean
            # as the baseline was "numerically equivalent to the original
            # hardcoded -0.5 for AUROC specifically, since mean_of_mean_auroc
            # is itself empirically ~0.5 by construction" — that is false;
            # mean_of_mean_auroc is ~0.573 at n_positive=10, not ~0.5, and
            # the previously-saved docs/winners_curse_results.json (computed
            # with the old hardcoded -0.5) overstated inflation by ~28% at
            # every sweep point as a direct result. Using each metric's own
            # measured mean_of_mean as its baseline (as this code already
            # does) is the CORRECT choice, not an equivalent shortcut to a
            # hardcoded constant — there was never a valid hardcoded constant
            # for reflected AUROC to begin with.
            result[f"mean_of_max_{m}"] = float(max_arr.mean())
            result[f"max_{m}_ci_95"] = [float(ci_low), float(ci_high)]
            result[f"std_of_max_{m}"] = float(max_arr.std())
            result[f"mean_of_mean_{m}"] = float(mean_arr.mean())
            result[f"{m}_inflation_above_null"] = float(max_arr.mean() - mean_arr.mean())

        results.append(result)

        logger.info(
            "n_positive=%5d  mean(max AUROC)=%.4f  95%% CI=[%.4f, %.4f]  "
            "AP infl=%.4f  F1 infl=%.4f  MCC infl=%.4f  [%.1fs]",
            n_positive, result["mean_of_max_auroc"],
            result["max_auroc_ci_95"][0], result["max_auroc_ci_95"][1],
            result["ap_inflation_above_null"], result["f1_inflation_above_null"],
            result["mcc_inflation_above_null"], time.time() - t0,
        )

    print("\n" + "=" * 100)
    print("WINNER'S CURSE SIMULATION — no real signal (every feature's true value is its null baseline)")
    print("=" * 100)
    print(
        f"{'n_positive':>10}  {'AUROC infl':>11}  {'AP infl':>9}  "
        f"{'F1 infl':>9}  {'MCC infl':>9}"
    )
    for r in results:
        print(
            f"{r['n_positive']:>10}  {r['auroc_inflation_above_null']:>11.4f}  "
            f"{r['ap_inflation_above_null']:>9.4f}  {r['f1_inflation_above_null']:>9.4f}  "
            f"{r['mcc_inflation_above_null']:>9.4f}"
        )
    print("=" * 100)
    print(
        "Expected pattern (metric-robustness check): inflation "
        "is large at small n_positive and shrinks toward 0 as n_positive grows, for ALL FOUR "
        "metrics, not just AUROC — this is pure selection bias (winner's curse) driven by "
        "reporting max_i over d_sae candidates, a property of the selection step itself, not "
        "of AUROC specifically. Every synthetic feature has zero true signal by construction."
    )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
