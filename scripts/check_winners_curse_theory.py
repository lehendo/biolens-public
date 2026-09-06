"""
Check the winner's-curse simulation's empirical AUROC-inflation curve
against the winner's-curse lemma's theoretical prediction.

The lemma predicts `E[max_i eps_i] ~ sigma * sqrt(2 ln d)` for `d` i.i.d.
mean-zero noise draws of standard deviation `sigma` (Gaussian order-
statistics), with `sigma` connected to sample size via the Hanley-McNeil
(1982) AUROC-variance formula. This script computes that prediction at
every `n_positive` point in the simulation's real output
(docs/winners_curse_results.json) and compares it against the measured
`auroc_inflation_above_null`.

`biolens.eval.probing._vectorized_auroc` (the function the simulation itself
uses) REFLECTS every AUROC below 0.5 up via `max(A, 1-A)` before the
max-over-features selection step -- see its docstring. The textbook,
unfolded sigma*sqrt(2 ln d) formula does not model this reflection, and
folding a Gaussian around its own mean provably shrinks its variance by a
factor of `1 - 2/pi` (Var(|X|) = sigma^2*(1-2/pi) for X ~ N(0, sigma^2)) --
a fixed, closed-form correction with no free parameter, not a value tuned
to improve the fit after the fact (see gaussian_max_inflation_prediction's
`folded=` argument). This script reports BOTH the unfolded and folded
predictions so the reasoning is auditable end to end, not just the version
that happens to match best.

No GPU, no new simulation -- pure re-analysis of the already-completed
simulation output sitting in docs/winners_curse_results.json.

Usage:
  python scripts/check_winners_curse_theory.py \\
      --results docs/winners_curse_results.json \\
      --output docs/winners_curse_theory_check.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.eval.statistics import (  # noqa: E402
    gaussian_max_inflation_prediction,
    hanley_mcneil_auroc_variance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def check_theory(results: list[dict]) -> dict:
    rows = []
    for r in results:
        n_pos = r["n_positive"]
        n_neg = r["n_total"] - n_pos
        d = r["d_sae"]
        sigma = hanley_mcneil_auroc_variance(0.5, n_pos, n_neg) ** 0.5

        theory_unfolded = gaussian_max_inflation_prediction(sigma, d, folded=False)
        theory_folded = gaussian_max_inflation_prediction(sigma, d, folded=True)
        empirical = r["auroc_inflation_above_null"]

        rows.append({
            "n_positive": n_pos,
            "n_negative": n_neg,
            "d_sae": d,
            "sigma_hanley_mcneil": sigma,
            "empirical_inflation": empirical,
            "theory_unfolded": theory_unfolded,
            "theory_folded": theory_folded,
            "ratio_empirical_over_unfolded": empirical / theory_unfolded,
            "ratio_empirical_over_folded": empirical / theory_folded,
        })

    ratios_unfolded = [row["ratio_empirical_over_unfolded"] for row in rows]
    ratios_folded = [row["ratio_empirical_over_folded"] for row in rows]

    return {
        "rows": rows,
        "unfolded_ratio_mean": statistics.mean(ratios_unfolded),
        "unfolded_ratio_stdev": statistics.stdev(ratios_unfolded),
        "folded_ratio_mean": statistics.mean(ratios_folded),
        "folded_ratio_stdev": statistics.stdev(ratios_folded),
        "interpretation": (
            "A ratio that is roughly CONSTANT across the two-decade n_positive "
            "sweep (low stdev relative to mean) is evidence the sigma*sqrt(2 ln d) "
            "power-law SHAPE is correct, independent of whether the absolute level "
            "matches 1.0 exactly. The folded correction is not fit to the data -- "
            "it is the closed-form variance of a reflected Gaussian, applied "
            "uniformly at every point with no free parameter."
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="docs/winners_curse_results.json")
    p.add_argument("--output", default="docs/winners_curse_theory_check.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.results) as f:
        results = json.load(f)

    check = check_theory(results)

    logger.info("=" * 78)
    logger.info(
        "%10s  %10s  %12s  %12s  %8s  %8s",
        "n_positive", "empirical", "theory(raw)", "theory(fold)", "ratio_raw", "ratio_fold",
    )
    for row in check["rows"]:
        logger.info(
            "%10d  %10.4f  %12.4f  %12.4f  %8.3f  %8.3f",
            row["n_positive"], row["empirical_inflation"], row["theory_unfolded"],
            row["theory_folded"], row["ratio_empirical_over_unfolded"],
            row["ratio_empirical_over_folded"],
        )
    logger.info("=" * 78)
    logger.info(
        "Unfolded ratio: mean=%.3f stdev=%.3f  |  Folded ratio: mean=%.3f stdev=%.3f",
        check["unfolded_ratio_mean"], check["unfolded_ratio_stdev"],
        check["folded_ratio_mean"], check["folded_ratio_stdev"],
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(check, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
