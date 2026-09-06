"""
Compute the Geuvadis case study's naive-arm mediated-fraction equivalence
test: "statistically equivalent to zero against a +/-10% margin".

Real gap fixed 2026-08-29: a previously reported equivalence-test result
(z=3.79, p=0.00008, +/-5.2% observed CI) had no source producing it anywhere
in the codebase -- a grep for "tost"/"equivalence" across every .py file,
tracked and untracked, returned zero hits. It independently reproduced by
hand against docs/geuvadis_case_study_results.json before this script
existed; this script is that hand computation made real and reproducible,
using biolens.eval.statistics.tost_equivalence_test.

Usage:
    python scripts/compute_geuvadis_equivalence.py \\
        --results docs/geuvadis_case_study_results.json \\
        --margin 0.10 \\
        --output docs/experiment_geuvadis_equivalence.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.eval.statistics import tost_equivalence_test  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="docs/geuvadis_case_study_results.json")
    p.add_argument(
        "--margin", type=float, default=0.10,
        help="Equivalence margin as a fraction (default 0.10 = +/-10 percentage points).",
    )
    p.add_argument("--output", default="docs/experiment_geuvadis_equivalence.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.results).read_text())

    # Same two gates the raw field names encode: ACME must be reportable
    # (excludes the degenerate zero-variance-mediator case, see
    # MediationResult.acme_reportable) AND proportion_mediated specifically
    # must be reportable (a ratio estimator is additionally unstable near a
    # near-zero total effect).
    reportable = [
        r for r in rows
        if r["naive_acme_reportable"] and r["naive_proportion_mediated_reportable"]
    ]
    values = [r["naive_proportion_mediated"] for r in reportable]
    excluded_loci = [
        r["variant_id"] for r in rows
        if r["naive_acme_reportable"] and not r["naive_proportion_mediated_reportable"]
    ]

    logger.info(
        "%d of %d ACME-reportable loci also have a reportable proportion_mediated "
        "(excluded for a near-zero total effect: %s)",
        len(values), sum(1 for r in rows if r["naive_acme_reportable"]), excluded_loci,
    )

    result = tost_equivalence_test(values, margin=args.margin)
    logger.info(str(result))

    genes = [r["gene_id"] for r in reportable]
    output = {
        "source_file": args.results,
        "n_loci_included": result.n,
        "included_variant_ids": [r["variant_id"] for r in reportable],
        "excluded_for_near_zero_total_effect": excluded_loci,
        "n_distinct_genes": len(set(genes)),
        "clustering_note": (
            "n_loci_included == n_distinct_genes: no repeated-gene clustering "
            "present, so the plain i.i.d. standard error below is not "
            "understating variance the way it would if genes repeated."
            if len(set(genes)) == result.n
            else "WARNING: repeated genes present among included loci -- "
            "the plain i.i.d. SE below is NOT cluster-robust and likely "
            "understates variance. Re-derive with a cluster-robust or "
            "cluster-bootstrap SE before reporting this number."
        ),
        "margin": result.margin,
        "mean": result.mean,
        "se": result.se,
        "ci95": list(result.ci95),
        "z_lower": result.z_lower,
        "z_upper": result.z_upper,
        "p_lower": result.p_lower,
        "p_upper": result.p_upper,
        "p_value_binding": result.p_value,
        "equivalent_at_alpha": {str(k): v for k, v in result.equivalent_at_alpha.items()},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
