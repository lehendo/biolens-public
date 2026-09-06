"""
Does GO annotation contamination (unannotated true positives) vary with GO
term size, beyond the count channel already captured by `n_positive`?

Refined 2026-08-27 (sharpened after an earlier, cruder "regress
completeness on term size" framing was rejected as conflating two channels):
the COUNT channel -- incompleteness meaning fewer annotated positives -- is
already on the confirmatory sweep's own x-axis, since the Hanley-McNeil
variance formula this project's theory check uses is driven by `n_positive`
AS MEASURED, not some unobservable "true" count. That's absorbed by
construction, not a confound. What's NOT captured is CONTAMINATION: true
positives sitting unlabeled in the negative set, which the measured
`n_positive` can't see. This script tests whether a proxy for that -- the
fraction of a GO term's real annotations that rest on experimental
(non-electronic/non-IEA) evidence, versus the bulk computational IEA
annotations -- varies with term size (`n_positive`, the same quantity the
confirmatory sweep already uses).

Explicitly NOT testing raw annotation completeness against term size (that
conflates the two channels and would double-count the sweep's own x-axis);
this is a depth/contamination proxy specifically.

Term set and n_positive: pulled directly from a real confirmatory-sweep
result file (go_probing_results.json, mp10 -- the largest, most complete
candidate pool, giving the widest range of n_positive), not the smaller,
skewed verified-features registry sample.

If QuickGO retrieval becomes unreliable or open-ended, ship the slope on
whatever subset was successfully queried and report the query's own
coverage rather than let this become an open-ended project.

Usage:
  python scripts/check_go_annotation_depth_vs_term_size.py \\
      --sweep-file <local copy of go_probing_results.json, mp10> \\
      --output docs/experiment_go_annotation_depth_vs_term_size.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import requests
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_QUICKGO_BASE = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
# Standard experimental/manually-curated GO evidence codes (excludes IEA —
# electronic/automatic annotation — and computational-inference-only codes
# like ISS/ISA/ISO/ISM/IGC/RCA/IBA/IBD/IKR/IRD/ISS, and author-statement-
# only codes TAS/NAS/IC, since none of those involve direct experimental
# evidence for the annotated function).
_EXPERIMENTAL_CODES = "EXP,IDA,IPI,IMP,IGI,IEP,HTP,HDA,HMP,HGI,HEP"


def query_evidence_fraction(go_id: str, session: requests.Session, timeout: float = 10.0) -> dict | None:
    """Returns {"total": int, "experimental": int, "fraction_experimental": float}
    for one GO term's real annotations in QuickGO, or None on failure."""
    try:
        total_resp = session.get(
            _QUICKGO_BASE, params={"goId": go_id, "goUsage": "exact", "limit": 1}, timeout=timeout,
        )
        total_resp.raise_for_status()
        total = total_resp.json()["numberOfHits"]

        if total == 0:
            return None

        exp_resp = session.get(
            _QUICKGO_BASE,
            params={"goId": go_id, "goUsage": "exact", "goIdEvidence": _EXPERIMENTAL_CODES, "limit": 1},
            timeout=timeout,
        )
        exp_resp.raise_for_status()
        experimental = exp_resp.json()["numberOfHits"]

        return {
            "total": total,
            "experimental": experimental,
            "fraction_experimental": experimental / total,
        }
    except Exception as exc:  # noqa: BLE001 -- one term's failure must not abort the batch
        logger.debug("Query failed for %s: %s", go_id, exc)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-file", required=True, help="Local go_probing_results.json to pull GO terms + n_positive from")
    p.add_argument("--output", required=True)
    p.add_argument("--sleep-between-calls", type=float, default=0.15)
    p.add_argument("--max-terms", type=int, default=None, help="Optional cap for a bounded run")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.sweep_file) as f:
        rows = json.load(f)
    terms = {r["go_id"]: r["n_positives"] for r in rows}
    logger.info("Loaded %d distinct GO terms from %s", len(terms), args.sweep_file)

    go_ids = sorted(terms.keys())
    if args.max_terms:
        go_ids = go_ids[: args.max_terms]

    session = requests.Session()
    results = []
    start = time.time()
    for i, go_id in enumerate(go_ids):
        r = query_evidence_fraction(go_id, session)
        if r is not None:
            r["go_id"] = go_id
            r["n_positive"] = terms[go_id]
            results.append(r)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            logger.info("%d/%d queried (%d succeeded), %.0fs elapsed", i + 1, len(go_ids), len(results), elapsed)
        time.sleep(args.sleep_between_calls)

    coverage = len(results) / len(go_ids) if go_ids else 0.0
    logger.info("Query coverage: %d/%d terms (%.1f%%)", len(results), len(go_ids), coverage * 100)

    out = {
        "n_terms_attempted": len(go_ids),
        "n_terms_succeeded": len(results),
        "coverage": coverage,
        "experimental_evidence_codes_used": _EXPERIMENTAL_CODES,
        "per_term": results,
    }

    if len(results) >= 10:
        n_pos = np.array([r["n_positive"] for r in results], dtype=float)
        frac_exp = np.array([r["fraction_experimental"] for r in results], dtype=float)
        log_n_pos = np.log(n_pos)

        X = sm.add_constant(log_n_pos)
        ols = sm.OLS(frac_exp, X).fit()
        slope = float(ols.params[1])
        ci_lo, ci_hi = ols.conf_int(alpha=0.05)[1]
        out["regression"] = {
            "outcome": "fraction_experimental_evidence (contamination-depth proxy)",
            "predictor": "log(n_positive) -- same quantity as the confirmatory sweep's own x-axis",
            "n_terms": len(results),
            "slope": slope,
            "slope_ci95": [float(ci_lo), float(ci_hi)],
            "p_value": float(ols.pvalues[1]),
            "r_squared": float(ols.rsquared),
            "mean_fraction_experimental": float(frac_exp.mean()),
        }
        logger.info(
            "Regression: slope=%.5f  95%% CI=[%.5f, %.5f]  p=%.4f  R^2=%.4f  n=%d",
            slope, ci_lo, ci_hi, ols.pvalues[1], ols.rsquared, len(results),
        )
    else:
        out["regression"] = None
        logger.warning("Fewer than 10 terms succeeded -- not enough to regress")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
