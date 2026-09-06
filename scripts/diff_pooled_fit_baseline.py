"""
Post-run reproducibility check for a re-run of scripts/fit_pooled_sweep.py.

Every pre-existing field in docs/experiment3_pooled_fit.json (the run
without --scale-param-counts) must come back byte-for-byte identical from a
re-run WITH --scale-param-counts, given the same random_state=42 and
n_bootstrap=2000 — the trend harvest is a dot product appended after the
existing rng.choice calls inside the bootstrap loop, so it does not perturb
the resample sequence (verified by reading _pooled_cluster_bootstrap_ci: the
only rng.choice calls happen before scale_trend_contrast is ever touched).
Any changed field means the /scratch sweep inputs moved since the baseline
was generated, not that the trend code is wrong — that ambiguity is exactly
why the re-run must never overwrite the baseline file in place (external
review, 2026-08).

Usage:
  python scripts/diff_pooled_fit_baseline.py \\
      docs/experiment3_pooled_fit.json \\
      docs/experiment3_pooled_fit_with_trend.json
"""

from __future__ import annotations

import json
import sys

PRE_EXISTING_FIELDS = [
    "reference_scale", "scales", "slope", "slope_se_cluster", "slope_ci_cluster",
    "slope_p_value_cluster", "slope_ci_bootstrap", "interaction_wald_stat",
    "interaction_p_value", "n_rows", "n_clusters_total", "n_clusters_per_scale",
    "n_bootstrap_successful",
]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} <baseline.json> <new_run.json>")

    baseline = json.loads(open(sys.argv[1]).read())
    new = json.loads(open(sys.argv[2]).read())

    mismatches = []
    for field in PRE_EXISTING_FIELDS:
        if field not in new:
            mismatches.append(f"{field}: MISSING from new run")
            continue
        if baseline[field] != new[field]:
            mismatches.append(f"{field}: baseline={baseline[field]!r} != new={new[field]!r}")

    if mismatches:
        print(f"REPRODUCIBILITY CHECK FAILED ({len(mismatches)} field(s) differ):")
        for m in mismatches:
            print(f"  - {m}")
        print(
            "\nThis means the /scratch sweep inputs changed since the baseline was "
            "generated (2026-07-24) — investigate before trusting any number in the "
            "new run, trend included."
        )
        sys.exit(1)

    print(f"All {len(PRE_EXISTING_FIELDS)} pre-existing fields match the baseline exactly.")
    print(f"scale_trend_slope       = {new.get('scale_trend_slope')}")
    print(f"scale_trend_ci_bootstrap = {new.get('scale_trend_ci_bootstrap')}")
    print(f"scale_trend_p_bootstrap  = {new.get('scale_trend_p_bootstrap')}")
    draws = new.get("scale_trend_bootstrap_draws")
    print(f"scale_trend_bootstrap_draws: {len(draws) if draws else 0} draws serialized")

    expected = -0.4690
    got = new.get("scale_trend_slope")
    if got is not None and abs(got - expected) > 0.001:
        if abs(got - (-0.4588)) < 0.001:
            print(
                f"\nWARNING: scale_trend_slope={got:.4f} matches the OLD equal-weight "
                f"bug's value (-0.4588), not the expected precision-weighted -0.4690 — "
                f"the estimand-mismatch fix did not take."
            )
        else:
            print(
                f"\nNOTE: scale_trend_slope={got:.4f} matches neither the predicted "
                f"-0.4690 nor the old bug's -0.4588 — investigate before quoting."
            )


if __name__ == "__main__":
    main()
