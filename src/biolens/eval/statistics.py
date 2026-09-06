"""
Confirmatory statistical model for the sample-size sweep: does the
verified-feature rate rise predictably with min_positives, as the
winner's-curse lemma predicts?

Fits `verified_rate ~ log(min_positives)` as the pre-specified logistic
regression, with CLUSTER-ROBUST inference on `feature_idx`: a single SAE
feature can be the top hit for multiple GO terms (feature 1862 hit 3,
feature 251 hit 5, per the existing verified-features registry), so
treating GO-term rows as independent observations understates variance
and overstates significance. Reports BOTH an analytic cluster-robust Wald
CI (via statsmodels' sandwich estimator) and a cluster (block) bootstrap
CI, since bootstrap confidence intervals are the primary reported result:
the bootstrap here resamples whole clusters (feature indices) with
replacement, not individual rows, which would silently reintroduce the
same non-independence problem the cluster-robust fit exists to fix.

Also provides multiplicity correction (apply_multiplicity_correction) for
families of per-locus/per-feature p-values tested in the same analysis —
e.g. the Geuvadis case study's per-locus ACME p-values. Real gap found by
external methodological review, 2026-08-05: no FDR/Bonferroni/Benjamini-
Hochberg correction existed anywhere in this codebase despite testing
multiple loci in the same run (26 in the current dataset, with hundreds
planned) — confirmed via a grep across src/scripts for "benjamini",
"bonferroni", and "fdr" returning zero hits before this was added.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import statsmodels.api as sm
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from biolens.eval.feature_inspection import VerifiedAnnotation

logger = logging.getLogger(__name__)

# Plausibility-weighted numeric target for the GLM: confirmed/spurious
# are the clean endpoints, plausible is scored as a genuine half-credit
# outcome rather than forced into either bucket. `unverifiable` has no
# numeric mapping — it means the evidence needed to score the claim at all
# doesn't exist, not that the claim is 50% true, so those rows are excluded
# from the fit entirely (see build_sweep_rows).
STATUS_TO_TARGET = {"confirmed": 1.0, "plausible": 0.5, "spurious": 0.0}


def build_sweep_rows(
    sweep_results: dict[int, list[dict]],
    verified: dict[int, VerifiedAnnotation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Join per-sweep-point go_probing_results.json rows against the verified-
    features registry to build (min_positives, verified_target, feature_idx)
    arrays for fit_verified_rate_sweep.

    Only includes a row when: (a) its `best_feature_idx` has a registry
    entry, (b) that entry's `claimed_go` list actually contains this row's
    `go_id` — i.e. this SPECIFIC claim was the one checked, not merely some
    other claim about the same feature — and (c) the status isn't
    "unverifiable" (no evidence to score against, excluded rather than
    guessed at). This is intentionally conservative: the confirmatory sweep
    fit should only run on ground truth that was actually checked for that
    exact (go_id, feature) pair, per the deliberate-sampling requirement.

    Args:
        sweep_results: min_positives -> list of go_probing_results.json rows
            (each a dict with at least "go_id" and "best_feature_idx"), one
            entry per sweep point (e.g. {10: [...], 20: [...], ...}).
        verified: feature_idx -> VerifiedAnnotation, from
            biolens.eval.feature_inspection.load_verified_annotations.

    Returns:
        (min_positives, verified_target, feature_idx) — three (n,) arrays,
        ready to pass directly to fit_verified_rate_sweep.
    """
    min_positives_out: list[int] = []
    target_out: list[float] = []
    feature_idx_out: list[int] = []

    for mp, rows in sweep_results.items():
        for row in rows:
            feat_idx = row["best_feature_idx"]
            annotation = verified.get(feat_idx)
            if annotation is None:
                continue
            if row["go_id"] not in annotation.claimed_go:
                continue
            target = STATUS_TO_TARGET.get(annotation.status)
            if target is None:
                continue  # "unverifiable" or an unrecognized status — excluded, not guessed
            min_positives_out.append(mp)
            target_out.append(target)
            feature_idx_out.append(feat_idx)

    return (
        np.array(min_positives_out, dtype=float),
        np.array(target_out, dtype=float),
        np.array(feature_idx_out),
    )


@dataclass
class SweepFitResult:
    """Fit of `verified_rate ~ log(min_positives)`, clustered on feature_idx."""

    intercept: float
    slope: float  # coefficient on log(min_positives) — the effect size of interest
    intercept_se_cluster: float
    slope_se_cluster: float
    slope_ci_cluster: tuple[float, float]  # analytic cluster-robust Wald CI
    slope_p_value_cluster: float
    slope_ci_bootstrap: tuple[float, float] | None  # cluster (block) bootstrap CI
    n_bootstrap_successful: int
    n_rows: int
    n_clusters: int  # distinct feature_idx values — the effective sample size
    # under clustering, which a naive power calculation on n_rows overstates.

    def predict(self, min_positives: float) -> float:
        """Predicted verified_rate at a given min_positives (logistic link)."""
        eta = self.intercept + self.slope * np.log(min_positives)
        return float(1.0 / (1.0 + np.exp(-eta)))

    def effect_per_doubling(self) -> float:
        """
        Change in verified_rate (in probability, evaluated at the model's own
        intercept-implied baseline of min_positives=1) associated with a
        doubling of min_positives — the "each doubling of min_positives is
        associated with an X-point change in verified rate" effect-size
        statement the plan calls for (Contribution 1), computed at the
        midpoint between predict(1) and predict(2) for a locally-linear
        approximation of the logistic curve's slope in probability units.
        """
        p_low = self.predict(1.0)
        p_high = self.predict(2.0)
        return p_high - p_low

    def __str__(self) -> str:
        lo, hi = self.slope_ci_cluster
        lines = [
            "verified_rate ~ log(min_positives), clustered on feature_idx",
            f"  n_rows={self.n_rows}  n_clusters={self.n_clusters} "
            f"(effective N for power/inference, not n_rows)",
            f"  slope={self.slope:.4f}  cluster-SE={self.slope_se_cluster:.4f}  "
            f"95% CI=[{lo:.4f}, {hi:.4f}]  p={self.slope_p_value_cluster:.4g}",
        ]
        if self.slope_ci_bootstrap is not None:
            blo, bhi = self.slope_ci_bootstrap
            lines.append(
                f"  cluster-bootstrap 95% CI=[{blo:.4f}, {bhi:.4f}] "
                f"({self.n_bootstrap_successful} successful resamples)"
            )
        lines.append(
            f"  effect per doubling of min_positives: "
            f"{self.effect_per_doubling():+.4f} verified_rate"
        )
        return "\n".join(lines)


def fit_verified_rate_sweep(
    min_positives: np.ndarray,
    verified: np.ndarray,
    feature_idx: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> SweepFitResult:
    """
    Fit the confirmatory sweep model for the sample-size sweep.

    Args:
        min_positives: (n_rows,) the min_positives threshold each row was
            probed at (or, for a per-GO-term dataset, the actual n_positives
            of that GO term).
        verified: (n_rows,) outcome in [0, 1] — 1/0 for confirmed-vs-not, or
            a continuous plausibility-weighted score (e.g. confirmed=1,
            plausible=0.5, spurious=0). GLM Binomial with non-binary [0,1]
            targets is a standard "proportion" formulation.
        feature_idx: (n_rows,) the SAE feature index each row's best hit was
            — the clustering variable. A feature that's the top hit for
            multiple GO terms contributes multiple rows sharing one cluster
            id here.
        n_bootstrap: Number of cluster-bootstrap resamples.
        confidence: Confidence level for both CIs (default 95%).
        random_state: Bootstrap RNG seed.

    Returns:
        SweepFitResult with the analytic cluster-robust fit and (if enough
        clusters exist to resample meaningfully) a cluster bootstrap CI.
    """
    min_positives = np.asarray(min_positives, dtype=float)
    verified = np.asarray(verified, dtype=float)
    feature_idx = np.asarray(feature_idx)
    n_rows = len(min_positives)
    if not (len(verified) == n_rows and len(feature_idx) == n_rows):
        raise ValueError("min_positives, verified, and feature_idx must be the same length")
    if n_rows == 0:
        raise ValueError("Cannot fit a sweep model with zero rows")

    unique_clusters = np.unique(feature_idx)
    n_clusters = len(unique_clusters)

    intercept, slope, se_intercept, se_slope, ci, p_value = _fit_glm_cluster(
        min_positives, verified, feature_idx, confidence
    )

    ci_bootstrap = None
    n_successful = 0
    if n_bootstrap <= 0:
        # Explicitly skipped (e.g. scripts/power_calculation.py's Monte Carlo
        # power sweep, which only needs the analytic cluster-robust CI and
        # would otherwise trigger a misleading "0/0 resamples converged"
        # warning from _cluster_bootstrap_ci on every one of thousands of
        # simulated fits) — not the same as "too few clusters," which gets
        # its own warning below since that case is a real data limitation.
        pass
    elif n_clusters >= 10:  # too few clusters to bootstrap meaningfully otherwise
        ci_bootstrap, n_successful = _cluster_bootstrap_ci(
            min_positives, verified, feature_idx, unique_clusters,
            n_bootstrap=n_bootstrap, confidence=confidence, random_state=random_state,
        )
    else:
        logger.warning(
            "Only %d distinct feature_idx clusters — skipping cluster bootstrap "
            "(need >= 10 for a meaningful resampling distribution); analytic "
            "cluster-robust CI is still reported.",
            n_clusters,
        )

    return SweepFitResult(
        intercept=intercept,
        slope=slope,
        intercept_se_cluster=se_intercept,
        slope_se_cluster=se_slope,
        slope_ci_cluster=ci,
        slope_p_value_cluster=p_value,
        slope_ci_bootstrap=ci_bootstrap,
        n_bootstrap_successful=n_successful,
        n_rows=n_rows,
        n_clusters=n_clusters,
    )


def _fit_glm_cluster(
    min_positives: np.ndarray,
    verified: np.ndarray,
    feature_idx: np.ndarray,
    confidence: float,
) -> tuple[float, float, float, float, tuple[float, float], float]:
    """Fit the GLM Binomial model with cluster-robust ('sandwich') standard
    errors via statsmodels — the standard tool for this, not a hand-rolled
    sandwich-variance implementation."""
    X = sm.add_constant(np.log(min_positives))
    model = sm.GLM(verified, X, family=sm.families.Binomial())
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": feature_idx})

    intercept, slope = float(fit.params[0]), float(fit.params[1])
    se_intercept, se_slope = float(fit.bse[0]), float(fit.bse[1])
    alpha = 1.0 - confidence
    conf = fit.conf_int(alpha=alpha)
    ci = (float(conf[1][0]), float(conf[1][1]))
    p_value = float(fit.pvalues[1])
    return intercept, slope, se_intercept, se_slope, ci, p_value


@dataclass
class InflationBootstrapResult:
    """Cluster-bootstrap CI for a mean auroc_inflation-style summary
    statistic (e.g. mean_auroc_inflation) — a different, simpler quantity
    than SweepFitResult's verified_rate ~ log(min_positives) model, but
    with the same non-independence concern: a single SAE feature/latent
    can be the best-hit for multiple GO terms/concepts, so naively
    bootstrapping individual rows understates the true variance of the
    mean. The real-data AUROC-inflation-vs-n curve is what makes "is this
    decay real or noise" answerable rather than asserted from a bare point
    estimate."""

    mean: float
    ci: tuple[float, float]
    n_bootstrap_successful: int
    n_rows: int
    n_clusters: int

    def __str__(self) -> str:
        lo, hi = self.ci
        return (
            f"mean_auroc_inflation={self.mean:.4f}  "
            f"95% cluster-bootstrap CI=[{lo:.4f}, {hi:.4f}]  "
            f"(n_rows={self.n_rows}, n_clusters={self.n_clusters}, "
            f"{self.n_bootstrap_successful} successful resamples)"
        )


def bootstrap_mean_auroc_inflation(
    auroc_inflation: np.ndarray,
    feature_idx: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> InflationBootstrapResult:
    """
    Cluster (block) bootstrap CI for a mean auroc_inflation value, clustering
    on feature_idx for the identical reason fit_verified_rate_sweep clusters
    on it — a feature that's the top hit for multiple GO terms/concepts
    contributes multiple non-independent rows, and resampling rows directly
    (an ordinary bootstrap) would treat them as independent, silently
    understating the interval's true width.

    Args:
        auroc_inflation: (n_rows,) per-GO-term/per-concept auroc_inflation
            values (naive_max_auroc - held_out_auroc), e.g. from
            go_probing_results.json at a single min_positives value.
        feature_idx: (n_rows,) each row's best_feature_idx — the clustering
            variable, same role as in fit_verified_rate_sweep.
        n_bootstrap: Number of cluster-bootstrap resamples.
        confidence: Confidence level (default 95%).
        random_state: Bootstrap RNG seed.

    Returns:
        InflationBootstrapResult with the point estimate and cluster-
        bootstrap CI. If fewer than 10 distinct clusters exist, the CI is
        still computed (unlike fit_verified_rate_sweep's stricter cutoff,
        a mean's bootstrap distribution degrades more gracefully than a
        GLM's does with few clusters) but a warning is logged.
    """
    auroc_inflation = np.asarray(auroc_inflation, dtype=float)
    feature_idx = np.asarray(feature_idx)
    n_rows = len(auroc_inflation)
    if len(feature_idx) != n_rows:
        raise ValueError("auroc_inflation and feature_idx must be the same length")
    if n_rows == 0:
        raise ValueError("Cannot bootstrap a mean with zero rows")

    unique_clusters = np.unique(feature_idx)
    n_clusters = len(unique_clusters)
    if n_clusters < 10:
        logger.warning(
            "Only %d distinct feature_idx clusters for mean_auroc_inflation "
            "bootstrap — CI may be unreliable (fit_verified_rate_sweep skips "
            "the bootstrap entirely below this threshold; a mean's resampling "
            "distribution degrades more gracefully, so it's still computed "
            "here, but treat a wide or unstable interval as expected, not a bug).",
            n_clusters,
        )

    rows_by_cluster = {c: np.where(feature_idx == c)[0] for c in unique_clusters}
    rng = np.random.default_rng(random_state)

    means = []
    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in sampled_clusters])
        means.append(float(np.mean(auroc_inflation[idx])))

    alpha = 1.0 - confidence
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return InflationBootstrapResult(
        mean=float(np.mean(auroc_inflation)),
        ci=(float(lo), float(hi)),
        n_bootstrap_successful=len(means),
        n_rows=n_rows,
        n_clusters=n_clusters,
    )


@dataclass
class RandomEffectsMetaResult:
    """
    DerSimonian-Laird random-effects meta-analysis of k per-model slope
    estimates (e.g. the four ESM2 scales' independent log-log AUROC/
    permuted-inflation-vs-n fits) into one pooled estimate.

    This replaces naively pooling the k models' raw ROWS into one regression
    (what `fit_auroc_inflation_curve.py`/`fit_permuted_inflation_curve.py`
    call the "pooled fit") — that approach implicitly weights each model by
    its row count, which is an artifact of how many `min_positives` points
    happened to be usable for that model, not a property of the underlying
    effect. Caught via external review, 2026-08: the real-label ("Route 1")
    row-level pooled slope (-0.722, CI excluding the theoretical -0.5) is the
    only estimator across every reasonable pooling choice that excludes the
    theory — a genuine artifact of row-count weighting, not evidence against
    the lemma's prediction.

    Standard random-effects meta-analysis instead treats each model's own
    slope as one "study": weight by inverse variance PLUS between-study
    heterogeneity (tau^2, estimated via the classical DerSimonian & Laird
    1986 method-of-moments estimator from Cochran's Q), so a model with a
    noisier fit contributes less, and residual disagreement between models
    widens the pooled CI rather than being silently absorbed into a
    row-count-weighted average.

    `ci` (normal-quantile, theta_re +/- z_.975*se_re) is kept for reference,
    but `ci_kh` (Knapp & Hartung 2003 small-sample correction) is the
    interval to actually REPORT. At k=4 studies, DL's tau^2 is itself
    estimated with substantial noise and the z-quantile ignores that
    extra uncertainty entirely, which undercovers once tau^2 > 0 — a
    40k-replicate Monte Carlo on this project's own Route-1 SE vector
    (external review, 2026-08, independently reproduced) found DL+z 95% CI
    coverage drops to ~90% at Route 1's own estimated tau^2=0.0226 (vs.
    ~94% for DL+Knapp-Hartung); at tau^2=0 (Route 2's regime) the two are
    close (~96% vs ~95%). Knapp-Hartung is the standard recommendation for
    meta-analyses with k<10 studies (Knapp & Hartung 2003; IntHout,
    Ioannidis & Borm 2014) for exactly this reason. `q_kh` (the weighted
    dispersion of the studies around theta_re) is FLOORED at 1.0 before
    scaling se_re — a real bug caught by external audit, 2026-08: without
    the floor, q_kh<1 (studies agreeing more than their own variances
    predict) NARROWS se_kh below se_re, the opposite of Knapp-Hartung's
    entire purpose. With the floor, `ci_kh` is always at least as wide as
    `ci` (t_{k-1}'s quantile exceeds the normal quantile at any finite df,
    and se_kh >= se_re unconditionally). Separately: state explicitly in
    any write-up that theta_re == theta_fixed_effect whenever
    tau_squared == 0, so a reader doesn't mistake it for a copy-paste error.
    """

    theta_re: float  # pooled (random-effects) slope estimate
    se_re: float
    ci: tuple[float, float]  # normal-quantile 95% CI, theta_re +/- z_.975*se_re -- reference only
    se_kh: float  # Knapp-Hartung-adjusted SE
    ci_kh: tuple[float, float]  # Knapp-Hartung 95% CI (t_{k-1} quantile) -- REPORT THIS ONE
    theta_fixed_effect: float  # inverse-variance-only pooled estimate (tau^2=0), for reference
    tau_squared: float  # between-study heterogeneity variance
    q_statistic: float  # Cochran's Q (homogeneity test statistic)
    q_p_value: float  # H0: all k studies share one true slope
    i_squared: float  # % of total variance attributable to heterogeneity, not sampling error
    k: int  # number of studies (models) pooled
    study_ses: list[float]  # each study's own SE, in input order

    def __str__(self) -> str:
        return (
            f"Random-effects pooled slope: {self.theta_re:.4f}  "
            f"95% CI (Knapp-Hartung) [{self.ci_kh[0]:.4f}, {self.ci_kh[1]:.4f}]  "
            f"(normal-quantile reference: [{self.ci[0]:.4f}, {self.ci[1]:.4f}])\n"
            f"  k={self.k} studies  tau^2={self.tau_squared:.5f}  "
            f"Q={self.q_statistic:.4f} (p={self.q_p_value:.4f})  "
            f"I^2={self.i_squared:.1f}%\n"
            f"  fixed-effect (no heterogeneity) reference: {self.theta_fixed_effect:.4f}"
        )


def fit_random_effects_meta(
    thetas: list[float] | np.ndarray, ses: list[float] | np.ndarray
) -> RandomEffectsMetaResult:
    """
    DerSimonian & Laird (1986) random-effects meta-analysis: pool k
    independent slope estimates (`thetas`) with known standard errors
    (`ses`) into one estimate that accounts for between-study heterogeneity,
    not just each study's own sampling variance.

    Args:
        thetas: k independent point estimates (e.g. one per model scale).
        ses: k standard errors, one per estimate, same order as thetas.

    Returns:
        RandomEffectsMetaResult with the pooled estimate, its Knapp-Hartung
        CI (`ci_kh` — report this one) and normal-quantile CI (`ci` —
        reference only, undercovers at small k once tau^2>0), and the
        heterogeneity diagnostics (Q, I^2, tau^2).
    """
    thetas = np.asarray(thetas, dtype=float)
    ses = np.asarray(ses, dtype=float)
    k = len(thetas)
    if len(ses) != k:
        raise ValueError(f"thetas and ses must be the same length, got {k} and {len(ses)}")
    if k < 2:
        raise ValueError(f"Random-effects meta-analysis needs at least 2 studies, got {k}")
    if np.any(ses <= 0):
        raise ValueError("All standard errors must be positive")

    w = 1.0 / ses**2
    theta_fixed = float(np.sum(w * thetas) / np.sum(w))
    q_statistic = float(np.sum(w * (thetas - theta_fixed) ** 2))

    # DerSimonian-Laird method-of-moments tau^2 estimator, floored at 0 (a
    # negative method-of-moments estimate means the data are consistent with
    # zero heterogeneity, not literally negative heterogeneity).
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau_squared = max(0.0, (q_statistic - (k - 1)) / c) if c > 0 else 0.0

    w_star = 1.0 / (ses**2 + tau_squared)
    theta_re = float(np.sum(w_star * thetas) / np.sum(w_star))
    se_re = float(np.sqrt(1.0 / np.sum(w_star)))
    z_crit = float(scipy_stats.norm.ppf(0.975))
    ci = (theta_re - z_crit * se_re, theta_re + z_crit * se_re)

    # Knapp & Hartung (2003) small-sample correction: rescale se_re by the
    # weighted dispersion of the studies around theta_re (q_kh), then use a
    # t_{k-1} quantile instead of a normal one -- both pieces are needed
    # together (see class docstring for the coverage simulation motivating
    # this; external review, 2026-08, independently reproduced).
    #
    # q_kh is FLOORED at 1.0 (the modified KH of Knapp & Hartung 2003 /
    # IntHout, Ioannidis & Borm 2014, the standard recommendation) -- a real
    # bug caught by external codebase audit, 2026-08: the unfloored formula
    # lets q_kh<1 when the studies happen to agree more than their own
    # variances predict, which NARROWS se_kh below se_re. That is the
    # opposite of Knapp-Hartung's entire purpose (a conservative correction
    # for tau^2's own estimation uncertainty at small k) -- an unfloored KH
    # interval can end up narrower than the plain normal-quantile one,
    # which no standard reference recommends. Confirmed this bit the real
    # results: Route 2's original (unfloored) se_kh=0.038 was smaller than
    # se_re=0.050 (q_kh=0.575), silently narrowing the reported CI instead
    # of widening it.
    q_kh = float(np.sum(w_star * (thetas - theta_re) ** 2) / (k - 1))
    se_kh = se_re * math.sqrt(max(q_kh, 1.0))
    t_crit = float(scipy_stats.t.ppf(0.975, k - 1))
    ci_kh = (theta_re - t_crit * se_kh, theta_re + t_crit * se_kh)

    i_squared = max(0.0, (q_statistic - (k - 1)) / q_statistic * 100) if q_statistic > 0 else 0.0
    q_p_value = float(1.0 - scipy_stats.chi2.cdf(q_statistic, k - 1))

    return RandomEffectsMetaResult(
        theta_re=theta_re,
        se_re=se_re,
        ci=ci,
        se_kh=se_kh,
        ci_kh=ci_kh,
        theta_fixed_effect=theta_fixed,
        tau_squared=tau_squared,
        q_statistic=q_statistic,
        q_p_value=q_p_value,
        i_squared=i_squared,
        k=k,
        study_ses=ses.tolist(),
    )


@dataclass
class PooledScaleFitResult:
    """
    Pooled fit of `verified_rate ~ log(min_positives) * model_scale` across
    multiple models' confirmatory sweeps in a single regression, clustered on
    (model_scale, feature_idx) jointly.

    This replaces informally comparing N separate per-model SweepFitResults
    (e.g. "8M's CI excludes 0, 35M's doesn't") — comparing significance
    across studies with unequal power that way is a known statistical
    mistake (the difference between significant and not significant is not
    itself significant), and is exactly the failure mode the GWAS winner's-
    curse replication-variability literature attributes anomalous replication
    patterns to. A single pooled model with a scale interaction term
    answers "does the slope
    actually differ by scale" directly, via one joint test, and naturally
    accommodates each model contributing a different number of clusters —
    no matched sample size required.

    feature_idx values are only meaningful WITHIN one model's own trained
    SAE (see feature_inspection.load_verified_annotations) — the same raw
    integer can label unrelated features across different models, so the
    clustering variable here is the (model_scale, feature_idx) pair, never
    feature_idx alone.

    Methodological precedent for each piece of this design (verified against
    real literature, not asserted from memory):
      - The core justification: Gelman, A. & Stern, H. (2006). "The
        Difference Between 'Significant' and 'Not Significant' is not
        Itself Statistically Significant." The American Statistician,
        60(4), 328-331. States the exact failure mode this replaces: one
        subgroup coefficient can be significant while another isn't, even
        when the two don't differ significantly from each other — "the
        correct test is the interaction."
      - The pooled-interaction-model itself is the modern implementation of
        the classical Chow test (Chow, 1960) for whether regression
        coefficients differ across groups: pooling the data, fitting a
        fully-interacted model, and jointly testing the interaction
        coefficients is mathematically equivalent to comparing separate
        per-group regressions via the traditional Chow F-statistic.
      - Cluster-robust ("sandwich") standard errors: Cameron, A.C. & Miller,
        D.L. (2015). "A Practitioner's Guide to Cluster-Robust Inference."
        Journal of Human Resources, 50(2), 317-372 — the standard reference
        for the same technique fit_verified_rate_sweep already uses for a
        single model.
      - Stratifying the cluster bootstrap by scale (never resampling a
        cluster from one model into another model's slot): standard
        practice for bootstrapping non-exchangeable strata — stratified
        designs bootstrap within each stratum, preserving each stratum's
        own cluster count across every resample.
    """

    reference_scale: str
    scales: list[str]  # reference_scale first, then the rest in fit order
    slope: dict[str, float]  # marginal slope estimate per scale
    slope_se_cluster: dict[str, float]
    slope_ci_cluster: dict[str, tuple[float, float]]
    slope_p_value_cluster: dict[str, float]  # H0: this scale's own slope == 0
    slope_ci_bootstrap: dict[str, tuple[float, float]] | None
    interaction_wald_stat: float
    interaction_p_value: float  # H0: all scales share the same slope (the real cross-model test)
    n_rows: int
    n_clusters_total: int
    n_clusters_per_scale: dict[str, int]
    n_bootstrap_successful: int
    # Cross-scale trend (e.g. "does the slope decline with model size?") — only
    # populated when `scale_param_counts` is passed to fit_pooled_scale_interaction.
    # NOT a naive regression of the four per-scale `slope` values against each
    # other's independence: those four estimates are linear contrasts sharing
    # the same reference-scale coefficient from ONE pooled GLM
    # (_extract_per_scale_slopes), so they are correlated, not independent —
    # treating them as four independent points and running an ordinary
    # weighted regression (with an analytic Wald or Knapp-Hartung SE) silently
    # assumes away that correlation. Caught during external review (2026-08),
    # confirmed against the real design: `_extract_per_scale_slopes` builds
    # every non-reference slope as `fit.params[1] + interaction_i` via a
    # linear contrast on one fit, so all four literally share fit.params[1].
    # The fix used here: harvest the trend statistic INSIDE the existing
    # stratified cluster bootstrap (_pooled_cluster_bootstrap_ci) instead of
    # computing it once from the four point estimates — each bootstrap
    # replicate's four per-scale slopes carry the same correlation structure
    # the real estimator has (since they come from one resampled GLM fit,
    # exactly as the real four do), so the spread of the per-replicate trend
    # statistic across replicates is a valid, correlation-correct SE with no
    # equicorrelation assumption needed. Both `scale_trend_slope` (the point
    # estimate) and every bootstrap draw use the SAME fixed, precision-
    # weighted linear contrast, computed once from the full-sample SEs — an
    # earlier version used equal weights for the bootstrap draws only,
    # reasoning that fixed weights were needed to avoid double-counting the
    # resampling uncertainty, which is true, but equal weights aren't the
    # only way to hold weights fixed, and using them silently computed the
    # bootstrap distribution of a DIFFERENT (unweighted) quantity than the
    # reported point estimate — caught via external review, 2026-08, by
    # checking that the full-sample value of the bootstrap statistic agreed
    # with the point estimate (it didn't, by ~2%). See
    # _pooled_cluster_bootstrap_ci's `scale_trend_contrast` docstring for
    # the corrected design.
    scale_trend_slope: float | None = None  # per log10-unit of scale_param_counts
    scale_trend_ci_bootstrap: tuple[float, float] | None = None
    scale_trend_p_bootstrap: float | None = None
    # Raw per-replicate trend draws (one per successful bootstrap resample).
    # Exposed directly — not just summarized as CI/p — so downstream code
    # (diagnostic plots, or a test asserting the draw mean agrees with
    # scale_trend_slope to real statistical precision rather than a crude
    # CI-width-based approximation) can use the actual distribution rather
    # than reconstructing it. None iff scale_trend_slope is None.
    scale_trend_bootstrap_draws: list[float] | None = None

    def __str__(self) -> str:
        lines = [
            "verified_rate ~ log(min_positives) * model_scale, "
            f"clustered on (model_scale, feature_idx), reference={self.reference_scale}",
            f"  n_rows={self.n_rows}  n_clusters_total={self.n_clusters_total} "
            f"({', '.join(f'{s}={n}' for s, n in self.n_clusters_per_scale.items())})",
            f"  interaction test (H0: slope is the same across all scales): "
            f"chi2={self.interaction_wald_stat:.4f}  p={self.interaction_p_value:.4g}",
        ]
        for s in self.scales:
            lo, hi = self.slope_ci_cluster[s]
            line = (
                f"  [{s}] slope={self.slope[s]:.4f}  SE={self.slope_se_cluster[s]:.4f}  "
                f"95% CI=[{lo:.4f}, {hi:.4f}]  p={self.slope_p_value_cluster[s]:.4g}"
            )
            if self.slope_ci_bootstrap is not None:
                blo, bhi = self.slope_ci_bootstrap[s]
                line += f"  bootstrap CI=[{blo:.4f}, {bhi:.4f}]"
            lines.append(line)
        if self.scale_trend_slope is not None:
            tlo, thi = self.scale_trend_ci_bootstrap
            lines.append(
                f"  cross-scale trend (per log10-unit): {self.scale_trend_slope:.4f}  "
                f"bootstrap 95% CI=[{tlo:.4f}, {thi:.4f}]  "
                f"bootstrap p={self.scale_trend_p_bootstrap:.4g} "
                f"(correlation-correct — see class docstring)"
            )
        return "\n".join(lines)


def build_pooled_sweep_rows(
    per_model_sweep_results: dict[str, dict[int, list[dict]]],
    per_model_verified: dict[str, dict[int, VerifiedAnnotation]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build pooled (min_positives, verified_target, feature_idx, model_scale)
    arrays across multiple models, reusing build_sweep_rows per model.

    Args:
        per_model_sweep_results: model_scale label (e.g. "esm2_8m") -> that
            model's build_sweep_rows-style sweep_results dict (min_positives
            -> list of go_probing_results.json rows).
        per_model_verified: model_scale label -> that model's verified-
            features registry (from load_verified_annotations).

    Returns:
        (min_positives, verified_target, feature_idx, model_scale) — four
        (n,) arrays, ready for fit_pooled_scale_interaction. feature_idx is
        only meaningful paired with its corresponding model_scale entry.
    """
    mp_all, target_all, feat_all, scale_all = [], [], [], []
    for scale, sweep_results in per_model_sweep_results.items():
        verified = per_model_verified[scale]
        mp, target, feat = build_sweep_rows(sweep_results, verified)
        mp_all.append(mp)
        target_all.append(target)
        feat_all.append(feat)
        scale_all.append(np.full(len(mp), scale, dtype=object))

    return (
        np.concatenate(mp_all),
        np.concatenate(target_all),
        np.concatenate(feat_all),
        np.concatenate(scale_all),
    )


def fit_pooled_scale_interaction(
    min_positives: np.ndarray,
    verified: np.ndarray,
    feature_idx: np.ndarray,
    model_scale: np.ndarray,
    reference_scale: str | None = None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
    scale_param_counts: dict[str, float] | None = None,
) -> PooledScaleFitResult:
    """
    Fit `verified_rate ~ log(min_positives) * model_scale` in one pooled
    regression, clustered on (model_scale, feature_idx).

    Args:
        min_positives, verified, feature_idx: same meaning and shape
            contract as fit_verified_rate_sweep, but concatenated across
            every model being compared.
        scale_param_counts: optional {scale: parameter count} (e.g.
            {"esm2_8m": 8e6, ...}) — if given, also fits a cross-scale trend
            (does the slope trend with model size?) as
            `scale_trend_slope`/`scale_trend_ci_bootstrap`/
            `scale_trend_p_bootstrap` on the result, using log10(count) as
            the moderator. See PooledScaleFitResult's docstring for why this
            is computed via the stratified cluster bootstrap rather than an
            ordinary regression on the four per-scale point estimates (they
            are correlated, not independent — an external-review-caught
            error, 2026-08).
        model_scale: (n_rows,) a label per row (e.g. "esm2_8m") identifying
            which model's sweep/registry that row came from — the grouping
            variable for the interaction term. Requires at least 2 distinct
            values.
        reference_scale: which scale's slope is the "base" `log(min_positives)`
            coefficient that other scales' interaction terms are offsets
            from. Purely a parameterization choice — every scale still gets
            its own marginal slope/CI/p-value in the result. Defaults to the
            first scale in sorted order.
        n_bootstrap: Number of cluster-bootstrap resamples (stratified by
            scale — each resample redraws each scale's own clusters with
            replacement, preserving that scale's cluster count, since
            clusters from different models aren't exchangeable with each
            other).
        confidence: Confidence level for all CIs.
        random_state: Bootstrap RNG seed.

    Returns:
        PooledScaleFitResult with the joint interaction test and each
        scale's own marginal slope/CI/p-value.
    """
    min_positives = np.asarray(min_positives, dtype=float)
    verified = np.asarray(verified, dtype=float)
    feature_idx = np.asarray(feature_idx)
    model_scale = np.asarray(model_scale, dtype=object)
    n_rows = len(min_positives)
    if not (len(verified) == n_rows and len(feature_idx) == n_rows and len(model_scale) == n_rows):
        raise ValueError(
            "min_positives, verified, feature_idx, and model_scale must be the same length"
        )
    if n_rows == 0:
        raise ValueError("Cannot fit a pooled sweep model with zero rows")

    scales = sorted(set(model_scale.tolist()))
    if len(scales) < 2:
        raise ValueError(f"Pooled fit needs at least 2 distinct model scales, got {scales}")
    if reference_scale is None:
        reference_scale = scales[0]
    elif reference_scale not in scales:
        raise ValueError(f"reference_scale={reference_scale!r} not among observed scales {scales}")
    other_scales = [s for s in scales if s != reference_scale]
    ordered_scales = [reference_scale] + other_scales
    n_dummies = len(other_scales)

    # Feature indices are only meaningful within one model's own SAE — the
    # cluster id must combine scale and feature_idx so rows from different
    # models never get pooled into the same cluster just because they share
    # a raw integer index.
    cluster_id = np.array([f"{s}::{f}" for s, f in zip(model_scale, feature_idx)])

    X, interaction_start = _build_pooled_design(min_positives, model_scale, other_scales)
    n_params = X.shape[1]

    fit = sm.GLM(verified, X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_id}
    )

    alpha = 1.0 - confidence
    slope, se, ci, pval = _extract_per_scale_slopes(
        fit, reference_scale, other_scales, interaction_start, alpha
    )

    # Joint Wald test: are ALL interaction coefficients simultaneously zero?
    # (i.e. does any scale's slope actually differ from the reference's)
    r_matrix = np.zeros((n_dummies, n_params))
    for i in range(n_dummies):
        r_matrix[i, interaction_start + i] = 1.0
    wald = fit.wald_test(r_matrix, scalar=True)
    interaction_stat = float(np.asarray(wald.statistic).ravel()[0])
    interaction_p = float(np.asarray(wald.pvalue).ravel()[0])

    n_clusters_per_scale = {
        s: len(set(feature_idx[model_scale == s].tolist())) for s in ordered_scales
    }
    n_clusters_total = len(set(cluster_id.tolist()))

    # Cross-scale trend: a FIXED linear contrast, computed once from the
    # full-sample precision weights, applied identically to every bootstrap
    # replicate below. Computed here (before the bootstrap call) because the
    # contrast needs `slope`/`se`, which are already available at this point
    # — and because the bootstrap must reuse the SAME weights, not
    # re-estimate them per replicate (see _pooled_cluster_bootstrap_ci's
    # docstring for why re-estimating per replicate would double-count the
    # uncertainty, and why using EQUAL weights instead — an earlier, real
    # bug in this exact function, caught via external review 2026-08 by
    # noticing the point estimate and the bootstrap draws' full-sample value
    # disagreed by 2% — is not the same thing as "don't re-estimate").
    scale_trend_contrast: dict[str, float] | None = None
    scale_trend_slope = None
    if scale_param_counts is not None:
        missing = set(ordered_scales) - set(scale_param_counts)
        if missing:
            raise ValueError(f"scale_param_counts is missing entries for {missing}")
        if any(c <= 0 for c in scale_param_counts.values()):
            raise ValueError("scale_param_counts values must be positive (log10 is taken)")
        xw = np.array([math.log10(scale_param_counts[s]) for s in ordered_scales])
        ww = np.array([1.0 / se[s] ** 2 for s in ordered_scales])
        xbar_w = float(np.sum(ww * xw) / np.sum(ww))
        a_vec = ww * (xw - xbar_w) / np.sum(ww * (xw - xbar_w) ** 2)
        scale_trend_contrast = {s: float(a_vec[i]) for i, s in enumerate(ordered_scales)}
        # sum(a) == 0 exactly (weighted-mean centering), so a . y == a . (y -
        # ybar) for ANY constant ybar — y-centering is mathematically
        # unnecessary here, not an omission.
        scale_trend_slope = float(
            np.sum(a_vec * np.array([slope[s] for s in ordered_scales]))
        )

    ci_bootstrap = None
    n_successful = 0
    trend_draws = None
    if n_bootstrap > 0:
        if all(n >= 10 for n in n_clusters_per_scale.values()):
            ci_bootstrap, n_successful, trend_draws = _pooled_cluster_bootstrap_ci(
                min_positives, verified, feature_idx, model_scale,
                reference_scale, other_scales,
                n_bootstrap=n_bootstrap, confidence=confidence, random_state=random_state,
                scale_trend_contrast=scale_trend_contrast,
            )
        else:
            logger.warning(
                "At least one scale has < 10 distinct feature_idx clusters (%s) — "
                "skipping pooled cluster bootstrap; analytic cluster-robust CIs "
                "are still reported.",
                n_clusters_per_scale,
            )

    scale_trend_ci_bootstrap = None
    scale_trend_p_bootstrap = None
    if trend_draws:
        alpha = 1.0 - confidence
        lo, hi = np.percentile(trend_draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        scale_trend_ci_bootstrap = (float(lo), float(hi))
        draws_arr = np.asarray(trend_draws)
        frac_le = float(np.mean(draws_arr <= 0))
        frac_ge = float(np.mean(draws_arr >= 0))
        p_raw = 2.0 * min(frac_le, frac_ge)
        # A bootstrap p-value of exactly 0.0 (no draw crossed zero, the
        # likely outcome for a strong effect at typical n_bootstrap) is not
        # reportable as-is — floor it at the resolution the resample count
        # can actually support, i.e. report "p < 1/n_draws" rather than
        # a false "p = 0". Caught via external review, 2026-08.
        scale_trend_p_bootstrap = min(1.0, max(p_raw, 1.0 / len(draws_arr)))

    return PooledScaleFitResult(
        reference_scale=reference_scale,
        scales=ordered_scales,
        slope=slope,
        slope_se_cluster=se,
        slope_ci_cluster=ci,
        slope_p_value_cluster=pval,
        slope_ci_bootstrap=ci_bootstrap,
        interaction_wald_stat=interaction_stat,
        interaction_p_value=interaction_p,
        n_rows=n_rows,
        n_clusters_total=n_clusters_total,
        n_clusters_per_scale=n_clusters_per_scale,
        n_bootstrap_successful=n_successful,
        scale_trend_slope=scale_trend_slope,
        scale_trend_ci_bootstrap=scale_trend_ci_bootstrap,
        scale_trend_p_bootstrap=scale_trend_p_bootstrap,
        scale_trend_bootstrap_draws=trend_draws,
    )


def _build_pooled_design(
    min_positives: np.ndarray, model_scale: np.ndarray, other_scales: list[str]
) -> tuple[np.ndarray, int]:
    """
    Build the design matrix for the pooled scale-interaction GLM.

    Column layout: [intercept, log(min_positives), scale-dummies for each
    non-reference scale, log(min_positives)*dummy interaction for each
    non-reference scale]. Returns (X, interaction_start_column_index).
    """
    log_mp = np.log(min_positives)
    n_dummies = len(other_scales)
    cols = [np.ones(len(min_positives)), log_mp]
    dummies = [(model_scale == s).astype(float) for s in other_scales]
    cols.extend(dummies)
    cols.extend(d * log_mp for d in dummies)
    X = np.column_stack(cols)
    interaction_start = 2 + n_dummies
    return X, interaction_start


def _extract_per_scale_slopes(
    fit, reference_scale: str, other_scales: list[str], interaction_start: int, alpha: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float]], dict[str, float]]:
    """Pull each scale's marginal slope out of a fitted pooled-interaction
    GLM: the reference scale's slope is the log(min_positives) coefficient
    directly; every other scale's slope is that same coefficient plus its
    own interaction offset, extracted via a linear contrast (fit.t_test) so
    its standard error correctly accounts for the covariance between the two
    terms rather than naively summing independent SEs."""
    slope: dict[str, float] = {}
    se: dict[str, float] = {}
    ci: dict[str, tuple[float, float]] = {}
    pval: dict[str, float] = {}

    n_params = len(fit.params)
    conf = fit.conf_int(alpha=alpha)
    slope[reference_scale] = float(fit.params[1])
    se[reference_scale] = float(fit.bse[1])
    ci[reference_scale] = (float(conf[1][0]), float(conf[1][1]))
    pval[reference_scale] = float(fit.pvalues[1])

    for i, s in enumerate(other_scales):
        r = np.zeros(n_params)
        r[1] = 1.0
        r[interaction_start + i] = 1.0
        tt = fit.t_test(r)
        slope[s] = float(np.asarray(tt.effect).ravel()[0])
        se[s] = float(np.asarray(tt.sd).ravel()[0])
        lo, hi = np.asarray(tt.conf_int(alpha=alpha)).reshape(-1)
        ci[s] = (float(lo), float(hi))
        pval[s] = float(np.asarray(tt.pvalue).ravel()[0])

        # A near-degenerate design (e.g. near-perfect separation) can make
        # statsmodels' cluster-robust sandwich covariance numerically
        # negative on the diagonal for this contrast -- sqrt(negative)
        # silently yields a NaN SE (and downstream NaN CI/p-value) rather
        # than raising. Caught via external codebase audit, 2026-08-27: no
        # real pooled fit in this project has ever hit this (all three
        # real outputs checked, all finite), but a future run on a
        # genuinely degenerate design could silently publish a NaN
        # interval instead of failing loudly.
        if math.isnan(se[s]):
            raise ValueError(
                f"Cluster-robust SE for scale {s!r} is NaN -- the sandwich covariance matrix "
                f"is numerically negative on this contrast, likely from a near-degenerate design "
                f"(e.g. near-perfect separation in that scale's data). Not a reportable result."
            )

    return slope, se, ci, pval


def _pooled_cluster_bootstrap_ci(
    min_positives: np.ndarray,
    verified: np.ndarray,
    feature_idx: np.ndarray,
    model_scale: np.ndarray,
    reference_scale: str,
    other_scales: list[str],
    n_bootstrap: int,
    confidence: float,
    random_state: int,
    scale_trend_contrast: dict[str, float] | None = None,
) -> tuple[dict[str, tuple[float, float]], int, list[float] | None]:
    """
    Stratified cluster bootstrap for the pooled model: each resample redraws
    every scale's own clusters independently (with replacement, same count
    as that scale actually has), then refits the full pooled-interaction GLM
    on the concatenated resample and extracts every scale's marginal slope.
    Stratifying by scale (rather than pooling all clusters into one draw
    pool) is required because clusters from different models are never
    exchangeable with each other — feature 42 in esm2_8m and feature 42 in
    esm2_650m are unrelated features, not repeated measurements of the same
    unit.

    Args:
        scale_trend_contrast: optional {scale: fixed linear-contrast weight}
            — if given, also harvests a per-replicate cross-scale TREND
            statistic (does the slope trend with some moderator, e.g. model
            size?) as `sum(scale_trend_contrast[s] * replicate_slope[s] for
            s in all_scales)` on each replicate's own four per-scale slopes.

            This is deliberately NOT a second regression run on the four
            already-computed point estimates: those four are linear
            contrasts sharing the reference scale's coefficient from a
            single GLM fit (see _extract_per_scale_slopes and
            PooledScaleFitResult's docstring), hence correlated, and an
            ordinary regression across them (with an analytic Wald or
            Knapp-Hartung SE) silently assumes independence that isn't
            there. Here, each bootstrap replicate's four slopes come from
            ONE resampled GLM fit exactly as the real four do, so they carry
            the same correlation structure — the spread of the per-replicate
            trend value across replicates is therefore a valid,
            correlation-correct standard error with no assumption about the
            correlation's sign or magnitude needed.

            The contrast weights themselves must be FIXED across every
            replicate (computed once, by the caller, from the full-sample
            precision weights) rather than re-estimated per replicate or
            set to equal weights. An earlier version of this function used
            equal weights inside the loop, reasoning that "the resampling
            IS the uncertainty draw, so re-weighting would double-count
            it" — that conflates two different things: avoiding
            double-counting requires the weights be held FIXED across
            replicates, not that they be EQUAL. Using equal weights instead
            of the real precision weights silently computes the bootstrap
            distribution of a DIFFERENT quantity than the reported point
            estimate (an unweighted trend, not the precision-weighted one)
            — caught via external review, 2026-08, by checking that the
            full-sample value of the bootstrap statistic actually agreed
            with the separately-computed point estimate (it didn't, by
            ~2%). Since the weights are fixed, `sum(scale_trend_contrast
            values) == 0` by construction (weighted-mean centering), so no
            y-centering is needed in the per-replicate dot product below.

    Returns:
        (per_scale_ci, n_successful, trend_draws) — trend_draws is None iff
        scale_trend_contrast is None, else the list of per-replicate trend
        values (one per successful resample).
    """
    rng = np.random.default_rng(random_state)
    all_scales = [reference_scale] + other_scales
    trend_draws: list[float] | None = [] if scale_trend_contrast is not None else None
    if scale_trend_contrast is not None:
        trend_a = np.array([scale_trend_contrast[s] for s in all_scales])

    rows_by_scale_cluster: dict[str, dict] = {}
    clusters_by_scale: dict[str, np.ndarray] = {}
    for s in all_scales:
        mask = model_scale == s
        feats = feature_idx[mask]
        idx_in_full = np.where(mask)[0]
        unique_c = np.unique(feats)
        clusters_by_scale[s] = unique_c
        rows_by_scale_cluster[s] = {c: idx_in_full[feats == c] for c in unique_c}

    per_scale_slopes: dict[str, list[float]] = {s: [] for s in all_scales}
    n_successful = 0

    for _ in range(n_bootstrap):
        idx_parts = []
        for s in all_scales:
            unique_c = clusters_by_scale[s]
            sampled = rng.choice(unique_c, size=len(unique_c), replace=True)
            idx_parts.append(np.concatenate([rows_by_scale_cluster[s][c] for c in sampled]))
        idx = np.concatenate(idx_parts)

        y = verified[idx]
        if y.min() == y.max():
            continue  # degenerate resample, GLM undefined

        X, interaction_start = _build_pooled_design(
            min_positives[idx], model_scale[idx], other_scales
        )
        try:
            resample_fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        except Exception as exc:  # noqa: BLE001 — a failed resample must not abort the whole CI
            logger.debug("Pooled cluster-bootstrap resample failed to fit: %s", exc)
            continue

        per_scale_slopes[reference_scale].append(float(resample_fit.params[1]))
        for i, s in enumerate(other_scales):
            per_scale_slopes[s].append(
                float(resample_fit.params[1] + resample_fit.params[interaction_start + i])
            )
        n_successful += 1

        if trend_draws is not None:
            slopes_r = np.array([per_scale_slopes[s][-1] for s in all_scales])
            trend_draws.append(float(np.sum(trend_a * slopes_r)))

    if n_successful < 50:
        logger.warning(
            "Only %d/%d pooled cluster-bootstrap resamples converged — CIs may be unreliable",
            n_successful, n_bootstrap,
        )

    alpha = 1.0 - confidence
    ci_out: dict[str, tuple[float, float]] = {}
    for s in all_scales:
        vals = per_scale_slopes[s]
        if not vals:
            ci_out[s] = (float("nan"), float("nan"))
            continue
        lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci_out[s] = (float(lo), float(hi))

    return ci_out, n_successful, trend_draws


def _cluster_bootstrap_ci(
    min_positives: np.ndarray,
    verified: np.ndarray,
    feature_idx: np.ndarray,
    unique_clusters: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    random_state: int,
) -> tuple[tuple[float, float], int]:
    """
    Block (cluster) bootstrap: resample whole feature_idx clusters with
    replacement, refit the (unclustered — the resampling itself is what
    propagates the clustering into the interval) GLM on each resample, and
    take the percentile interval of the resulting slope distribution.

    Resampling CLUSTERS rather than individual rows is what makes this a
    valid cluster bootstrap — resampling rows directly would treat GO-term
    observations as independent, exactly the assumption being corrected for.
    """
    rng = np.random.default_rng(random_state)
    rows_by_cluster = {c: np.where(feature_idx == c)[0] for c in unique_clusters}

    slopes = []
    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in sampled_clusters])
        y = verified[idx]
        if y.min() == y.max():
            continue  # degenerate resample (no outcome variance) — GLM undefined, skip
        X = sm.add_constant(np.log(min_positives[idx]))
        try:
            fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            slopes.append(float(fit.params[1]))
        except Exception as exc:  # noqa: BLE001 — a failed resample must not abort the whole CI
            logger.debug("Cluster-bootstrap resample failed to fit: %s", exc)
            continue

    if len(slopes) < 50:
        logger.warning(
            "Only %d/%d cluster-bootstrap resamples converged — CI may be unreliable",
            len(slopes), n_bootstrap,
        )
    if not slopes:
        return (float("nan"), float("nan")), 0

    alpha = 1.0 - confidence
    lo, hi = np.percentile(slopes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi)), len(slopes)


@dataclass
class MultiplicityCorrectionResult:
    raw_p_values: np.ndarray = field(repr=False)
    adjusted_p_values: np.ndarray = field(repr=False)
    rejected: np.ndarray = field(repr=False)  # bool array, True where adjusted p < alpha
    method: str
    alpha: float
    n_tests: int
    n_significant: int

    def __str__(self) -> str:
        return (
            f"{self.method} correction across {self.n_tests} tests at alpha={self.alpha}: "
            f"{self.n_significant}/{self.n_tests} remain significant after adjustment"
        )


def apply_multiplicity_correction(
    p_values: np.ndarray | list[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> MultiplicityCorrectionResult:
    """
    Multiplicity correction across a family of p-values from testing multiple
    loci/features in the same analysis — e.g. the Geuvadis case study's
    per-locus ACME p-values (real gap found by external methodological
    review, 2026-08-05: no FDR/Bonferroni correction existed anywhere in this
    codebase despite testing 26 loci in the current dataset, with hundreds
    planned — an obvious gap to a geneticist reviewing this code).

    Uses statsmodels.stats.multitest.multipletests directly rather than
    hand-rolling Benjamini-Hochberg — this is squarely "Layer 1: tried and
    true," not something worth reimplementing.

    Args:
        p_values: Raw, uncorrected p-values across every test in the family
            (e.g. every locus's naive-arm ACME p-value from one Geuvadis
            run). All tests in a single call must belong to the same
            hypothesis family — see biolens.data.geuvadis's call site for
            why the naive and verified arms are corrected as two SEPARATE
            families rather than pooled into one.
        method: Passed through to statsmodels' multipletests — "fdr_bh"
            (Benjamini-Hochberg, default) is the standard choice for a
            genomics-style many-loci screen (controls the false discovery
            rate, not the family-wise error rate, which is the field
            convention here, not Bonferroni's stricter FWER control).
        alpha: Significance threshold applied AFTER correction.

    Returns:
        MultiplicityCorrectionResult with adjusted p-values and a boolean
        rejection mask, same order and length as the input.

    Raises:
        ValueError: if p_values is empty — nothing to correct, and silently
            returning an empty result would be a confusing way to surface
            that upstream (e.g. every locus in a run was skipped).
    """
    from statsmodels.stats.multitest import multipletests

    p_values = np.asarray(p_values, dtype=float)
    if len(p_values) == 0:
        raise ValueError("Cannot apply multiplicity correction to zero p-values")

    rejected, adjusted, _, _ = multipletests(p_values, alpha=alpha, method=method)

    return MultiplicityCorrectionResult(
        raw_p_values=p_values,
        adjusted_p_values=adjusted,
        rejected=rejected,
        method=method,
        alpha=alpha,
        n_tests=len(p_values),
        n_significant=int(rejected.sum()),
    )


@dataclass
class TostEquivalenceResult:
    mean: float
    se: float
    n: int
    margin: float
    z_lower: float
    z_upper: float
    p_lower: float
    p_upper: float
    p_value: float  # binding (larger) one-sided p-value, per TOST convention
    ci95: tuple[float, float]
    equivalent_at_alpha: dict[float, bool] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"TOST equivalence to zero at +/-{self.margin:.2%} margin: "
            f"mean={self.mean:.4f}, se={self.se:.4f}, n={self.n}, "
            f"z={min(self.z_lower, self.z_upper):.3f}, p={self.p_value:.2e}, "
            f"95% CI=[{self.ci95[0]:.4f}, {self.ci95[1]:.4f}]"
        )


def tost_equivalence_test(
    values: np.ndarray | list[float], margin: float, alpha: float = 0.05
) -> TostEquivalenceResult:
    """
    Two one-sided test (TOST) of whether the mean of `values` is
    statistically equivalent to zero within +/-`margin` — e.g. the Geuvadis
    case study's claim that the average naive-arm mediated fraction is
    equivalence-bounded near zero.

    Real gap found by external review, 2026-08-29: this number appeared in
    write-ups of the Geuvadis case study results with no corresponding
    function anywhere in the codebase — a grep for "tost" and "equivalence"
    across every tracked and untracked .py file returned zero hits before
    this was added. The number itself independently reproduced exactly
    (z=3.787->3.79, p=7.61e-05->0.00008) against the 18
    reportable-`proportion_mediated` loci in the Geuvadis case study
    results, using the plain i.i.d. standard error below (the 18 loci
    map to 18 distinct genes — no repeated-gene clustering to correct for,
    confirmed by direct check) — but "reproduces once by hand" is exactly
    the failure mode this project's own verification discipline argues
    against tolerating, so it is now real, tested code rather than a number
    that only ever existed in prose.

    Two one-sided tests against the margin:
      H0: mean <= -margin  vs  H1: mean > -margin   (z_lower, p_lower)
      H0: mean >= +margin  vs  H1: mean < +margin   (z_upper, p_upper)
    Equivalence is established only if BOTH null hypotheses are rejected, so
    the reported p-value is the larger (binding) of the two one-sided
    p-values, per standard TOST convention.

    Args:
        values: Per-unit estimates whose mean is being tested for
            equivalence to zero (e.g. per-locus proportion-mediated
            estimates, already filtered to the reportable subset upstream —
            this function does not know about `proportion_mediated_
            reportable` gating and assumes the caller has already applied
            it).
        margin: The equivalence margin, as a fraction (e.g. 0.10 for +/-10
            percentage points). Must be positive.
        alpha: Significance level for the `equivalent_at_alpha` convenience
            flag (checked at both 0.05 and this value).

    Returns:
        TostEquivalenceResult with both one-sided z/p-values, the binding
        p-value, the plain-normal 95% CI on the mean, and equivalence flags
        at alpha=0.05 and the passed `alpha`.

    Raises:
        ValueError: if fewer than 2 values are passed (SE undefined), or if
            margin <= 0.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError(
            f"tost_equivalence_test needs at least 2 values to estimate a "
            f"standard error, got {len(values)}"
        )
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")

    n = len(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n))

    z_lower = (mean - (-margin)) / se
    z_upper = (margin - mean) / se
    p_lower = float(1 - scipy_stats.norm.cdf(z_lower))
    p_upper = float(1 - scipy_stats.norm.cdf(z_upper))
    p_value = max(p_lower, p_upper)

    z95 = scipy_stats.norm.ppf(0.975)
    ci95 = (mean - z95 * se, mean + z95 * se)

    equivalent_at_alpha = {
        0.05: p_value < 0.05,
        alpha: p_value < alpha,
    }

    return TostEquivalenceResult(
        mean=mean,
        se=se,
        n=n,
        margin=margin,
        z_lower=z_lower,
        z_upper=z_upper,
        p_lower=p_lower,
        p_upper=p_upper,
        p_value=p_value,
        ci95=ci95,
        equivalent_at_alpha=equivalent_at_alpha,
    )


def hanley_mcneil_auroc_variance(auc: float, n_positive: int, n_negative: int) -> float:
    """
    Hanley & McNeil (1982) closed-form variance of the empirical AUROC
    estimator, used by the winner's-curse lemma to connect per-feature
    AUROC noise `sigma` to sample size.

        Var(AUC) = [AUC(1-AUC) + (n_pos-1)(Q1-AUC^2) + (n_neg-1)(Q2-AUC^2)]
                   / (n_pos * n_neg)

    where Q1 = AUC/(2-AUC) and Q2 = 2*AUC^2/(1+AUC) are the probabilities
    that a randomly chosen positive (resp. two positives) ranks above two
    randomly chosen negatives (resp. one negative) — the standard
    derivation via the Mann-Whitney U statistic's variance under the
    assumption that AUC is itself the estimator's true value (i.e. treating
    the plug-in AUC as the population parameter, the same approximation the
    original 1982 paper and all standard uses of this formula make).

    Args:
        auc: The (true or estimated) AUROC value to evaluate the variance
            at — for the winner's-curse null case this is 0.5.
        n_positive, n_negative: Sample sizes of the positive and negative
            classes.

    Returns:
        The variance of the AUROC estimator (not its standard deviation).
    """
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    return (
        auc * (1 - auc)
        + (n_positive - 1) * (q1 - auc**2)
        + (n_negative - 1) * (q2 - auc**2)
    ) / (n_positive * n_negative)


def gaussian_max_inflation_prediction(sigma: float, d: int, folded: bool = True) -> float:
    """
    Refined Gaussian order-statistics prediction for the winner's-curse
    lemma's bias term, `E[max_i eps_i]`, for `d` i.i.d. mean-zero noise
    draws each with standard deviation `sigma`. Bias magnitude grows with
    estimator variance.

    CORRECTED 2026-08-27 (external codebase audit, independently re-derived
    and verified before adopting): the previous version of this function
    used the crude leading-order asymptotic `sigma*sqrt(2 ln d)`, and — for
    `folded=True` — rescaled `sigma` by the folded-normal SD factor
    `sqrt(1-2/pi)`. That rescaling is a category error: extreme-value
    scale is governed by TAIL behavior, not by the overall variance of the
    folded variable. `biolens.eval.probing._vectorized_auroc` reflects via
    `max(A, 1-A)`, i.e. computes `max_i |eps_i|` over `d` draws — and
    `max_i |eps_i| = max` over the `2d` SIGNED values `{eps_1,-eps_1,...,
    eps_d,-eps_d}` exactly, as a matter of arithmetic (`|x|=max(x,-x)`).
    A half-normal's tail is exactly twice a normal's tail at the same `y`
    (`P(|X|>y) = 2*P(X>y)`), and doubling the tail probability at every
    point is, to leading order, equivalent to doubling the effective
    sample count in the max-over-iid-Gaussians asymptotic — NOT to
    shrinking sigma. Reflection therefore corresponds to evaluating the
    (unreflected) max-of-n-Gaussians prediction at `n=2d`, not to rescaling
    sigma by a folded-normal factor.

    Separately, the crude leading-order asymptotic `sigma*sqrt(2 ln n)` is
    known to converge slowly and underestimates the true expected maximum
    at any practically-sized `n` — this function now uses the standard
    refined (Gumbel-mean) correction instead (see e.g. the classical
    normalizing-constant refinement for Gaussian maxima): with
    `a_n = sqrt(2 ln n)` and `b_n = a_n - (ln(ln n) + ln(4*pi)) / (2*a_n)`,
    `E[max of n iid N(0,1)] ~ b_n + gamma/a_n` (`gamma` = Euler-Mascheroni
    constant), i.e. the Gumbel distribution's mean under the classical
    normalizing constants, not just its location parameter `b_n` alone.

    Combining both corrections and checking against a 40k-replicate Monte
    Carlo simulation of `max_i |eps_i|` (matching what `_vectorized_auroc`
    actually computes) gives near-exact agreement (ratio ~0.99-1.00) at
    realistic (sigma, d) values from the winner's-curse simulation's own
    sweep — a real, substantive improvement over the previous folded
    formula, which
    UNDERESTIMATED true inflation by roughly 35% (confirmed: repo folded
    0.218 vs. simulated 0.338 at n_positive=10, d=2560).

    Args:
        sigma: Per-draw standard deviation (e.g. from
            hanley_mcneil_auroc_variance).
        d: Number of candidates the max is taken over.
        folded: If True (default), the noise draws have been REFLECTED via
            max(A, 1-A) before the max-over-candidates selection step
            (exactly what _vectorized_auroc does) — evaluates the
            refined asymptotic at `n=2d`. Set to False for a selection
            statistic that is NOT reflected (n=d).

    Returns:
        The predicted inflation magnitude (same units as `sigma`).
    """
    n = 2 * d if folded else d
    a_n = math.sqrt(2 * math.log(n))
    b_n = a_n - (math.log(math.log(n)) + math.log(4 * math.pi)) / (2 * a_n)
    euler_gamma = 0.5772156649015329
    return sigma * (b_n + euler_gamma / a_n)


# Ordinal encoding for weighted Cohen's kappa on the three "on-scale" status
# categories: they are quasi-ordinal, confirmed > plausible > spurious.
# `unverifiable` deliberately has NO entry here —
# it is not "less than spurious," it means the evidence needed to judge the
# claim at all doesn't exist, so it does not belong on this ordinal scale
# and is handled separately (see compute_agreement).
_STATUS_ORDINAL = {"spurious": 0, "plausible": 1, "confirmed": 2}
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class AgreementResult:
    n_items: int
    percent_agreement: float
    n_unverifiable_by_either: int
    percent_agreement_on_scale: float | None  # excludes items where either said unverifiable
    weighted_kappa: float | None  # None if <2 distinct on-scale categories present to compare
    n_resolved: int
    n_excluded_tied_disagreement: int

    def __str__(self) -> str:
        kappa_str = f"{self.weighted_kappa:.3f}" if self.weighted_kappa is not None else "N/A"
        on_scale_str = (
            f"{self.percent_agreement_on_scale:.1%}"
            if self.percent_agreement_on_scale is not None else "N/A"
        )
        return (
            f"n={self.n_items}, percent_agreement={self.percent_agreement:.1%} "
            f"(on-scale only: {on_scale_str}), weighted_kappa={kappa_str}, "
            f"{self.n_unverifiable_by_either} item(s) involved 'unverifiable', "
            f"{self.n_excluded_tied_disagreement} excluded from resolved consensus "
            f"(tied-confidence disagreement)"
        )


def resolve_consensus(
    label_a: str, confidence_a: str, label_b: str, confidence_b: str,
) -> tuple[str | None, str]:
    """
    Resolve one item's two-annotator disagreement into a single "resolved
    consensus" label, per the pre-registered protocol (written before any
    annotation happened): higher stated confidence wins;
    when confidence is TIED and the labels still disagree, no one
    unilaterally breaks the tie (that would reintroduce exactly the
    non-independence problem two annotators exist to solve) — the item is
    excluded from the resolved-consensus set instead, with its own rate
    reported separately as an informative number in its own right ("how
    often does the task admit two defensible answers?").

    The written protocol states this explicitly only for the both-`high`
    case; extended here to ANY tied confidence level (both `medium`, both
    `low`, etc.) for the identical stated reason — having one party
    unilaterally decide is exactly as much of a non-independence problem at
    any tied confidence level, not just at `high`.

    Returns:
        (resolved_label, reason) — resolved_label is None when excluded.
    """
    if label_a == label_b:
        return label_a, "agreement"
    rank_a, rank_b = _CONFIDENCE_RANK[confidence_a], _CONFIDENCE_RANK[confidence_b]
    if rank_a > rank_b:
        return label_a, f"annotator A's higher confidence ({confidence_a} > {confidence_b}) breaks the tie"
    if rank_b > rank_a:
        return label_b, f"annotator B's higher confidence ({confidence_b} > {confidence_a}) breaks the tie"
    return None, (
        f"tied confidence ({confidence_a}) disagreement ({label_a} vs {label_b}) — "
        f"excluded from resolved consensus, not unilaterally broken"
    )


def compute_agreement(
    labels_a: list[str], confidences_a: list[str],
    labels_b: list[str], confidences_b: list[str],
) -> AgreementResult:
    """
    Full inter-annotator agreement report for the shared overlap subset.
    Both percent agreement AND weighted Cohen's kappa are required, not
    one or the other: raw percent agreement can be inflated by a degenerate
    strategy
    like always answering the modal category, especially given this
    registry's skewed label distribution; kappa corrects for chance
    agreement but is itself only meaningful on the three genuinely ordinal
    categories, hence computed on-scale-only, separately from the
    `unverifiable` handling below).

    Args:
        labels_a/labels_b: Status calls ("confirmed"/"plausible"/
            "spurious"/"unverifiable"), same order, same items.
        confidences_a/confidences_b: Matching confidence levels
            ("high"/"medium"/"low").

    Returns:
        AgreementResult with both agreement metrics and the resolved-
        consensus bookkeeping from resolve_consensus.
    """
    from sklearn.metrics import cohen_kappa_score

    n = len(labels_a)
    if not (len(labels_b) == len(confidences_a) == len(confidences_b) == n):
        raise ValueError("labels_a, confidences_a, labels_b, confidences_b must be the same length")
    if n == 0:
        raise ValueError("Cannot compute agreement over zero items")

    percent_agreement = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    on_scale_pairs = [
        (a, b) for a, b in zip(labels_a, labels_b)
        if a in _STATUS_ORDINAL and b in _STATUS_ORDINAL
    ]
    n_unverifiable = n - len(on_scale_pairs)

    if on_scale_pairs:
        on_scale_agreement = sum(a == b for a, b in on_scale_pairs) / len(on_scale_pairs)
        ord_a = [_STATUS_ORDINAL[a] for a, _ in on_scale_pairs]
        ord_b = [_STATUS_ORDINAL[b] for _, b in on_scale_pairs]
        # cohen_kappa_score is undefined (and sklearn errors or returns NaN)
        # when fewer than 2 distinct categories appear across both raters —
        # e.g. a tiny on-scale subset that happens to be all "spurious".
        if len(set(ord_a) | set(ord_b)) >= 2:
            weighted_kappa = float(cohen_kappa_score(ord_a, ord_b, weights="linear"))
        else:
            weighted_kappa = None
            logger.warning(
                "Only one distinct on-scale category present across both annotators "
                "(%d on-scale items) — weighted kappa is undefined, not computed",
                len(on_scale_pairs),
            )
    else:
        on_scale_agreement = None
        weighted_kappa = None

    n_resolved = 0
    n_excluded = 0
    for la, ca, lb, cb in zip(labels_a, confidences_a, labels_b, confidences_b):
        resolved, _ = resolve_consensus(la, ca, lb, cb)
        if resolved is None:
            n_excluded += 1
        else:
            n_resolved += 1

    return AgreementResult(
        n_items=n,
        percent_agreement=percent_agreement,
        n_unverifiable_by_either=n_unverifiable,
        percent_agreement_on_scale=on_scale_agreement,
        weighted_kappa=weighted_kappa,
        n_resolved=n_resolved,
        n_excluded_tied_disagreement=n_excluded,
    )
