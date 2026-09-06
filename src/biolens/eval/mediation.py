"""
Imai-Keele-Tingley (2010) causal mediation analysis for the Geuvadis case
study: does verified-vs-naive SAE feature selection change the
causal-mediation conclusion for a DNA regulatory variant's effect on gene
expression?

  Treatment (T): genotype dosage at a variant (0/1/2 copies of the alt allele)
  Mediator  (M): Evo 2 SAE feature-activation shift between ref/alt allele
                 sequence context (continuous)
  Outcome   (Y): real RNA-seq expression (continuous)

Uses statsmodels.stats.mediation.Mediation — the standard, tested Python
implementation of the IKT simulation-based mediation algorithm (the same
approach R's `mediation` package implements) — rather than hand-rolling the
ACME/ADE simulation loop. This is NOT classic Baron-Kenny: IKT's simulation
approach correctly generalizes beyond simple product-of-coefficients and is
the field-standard framework for this.

The sensitivity analysis (how much unmeasured confounding, i.e. sequential-
ignorability violation, would it take to null out the ACME) is NOT built
into statsmodels' Mediation class, so it's implemented here directly, using
the closed-form result for the continuous-mediator/continuous-outcome
(linear-linear) case from Imai, Keele & Yamamoto (2010), "Identification,
Inference, and Sensitivity Analysis for Causal Mediation Effects,"
Statistical Science: under jointly normal mediator/outcome equation errors
with correlation rho (induced by an unmeasured confounder of the
mediator-outcome relationship), the true ACME is

    ACME(rho) = beta * (delta_hat - rho * sigma_2_hat / (sigma_1_hat * sqrt(1 - rho^2)))

where beta is the treatment effect on the mediator, delta_hat is the naive
(rho=0) mediator-on-outcome coefficient, and sigma_1_hat/sigma_2_hat are the
residual standard deviations of the NAIVELY-FITTED (rho=0) mediator/outcome
models. The sqrt(1-rho^2) term is real, not optional — an earlier version of
this function omitted it (real bug, caught by external methodological
review, 2026-08; confirmed via Monte Carlo, N=2,000,000: the naive delta_hat
has probability limit `gamma + rho*sigma_2/sigma_1` using the TRUE sigma_2,
but the naively-fitted sigma_2_hat is itself attenuated under confounding to
sigma_2*sqrt(1-rho^2) — so correcting delta_hat with the observed,
attenuated sigma_2_hat directly (no sqrt term) systematically understates
the bias correction at every nonzero rho, making the reported
robustness-to-confounding threshold rho* too large — i.e. falsely
overstating robustness). At rho=0 this reduces to the standard
product-of-coefficients ACME (beta * delta_hat), which is the same quantity
statsmodels' Mediation class estimates by simulation — see test_mediation.py
for a direct cross-check between the two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation

logger = logging.getLogger(__name__)


@dataclass
class MediationResult:
    acme_estimate: float
    acme_ci: tuple[float, float]
    acme_p_value: float
    ade_estimate: float
    ade_ci: tuple[float, float]
    ade_p_value: float
    total_effect: float
    total_effect_ci: tuple[float, float]
    total_effect_p_value: float
    proportion_mediated: float
    mediator_std: float
    n_rep: int
    n_obs: int

    @property
    def acme_reportable(self) -> bool:
        """
        True iff the mediator has nonzero variance in this sample.

        When the mediator is constant (e.g. an SAE feature that never
        activates anywhere near this locus's haplotype window, so every
        sample gets the same activation-shift value), the IKT bootstrap's
        simulated ACME distribution collapses to a point mass at exactly
        zero. statsmodels' two-sided p-value, computed as
        2*min(P(sim<=0), P(sim>=0)), then degenerates to 2*min(0,0) = 0
        (neither "<=0" nor ">=0" holds in the strict sense the formula
        uses when every simulated value equals exactly 0) -- reporting
        "p=0, highly significant" for an estimate that is, by
        construction, untestable. Zero variance means zero information
        about mediation, not overwhelming evidence for it.

        Caught empirically in the Geuvadis case study (job 9904287,
        2026-08-13): 7/26 loci showed this exact signature -- ACME CI
        == [0, 0], ADE == total effect to full float precision (the
        mediator contributed nothing), p=0 -- before being reported as
        genuine findings. ADE and total effect remain valid and
        unaffected: T still has real variance, so those estimates and
        their p-values are legitimate regardless of the mediator's
        degeneracy.
        """
        return self.mediator_std > 1e-8

    @property
    def proportion_mediated_reportable(self) -> bool:
        """
        True iff the total effect's 95% CI excludes zero.

        proportion_mediated = ACME / total_effect is a ratio estimator, and
        ratio estimators are badly behaved near a zero denominator (a
        Fieller-type instability, not a small-sample artifact fixable by
        more data — flagged by external methodological review, 2026-08).
        ACME and ADE are the primary, always-interpretable estimates;
        proportion_mediated should only be reported/interpreted for loci
        where the total effect is itself distinguishable from zero, since
        otherwise the ratio's denominator could plausibly BE zero and the
        reported value carries no reliable information regardless of its
        magnitude.
        """
        lo, hi = self.total_effect_ci
        return not (lo <= 0.0 <= hi)

    def __str__(self) -> str:
        ade_lo, ade_hi = self.ade_ci
        te_lo, te_hi = self.total_effect_ci
        if self.acme_reportable:
            lo, hi = self.acme_ci
            acme_part = (
                f"ACME={self.acme_estimate:.4f} (95% CI [{lo:.4f}, {hi:.4f}], "
                f"p={self.acme_p_value:.4g}), "
            )
        else:
            acme_part = "ACME=N/A (mediator has zero variance -- untestable), "
        s = (
            f"{acme_part}ADE={self.ade_estimate:.4f} "
            f"(95% CI [{ade_lo:.4f}, {ade_hi:.4f}], p={self.ade_p_value:.4g}), "
            f"total={self.total_effect:.4f} (95% CI [{te_lo:.4f}, {te_hi:.4f}], "
            f"p={self.total_effect_p_value:.4g})"
        )
        if self.proportion_mediated_reportable:
            s += f", prop. mediated={self.proportion_mediated:.3f}"
        else:
            s += ", prop. mediated=N/A (total effect CI includes zero -- ratio undefined/unstable)"
        return s + f" (n={self.n_obs}, n_rep={self.n_rep})"


def run_ikt_mediation(
    treatment: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
    covariates: pd.DataFrame | None = None,
    n_rep: int = 1000,
    random_state: int = 42,
) -> MediationResult:
    """
    Run IKT mediation analysis with continuous mediator and continuous
    outcome (both OLS) — matching this project's use case (SAE feature-
    activation shift as mediator, RNA-seq expression as outcome).

    Args:
        treatment:  (n,) genotype dosage (0/1/2) or other treatment values.
        mediator:   (n,) SAE feature-activation shift (or any continuous
                    mediator).
        outcome:    (n,) real expression values (or any continuous outcome).
        covariates: Optional (n, k) DataFrame of additional covariates
                    (e.g. genotype PCs, PEER factors) included in both the
                    mediator and outcome regressions.
        n_rep:      Bootstrap simulation repetitions (mediation_n_bootstrap=1000
                    in configs/eval.yaml).
        random_state: For reproducibility of the simulation-based CI.

    Returns:
        MediationResult with ACME, ADE, total effect, proportion mediated,
        each with a bootstrap 95% CI and p-value.
    """
    n = len(treatment)
    if not (len(mediator) == n and len(outcome) == n):
        raise ValueError("treatment, mediator, and outcome must be the same length")

    data = pd.DataFrame({"T": treatment, "M": mediator, "Y": outcome})
    covariate_cols: list[str] = []
    if covariates is not None:
        for col in covariates.columns:
            data[col] = covariates[col].to_numpy()
            covariate_cols.append(col)

    mediator_formula = "M ~ T" + ("".join(f" + {c}" for c in covariate_cols))
    outcome_formula = "Y ~ T + M" + ("".join(f" + {c}" for c in covariate_cols))

    np.random.seed(random_state)  # statsmodels' Mediation uses the global numpy RNG
    mediator_model = sm.OLS.from_formula(mediator_formula, data)
    outcome_model = sm.OLS.from_formula(outcome_formula, data)

    med = Mediation(outcome_model, mediator_model, exposure="T", mediator="M").fit(
        method="parametric", n_rep=n_rep
    )
    summary = med.summary()

    return MediationResult(
        acme_estimate=float(summary.loc["ACME (average)", "Estimate"]),
        acme_ci=(
            float(summary.loc["ACME (average)", "Lower CI bound"]),
            float(summary.loc["ACME (average)", "Upper CI bound"]),
        ),
        acme_p_value=float(summary.loc["ACME (average)", "P-value"]),
        ade_estimate=float(summary.loc["ADE (average)", "Estimate"]),
        ade_ci=(
            float(summary.loc["ADE (average)", "Lower CI bound"]),
            float(summary.loc["ADE (average)", "Upper CI bound"]),
        ),
        ade_p_value=float(summary.loc["ADE (average)", "P-value"]),
        total_effect=float(summary.loc["Total effect", "Estimate"]),
        total_effect_ci=(
            float(summary.loc["Total effect", "Lower CI bound"]),
            float(summary.loc["Total effect", "Upper CI bound"]),
        ),
        total_effect_p_value=float(summary.loc["Total effect", "P-value"]),
        proportion_mediated=float(summary.loc["Prop. mediated (average)", "Estimate"]),
        mediator_std=float(np.std(mediator)),
        n_rep=n_rep,
        n_obs=n,
    )


def run_negative_control(
    treatment: np.ndarray,
    outcome: np.ndarray,
    covariates: pd.DataFrame | None = None,
    n_rep: int = 1000,
    random_state: int = 42,
) -> MediationResult:
    """
    Negative-control arm: run the identical IKT pipeline on a
    randomly-selected, uninformative feature and confirm it does NOT find a
    significant mediation effect. Standard causal-inference practice,
    preempting "how do we know your pipeline doesn't just always find
    something significant."

    Uses a mediator drawn as pure Gaussian noise, independent of both
    treatment and outcome by construction — the ACME should not be
    significantly different from zero.
    """
    rng = np.random.default_rng(random_state)
    fake_mediator = rng.standard_normal(len(treatment))
    return run_ikt_mediation(
        treatment, fake_mediator, outcome, covariates=covariates,
        n_rep=n_rep, random_state=random_state,
    )


@dataclass
class PositiveControlResult:
    true_proportion_mediated: float
    recovered_proportion_mediated: float
    recovered_acme: float
    recovered_ade: float
    recovered_total_effect: float

    def __str__(self) -> str:
        return (
            f"true prop. mediated={self.true_proportion_mediated:.3f}  "
            f"recovered={self.recovered_proportion_mediated:.3f}  "
            f"(ACME={self.recovered_acme:.4f}, ADE={self.recovered_ade:.4f}, "
            f"total={self.recovered_total_effect:.4f})"
        )


def run_positive_control(
    true_proportion_mediated: float,
    n: int = 445,
    total_effect: float = 2.0,
    zeta: float = 1.0,
    noise_sd: float = 0.5,
    n_rep: int = 1000,
    random_state: int = 42,
) -> PositiveControlResult:
    """
    Construct synthetic data with a KNOWN, exact true mediation fraction and
    run it through the actual production mediation pipeline
    (run_ikt_mediation) to check the recovered proportion_mediated matches
    the known truth — the single cheapest, most effective safeguard against
    a real class of bug this project's causal design had (found by external
    methodological review, 2026-08): a mediator built as a deterministic
    rescaling of the treatment, `M = Delta * T`, makes the outcome
    regression's design matrix rank-deficient (M and T are then perfectly
    collinear), so the mediation effect and the direct effect are not
    separably identified — two datasets with true mediation fractions of
    100% and 0% produced statistically indistinguishable results under that
    construction (confirmed empirically). Had this positive control existed
    beforehand, it would have caught that failure immediately: recovered
    proportion_mediated near zero for a dataset engineered to be 100%
    mediated is a direct, unmissable signal something is wrong, independent
    of any real biological data.

    Critically, unlike the degenerate `M = Delta*T` construction, the
    mediator built here has genuine INDEPENDENT variance beyond what
    treatment alone determines (`zeta*T + noise`, not just `Delta*T`) — this
    is what makes the design well-identified in the first place, and is the
    exact property any real mediator construction (e.g. a per-individual
    measured SAE feature activation, not a single locus-level scalar) must
    have to be valid; see biolens.data.geuvadis.build_haplotype_sequence's
    docstring for the real-data fix this positive control validates against.

    Args:
        true_proportion_mediated: The exact fraction of `total_effect` that
            should flow through the mediator, by construction (0 to 1).
        n, total_effect, zeta, noise_sd: Data-generating parameters —
            defaults chosen to roughly match a real Geuvadis locus (n~445
            samples, a moderate total effect, meaningful residual noise).
        n_rep, random_state: Passed through to run_ikt_mediation.

    Returns:
        PositiveControlResult with both the known truth and what the
        pipeline actually recovered, for direct comparison.
    """
    if not (0.0 <= true_proportion_mediated <= 1.0):
        raise ValueError(
            f"true_proportion_mediated must be in [0, 1], got {true_proportion_mediated}"
        )

    rng = np.random.default_rng(random_state)
    treatment = rng.integers(0, 3, size=n).astype(float)

    indirect_effect = total_effect * true_proportion_mediated
    direct_effect = total_effect * (1.0 - true_proportion_mediated)
    gamma = indirect_effect / zeta if zeta != 0 else 0.0  # mediator -> outcome effect

    # Genuine per-sample noise on the mediator -- NOT a deterministic
    # function of treatment alone. This is the property the buggy
    # M = Delta*T construction lacked; see the docstring above.
    mediator = zeta * treatment + rng.normal(0, noise_sd, size=n)
    outcome = (
        direct_effect * treatment + gamma * mediator + rng.normal(0, noise_sd, size=n)
    )

    result = run_ikt_mediation(treatment, mediator, outcome, n_rep=n_rep, random_state=random_state)
    return PositiveControlResult(
        true_proportion_mediated=true_proportion_mediated,
        recovered_proportion_mediated=result.proportion_mediated,
        recovered_acme=result.acme_estimate,
        recovered_ade=result.ade_estimate,
        recovered_total_effect=result.total_effect,
    )


@dataclass
class SensitivityResult:
    rho_grid: np.ndarray = field(repr=False)
    acme_at_rho: np.ndarray = field(repr=False)
    acme_at_rho_zero: float
    rho_star: float | None  # rho at which ACME crosses zero; None if it never does on the grid

    def __str__(self) -> str:
        if self.rho_star is None:
            return (
                f"ACME(rho=0)={self.acme_at_rho_zero:.4f}; does not cross zero for "
                f"rho in [{self.rho_grid.min():.2f}, {self.rho_grid.max():.2f}] — "
                f"robust to unmeasured confounding across the checked range"
            )
        return (
            f"ACME(rho=0)={self.acme_at_rho_zero:.4f}; crosses zero at rho*="
            f"{self.rho_star:.3f} — the mediation conclusion would be nullified by "
            f"unmeasured confounding of this magnitude between the mediator and "
            f"outcome equations"
        )


def sensitivity_analysis(
    treatment: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
    covariates: pd.DataFrame | None = None,
    rho_grid: np.ndarray | None = None,
) -> SensitivityResult:
    """
    Imai-Keele-Yamamoto (2010) sensitivity analysis for the linear-linear
    mediation case — see module docstring for the closed-form derivation
    and its source. Answers: how much unmeasured confounding (correlation
    rho between the mediator- and outcome-equation errors) would it take to
    null out the ACME?

    Args:
        rho_grid: Values of rho to check; default -0.9 to 0.9 in 0.05 steps
                  (following the conventional range used in sensitivity
                  analyses of this kind — rho=+/-1 is a degenerate boundary).
    """
    if rho_grid is None:
        rho_grid = np.arange(-0.9, 0.9 + 1e-9, 0.05)

    data = pd.DataFrame({"T": treatment, "M": mediator, "Y": outcome})
    covariate_cols: list[str] = []
    if covariates is not None:
        for col in covariates.columns:
            data[col] = covariates[col].to_numpy()
            covariate_cols.append(col)

    mediator_formula = "M ~ T" + ("".join(f" + {c}" for c in covariate_cols))
    outcome_formula = "Y ~ T + M" + ("".join(f" + {c}" for c in covariate_cols))

    mediator_fit = sm.OLS.from_formula(mediator_formula, data).fit()
    outcome_fit = sm.OLS.from_formula(outcome_formula, data).fit()

    beta = float(mediator_fit.params["T"])       # treatment -> mediator effect
    delta = float(outcome_fit.params["M"])        # mediator -> outcome effect (naive, rho=0)
    sigma_1 = float(np.std(mediator_fit.resid, ddof=1))
    sigma_2 = float(np.std(outcome_fit.resid, ddof=1))

    # sqrt(1 - rho^2) denominator term is real, not optional -- see module
    # docstring for the Monte Carlo verification of why (an earlier version
    # omitted it, a real bug caught by external review, 2026-08). rho_grid
    # excludes +/-1 by construction (default range is -0.9 to 0.9), so this
    # never divides by zero for the default grid; a caller-supplied grid
    # reaching +/-1 would hit a real, correct singularity (the correction
    # is genuinely undefined there), not a bug to guard against silently.
    acme_at_rho = beta * (delta - rho_grid * sigma_2 / (sigma_1 * np.sqrt(1 - rho_grid**2)))
    acme_at_rho_zero = beta * delta

    rho_star = _find_zero_crossing(rho_grid, acme_at_rho)

    return SensitivityResult(
        rho_grid=rho_grid, acme_at_rho=acme_at_rho,
        acme_at_rho_zero=acme_at_rho_zero, rho_star=rho_star,
    )


def _find_zero_crossing(x: np.ndarray, y: np.ndarray) -> float | None:
    """Linear-interpolate the x value where y crosses zero, or None if y
    doesn't change sign anywhere on the grid."""
    sign_changes = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    i = sign_changes[0]
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]
    return float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
