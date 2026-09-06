"""
Tests for the IKT causal mediation engine (the Geuvadis case study).

Correctness strategy: construct synthetic data with a KNOWN true mediation
structure (M = beta*T + noise, Y = ade*T + delta*M + noise), and check the
recovered ACME/ADE match the known beta*delta / ade within a tolerance wide
enough for genuine Monte Carlo/sampling noise but tight enough to catch a
real implementation bug. Also cross-checks the closed-form sensitivity
analysis's rho=0 value against statsmodels' independently-simulated ACME —
two different computational paths to the same quantity should agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_mediation_data(n=2000, beta=1.5, delta=2.0, ade=0.5, noise_sd=1.0, seed=0):
    """T -> M -> Y with known coefficients, plus a direct T -> Y (ADE) path."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 3, size=n).astype(float)  # genotype dosage 0/1/2
    mediator = beta * treatment + rng.normal(0, noise_sd, size=n)
    outcome = ade * treatment + delta * mediator + rng.normal(0, noise_sd, size=n)
    return treatment, mediator, outcome


class TestRunIktMediation:
    def test_recovers_known_acme_and_ade(self):
        from biolens.eval.mediation import run_ikt_mediation

        beta, delta, ade = 1.5, 2.0, 0.5
        treatment, mediator, outcome = _synthetic_mediation_data(
            n=3000, beta=beta, delta=delta, ade=ade, seed=1
        )
        result = run_ikt_mediation(treatment, mediator, outcome, n_rep=500, random_state=1)

        true_acme = beta * delta
        assert result.acme_estimate == pytest.approx(true_acme, abs=0.3)
        assert result.ade_estimate == pytest.approx(ade, abs=0.3)
        # A real, strong mediation effect should be clearly significant at n=3000.
        assert result.acme_p_value < 0.01

    def test_proportion_mediated_between_zero_and_one_for_positive_effects(self):
        from biolens.eval.mediation import run_ikt_mediation

        treatment, mediator, outcome = _synthetic_mediation_data(
            n=2000, beta=1.5, delta=2.0, ade=0.5, seed=2
        )
        result = run_ikt_mediation(treatment, mediator, outcome, n_rep=300, random_state=2)
        assert 0.0 < result.proportion_mediated < 1.0

    def test_with_covariates(self):
        from biolens.eval.mediation import run_ikt_mediation

        rng = np.random.default_rng(3)
        n = 1500
        treatment = rng.integers(0, 3, size=n).astype(float)
        covariate = rng.normal(0, 1, size=n)
        mediator = 1.0 * treatment + 0.8 * covariate + rng.normal(0, 1, size=n)
        outcome = 0.3 * treatment + 1.5 * mediator + 0.5 * covariate + rng.normal(0, 1, size=n)

        result = run_ikt_mediation(
            treatment, mediator, outcome,
            covariates=pd.DataFrame({"pc1": covariate}),
            n_rep=300, random_state=3,
        )
        assert result.acme_estimate == pytest.approx(1.0 * 1.5, abs=0.4)

    def test_mismatched_lengths_raise(self):
        from biolens.eval.mediation import run_ikt_mediation

        with pytest.raises(ValueError):
            run_ikt_mediation(np.array([1, 2, 3]), np.array([1, 2]), np.array([1, 2, 3]))

    def test_str_output(self):
        from biolens.eval.mediation import MediationResult

        r = MediationResult(
            acme_estimate=0.5, acme_ci=(0.2, 0.8), acme_p_value=0.001,
            ade_estimate=0.3, ade_ci=(0.1, 0.5), ade_p_value=0.02,
            total_effect=0.8, total_effect_ci=(0.5, 1.1), total_effect_p_value=0.0001,
            proportion_mediated=0.625, mediator_std=1.0, n_rep=1000, n_obs=500,
        )
        s = str(r)
        assert "ACME=0.5000" in s
        assert "n=500" in s
        assert "prop. mediated=0.625" in s

    def test_proportion_mediated_reportable_when_total_effect_ci_excludes_zero(self):
        from biolens.eval.mediation import MediationResult

        r = MediationResult(
            acme_estimate=0.5, acme_ci=(0.2, 0.8), acme_p_value=0.001,
            ade_estimate=0.3, ade_ci=(0.1, 0.5), ade_p_value=0.02,
            total_effect=0.8, total_effect_ci=(0.5, 1.1), total_effect_p_value=0.0001,
            proportion_mediated=0.625, mediator_std=1.0, n_rep=1000, n_obs=500,
        )
        assert r.proportion_mediated_reportable is True
        assert "N/A" not in str(r)

    def test_proportion_mediated_not_reportable_when_total_effect_ci_includes_zero(self):
        """proportion_mediated = ACME/total_effect is a ratio estimator --
        badly behaved near a zero denominator (Fieller-type instability,
        flagged by external methodological review, 2026-08). A locus whose
        total-effect CI straddles zero shouldn't have proportion_mediated
        reported as if it were a reliable number."""
        from biolens.eval.mediation import MediationResult

        r = MediationResult(
            acme_estimate=0.01, acme_ci=(-0.3, 0.32), acme_p_value=0.9,
            ade_estimate=0.02, ade_ci=(-0.4, 0.44), ade_p_value=0.9,
            total_effect=0.03, total_effect_ci=(-0.5, 0.56), total_effect_p_value=0.9,
            proportion_mediated=0.333, mediator_std=1.0, n_rep=1000, n_obs=500,
        )
        assert r.proportion_mediated_reportable is False
        s = str(r)
        assert "N/A" in s
        assert "0.333" not in s

    def test_total_effect_ci_and_p_value_populated(self):
        from biolens.eval.mediation import run_ikt_mediation

        treatment, mediator, outcome = _synthetic_mediation_data(
            n=2000, beta=1.5, delta=2.0, ade=0.5, seed=12
        )
        result = run_ikt_mediation(treatment, mediator, outcome, n_rep=300, random_state=12)
        lo, hi = result.total_effect_ci
        assert lo < result.total_effect < hi
        # A real, strong total effect at n=2000 should be clearly significant
        # and correctly flagged as reportable.
        assert result.total_effect_p_value < 0.01
        assert result.proportion_mediated_reportable is True


class TestAcmeReportable:
    """acme_reportable guards against the degenerate zero-variance-mediator
    artifact caught in the Geuvadis case study run (job 9904287, 2026-08-13):
    7/26 naive-arm loci had a constant mediator (an SAE feature that never
    activated anywhere near that locus's haplotype window), which collapsed
    statsmodels' IKT bootstrap ACME distribution to a point mass at exactly
    zero and produced a spurious p=0 "highly significant" result — ADE ==
    total_effect exactly in every such case, confirming the mediator
    contributed nothing."""

    def test_constant_mediator_is_not_reportable(self):
        from biolens.eval.mediation import run_ikt_mediation

        treatment, _, outcome = _synthetic_mediation_data(n=500, seed=7)
        constant_mediator = np.zeros(500)  # SAE feature that never activates
        result = run_ikt_mediation(treatment, constant_mediator, outcome, n_rep=300, random_state=7)

        assert result.mediator_std == 0.0
        assert result.acme_reportable is False
        # ADE stays valid and equals the total effect exactly -- the
        # mediator contributed nothing, which is real signal, not an error.
        assert result.ade_estimate == pytest.approx(result.total_effect)

    def test_constant_mediator_str_shows_na_not_a_p_value(self):
        from biolens.eval.mediation import run_ikt_mediation

        treatment, _, outcome = _synthetic_mediation_data(n=500, seed=7)
        constant_mediator = np.full(500, 3.0)  # nonzero but still constant
        result = run_ikt_mediation(treatment, constant_mediator, outcome, n_rep=300, random_state=7)

        s = str(result)
        assert "ACME=N/A (mediator has zero variance -- untestable)" in s
        assert "p=0" not in s.split("ADE=")[0], (
            "a degenerate zero-variance mediator must never be reported as p=0 significant"
        )

    def test_real_varying_mediator_is_reportable(self):
        from biolens.eval.mediation import run_ikt_mediation

        treatment, mediator, outcome = _synthetic_mediation_data(n=500, seed=7)
        result = run_ikt_mediation(treatment, mediator, outcome, n_rep=300, random_state=7)

        assert result.mediator_std > 1e-8
        assert result.acme_reportable is True
        assert "ACME=N/A" not in str(result)


class TestRunNegativeControl:
    def test_pure_noise_mediator_gives_no_significant_acme(self):
        from biolens.eval.mediation import run_negative_control

        rng = np.random.default_rng(4)
        n = 2000
        treatment = rng.integers(0, 3, size=n).astype(float)
        outcome = 2.0 * treatment + rng.normal(0, 1, size=n)  # real T->Y effect, no real mediator

        result = run_negative_control(treatment, outcome, n_rep=500, random_state=4)
        lo, hi = result.acme_ci
        assert lo < 0 < hi, (
            f"Negative control should NOT find a significant mediation effect: "
            f"ACME 95% CI = [{lo}, {hi}]"
        )


class TestSensitivityAnalysis:
    def test_rho_zero_matches_naive_product_of_coefficients(self):
        """acme_at_rho_zero should equal beta*delta from a direct OLS fit —
        the same quantity the IKT simulation estimates via a completely
        different computational path (bootstrap simulation vs. closed-form
        product of two point estimates); they should agree closely."""
        from biolens.eval.mediation import sensitivity_analysis

        treatment, mediator, outcome = _synthetic_mediation_data(
            n=3000, beta=1.5, delta=2.0, ade=0.5, seed=5
        )
        result = sensitivity_analysis(treatment, mediator, outcome)
        assert result.acme_at_rho_zero == pytest.approx(1.5 * 2.0, abs=0.3)

    def test_cross_check_against_ikt_simulation(self):
        """The closed-form rho=0 ACME and the statsmodels-simulated ACME
        (run_ikt_mediation) should be close — two independent computational
        paths to the same quantity."""
        from biolens.eval.mediation import run_ikt_mediation, sensitivity_analysis

        treatment, mediator, outcome = _synthetic_mediation_data(
            n=3000, beta=1.2, delta=1.8, ade=0.4, seed=6
        )
        ikt_result = run_ikt_mediation(treatment, mediator, outcome, n_rep=500, random_state=6)
        sens_result = sensitivity_analysis(treatment, mediator, outcome)

        assert sens_result.acme_at_rho_zero == pytest.approx(ikt_result.acme_estimate, abs=0.2)

    def test_acme_monotonically_decreasing_in_rho_for_positive_effects(self):
        from biolens.eval.mediation import sensitivity_analysis

        treatment, mediator, outcome = _synthetic_mediation_data(
            n=2000, beta=1.5, delta=2.0, ade=0.5, seed=7
        )
        result = sensitivity_analysis(treatment, mediator, outcome)
        # beta and sigma2/sigma1 are both positive here, so ACME(rho) should
        # be strictly decreasing as rho increases (see closed-form in the
        # module docstring: slope = -beta*sigma2/sigma1 < 0).
        assert np.all(np.diff(result.acme_at_rho) < 0)

    def test_zero_crossing_found_when_effect_is_weak(self):
        """A weak true mediation effect should have a rho* within the
        checked grid — the sensitivity analysis should find where it nulls out."""
        from biolens.eval.mediation import sensitivity_analysis

        # Small beta*delta relative to the noise -> small ACME, and a
        # correspondingly small |rho| should null it out.
        treatment, mediator, outcome = _synthetic_mediation_data(
            n=2000, beta=0.3, delta=0.3, ade=0.1, noise_sd=1.0, seed=8
        )
        result = sensitivity_analysis(treatment, mediator, outcome)
        assert result.rho_star is not None
        assert -0.9 <= result.rho_star <= 0.9

    def test_no_crossing_returns_none_for_very_strong_effect(self):
        """A very strong, low-noise mediation effect shouldn't be nullified
        anywhere on a modest rho grid."""
        from biolens.eval.mediation import sensitivity_analysis

        treatment, mediator, outcome = _synthetic_mediation_data(
            n=2000, beta=10.0, delta=10.0, ade=1.0, noise_sd=0.5, seed=9
        )
        result = sensitivity_analysis(
            treatment, mediator, outcome, rho_grid=np.arange(-0.3, 0.3 + 1e-9, 0.05)
        )
        assert result.rho_star is None

    def test_zero_crossing_value_is_analytically_correct(self):
        """Construct a case with a precisely known rho* by controlling
        beta, delta, sigma1, sigma2 directly, and check the returned rho*
        matches the CORRECTED closed-form rho* = k / sqrt(1 + k^2), where
        k = delta * sigma_1 / sigma_2 (from setting ACME(rho)=0 in the
        sqrt(1-rho^2)-corrected formula in the module docstring).

        Regression test for a real bug (caught by external methodological
        review, 2026-08): an earlier version of sensitivity_analysis omitted
        the sqrt(1-rho^2) denominator term, whose zero-crossing is the
        UNCORRECTED rho* = k directly — 0.5 here, not the correct 0.447.
        Asserting the corrected value, not the old one, is the actual point
        of this test."""
        from biolens.eval.mediation import sensitivity_analysis

        rng = np.random.default_rng(10)
        n = 5000
        treatment = rng.integers(0, 3, size=n).astype(float)
        # Fix residual SDs precisely via large n and explicit noise scale.
        mediator = 1.0 * treatment + rng.normal(0, 2.0, size=n)   # sigma_1 ~= 2.0
        outcome = 0.0 * treatment + 1.0 * mediator + rng.normal(0, 4.0, size=n)  # sigma_2 ~= 4.0

        result = sensitivity_analysis(treatment, mediator, outcome, rho_grid=np.arange(-0.95, 0.95, 0.01))
        k = 1.0 * 2.0 / 4.0  # delta * sigma_1 / sigma_2 ~= 0.5
        expected_rho_star = k / np.sqrt(1 + k**2)  # ~= 0.4472
        assert result.rho_star == pytest.approx(expected_rho_star, abs=0.02)


class TestRunPositiveControl:
    """Real, permanent regression coverage for the most important safeguard
    against a silently-broken mediation pipeline: a synthetic dataset with
    a KNOWN true mediation fraction, run through the actual production
    pipeline, checked against the known truth. See run_positive_control's
    docstring for the real bug this directly guards against."""

    @pytest.mark.parametrize("true_prop", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_recovers_known_mediation_fraction(self, true_prop):
        from biolens.eval.mediation import run_positive_control

        result = run_positive_control(true_prop, n=445, n_rep=500, random_state=100)
        assert result.recovered_proportion_mediated == pytest.approx(true_prop, abs=0.15)

    def test_invalid_proportion_raises(self):
        from biolens.eval.mediation import run_positive_control

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            run_positive_control(1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            run_positive_control(-0.1)

    def test_degenerate_scalar_mediator_construction_fails_this_check(self):
        """Regression test locking in the bug's actual signature: the
        original, broken `M = Delta * T` construction (a single scalar
        times the treatment vector, with NO independent per-sample
        variance) must FAIL to distinguish a 100%-mediated dataset from a
        0%-mediated one under this exact positive-control design — this is
        the concrete, reproducible failure an external review caught, and
        this test ensures nobody can silently reintroduce a
        scalar-times-treatment mediator construction without a test
        catching it immediately."""
        from biolens.eval.mediation import run_ikt_mediation

        rng = np.random.default_rng(11)
        n = 445
        treatment = rng.integers(0, 3, size=n).astype(float)
        Delta = 0.03  # a small, real-scale SAE-activation-shift-like constant

        # World A: TRUE 100% mediated (Y depends only on M).
        mediator_A = Delta * treatment
        gamma = 20.0
        outcome_A = gamma * mediator_A + rng.normal(0, 0.01, size=n)
        result_A = run_ikt_mediation(treatment, mediator_A, outcome_A, n_rep=500, random_state=1)

        # World B: TRUE 0% mediated (Y depends only on T directly; same
        # mediator construction, feature is causally irrelevant).
        mediator_B = Delta * treatment
        tau = gamma * Delta
        outcome_B = tau * treatment + rng.normal(0, 0.01, size=n)
        result_B = run_ikt_mediation(treatment, mediator_B, outcome_B, n_rep=500, random_state=1)

        # The whole point: both "recover" essentially the same (near-zero)
        # proportion mediated, regardless of the true answer being 100% or
        # 0% -- proving the degenerate construction carries no information,
        # exactly as the external review found.
        assert result_A.proportion_mediated == pytest.approx(0.0, abs=0.05)
        assert result_B.proportion_mediated == pytest.approx(0.0, abs=0.05)
