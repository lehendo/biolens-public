"""
Tests for the clustered confirmatory sweep model.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestFitVerifiedRateSweep:
    def test_positive_relationship_detected(self):
        """A genuine positive relationship between log(min_positives) and
        verified_rate should be detected with a CI excluding zero."""
        from biolens.eval.statistics import fit_verified_rate_sweep

        rng = np.random.default_rng(0)
        n = 400
        min_positives = rng.choice([10, 20, 50, 100, 150, 200, 300], size=n)
        # True logistic relationship: higher min_positives -> higher verified_rate
        logit = -1.5 + 0.6 * np.log(min_positives)
        p = 1 / (1 + np.exp(-logit))
        verified = (rng.random(n) < p).astype(float)
        feature_idx = np.arange(n)  # every row its own cluster (no clustering effect)

        result = fit_verified_rate_sweep(min_positives, verified, feature_idx, random_state=1)

        assert result.slope > 0
        lo, hi = result.slope_ci_cluster
        assert lo > 0, f"95% CI should exclude 0 for a real effect: [{lo}, {hi}]"
        assert result.n_rows == n
        assert result.n_clusters == n  # every row is its own cluster here

    def test_no_relationship_ci_includes_zero(self):
        """verified_rate independent of min_positives -> slope CI should
        plausibly include zero (not a guaranteed property of any single
        random draw, but true at this sample size/seed)."""
        from biolens.eval.statistics import fit_verified_rate_sweep

        rng = np.random.default_rng(5)
        n = 300
        min_positives = rng.choice([10, 20, 50, 100, 150, 200, 300], size=n)
        verified = (rng.random(n) < 0.4).astype(float)  # independent of min_positives
        feature_idx = np.arange(n)

        result = fit_verified_rate_sweep(min_positives, verified, feature_idx, random_state=2)
        lo, hi = result.slope_ci_cluster
        assert lo < 0 < hi, f"No real effect — CI should span 0: [{lo}, {hi}]"

    def test_n_clusters_counts_distinct_feature_idx_not_rows(self):
        """Effective sample size for inference is the number of distinct
        clusters, not the row count — many GO terms sharing one feature_idx
        must count once."""
        from biolens.eval.statistics import fit_verified_rate_sweep

        rng = np.random.default_rng(3)
        n_clusters = 15
        rows_per_cluster = 8
        n = n_clusters * rows_per_cluster
        min_positives = rng.choice([20, 50, 100, 200], size=n)
        verified = (rng.random(n) < 0.5).astype(float)
        feature_idx = np.repeat(np.arange(n_clusters), rows_per_cluster)

        result = fit_verified_rate_sweep(min_positives, verified, feature_idx, random_state=4)
        assert result.n_rows == n
        assert result.n_clusters == n_clusters

    def test_clustering_widens_standard_error_under_intra_cluster_correlation(self):
        """When rows within a cluster are strongly correlated (many GO terms
        sharing one feature_idx, all getting the same outcome), the naive
        (unclustered) standard error understates true uncertainty — the
        cluster-robust SE from our fit should exceed the naive GLM SE
        computed on the identical data. This is the exact failure mode
        clustering is meant to fix."""
        import statsmodels.api as sm

        from biolens.eval.statistics import fit_verified_rate_sweep

        rng = np.random.default_rng(6)
        n_clusters = 20
        rows_per_cluster = 15

        cluster_min_positives = rng.choice([10, 20, 50, 100, 150, 200, 300], size=n_clusters)
        # Each cluster's outcome is decided ONCE (not per-row) — maximal
        # intra-cluster correlation, the worst case naive SE handles badly.
        logit = -0.5 + 0.4 * np.log(cluster_min_positives)
        p = 1 / (1 + np.exp(-logit))
        cluster_verified = (rng.random(n_clusters) < p).astype(float)

        min_positives = np.repeat(cluster_min_positives, rows_per_cluster)
        verified = np.repeat(cluster_verified, rows_per_cluster)
        feature_idx = np.repeat(np.arange(n_clusters), rows_per_cluster)

        result = fit_verified_rate_sweep(min_positives, verified, feature_idx, random_state=7)

        naive_fit = sm.GLM(
            verified, sm.add_constant(np.log(min_positives)), family=sm.families.Binomial()
        ).fit()
        naive_se_slope = float(naive_fit.bse[1])

        assert result.slope_se_cluster > naive_se_slope, (
            f"Cluster-robust SE ({result.slope_se_cluster}) should exceed the naive, "
            f"non-clustered SE ({naive_se_slope}) under strong intra-cluster correlation"
        )

    def test_bootstrap_ci_produced_with_enough_clusters(self):
        from biolens.eval.statistics import fit_verified_rate_sweep

        rng = np.random.default_rng(8)
        n_clusters = 30
        rows_per_cluster = 4
        n = n_clusters * rows_per_cluster
        min_positives = rng.choice([10, 20, 50, 100, 150, 200, 300], size=n)
        verified = (rng.random(n) < 0.5).astype(float)
        feature_idx = np.repeat(np.arange(n_clusters), rows_per_cluster)

        result = fit_verified_rate_sweep(
            min_positives, verified, feature_idx, n_bootstrap=200, random_state=9
        )
        assert result.slope_ci_bootstrap is not None
        assert result.n_bootstrap_successful > 0
        lo, hi = result.slope_ci_bootstrap
        assert lo <= hi

    def test_bootstrap_skipped_with_too_few_clusters(self):
        """Fewer than 10 distinct clusters -> bootstrap CI is None, not a
        misleadingly narrow interval from an unreliable resampling
        distribution."""
        from biolens.eval.statistics import fit_verified_rate_sweep

        rng = np.random.default_rng(10)
        n_clusters = 5
        rows_per_cluster = 10
        n = n_clusters * rows_per_cluster
        min_positives = rng.choice([10, 20, 50, 100], size=n)
        verified = (rng.random(n) < 0.5).astype(float)
        feature_idx = np.repeat(np.arange(n_clusters), rows_per_cluster)

        result = fit_verified_rate_sweep(min_positives, verified, feature_idx, random_state=11)
        assert result.slope_ci_bootstrap is None
        assert result.n_bootstrap_successful == 0

    def test_n_bootstrap_zero_skipped_without_warning(self, caplog):
        """n_bootstrap=0 (e.g. scripts/power_calculation.py's Monte Carlo
        power sweep, which only needs the analytic CI) must skip the
        bootstrap cleanly — no misleading '0/0 resamples converged'
        warning, which would look like a real reliability problem rather
        than an intentional skip."""
        import logging

        from biolens.eval.statistics import fit_verified_rate_sweep

        rng = np.random.default_rng(12)
        n_clusters = 20  # comfortably >= 10, so this isn't the too-few-clusters path
        rows_per_cluster = 5
        n = n_clusters * rows_per_cluster
        min_positives = rng.choice([10, 20, 50, 100], size=n)
        verified = (rng.random(n) < 0.5).astype(float)
        feature_idx = np.repeat(np.arange(n_clusters), rows_per_cluster)

        with caplog.at_level(logging.WARNING, logger="biolens.eval.statistics"):
            result = fit_verified_rate_sweep(
                min_positives, verified, feature_idx, n_bootstrap=0, random_state=13
            )
        assert result.slope_ci_bootstrap is None
        assert result.n_bootstrap_successful == 0
        assert not any("resamples converged" in r.message for r in caplog.records)

    def test_mismatched_lengths_raise(self):
        from biolens.eval.statistics import fit_verified_rate_sweep

        with pytest.raises(ValueError):
            fit_verified_rate_sweep(
                np.array([10, 20, 30]), np.array([1.0, 0.0]), np.array([0, 1, 2])
            )

    def test_empty_input_raises(self):
        from biolens.eval.statistics import fit_verified_rate_sweep

        with pytest.raises(ValueError):
            fit_verified_rate_sweep(np.array([]), np.array([]), np.array([]))

    def test_predict_is_monotonic_in_min_positives_for_positive_slope(self):
        from biolens.eval.statistics import SweepFitResult

        result = SweepFitResult(
            intercept=-1.0, slope=0.5,
            intercept_se_cluster=0.1, slope_se_cluster=0.1,
            slope_ci_cluster=(0.3, 0.7), slope_p_value_cluster=0.001,
            slope_ci_bootstrap=None, n_bootstrap_successful=0,
            n_rows=100, n_clusters=20,
        )
        assert result.predict(10) < result.predict(100) < result.predict(1000)

    def test_effect_per_doubling_positive_for_positive_slope(self):
        from biolens.eval.statistics import SweepFitResult

        result = SweepFitResult(
            intercept=-1.0, slope=0.5,
            intercept_se_cluster=0.1, slope_se_cluster=0.1,
            slope_ci_cluster=(0.3, 0.7), slope_p_value_cluster=0.001,
            slope_ci_bootstrap=None, n_bootstrap_successful=0,
            n_rows=100, n_clusters=20,
        )
        assert result.effect_per_doubling() > 0

class TestBuildSweepRows:
    def _annotation(self, status, claimed_go):
        from biolens.eval.feature_inspection import VerifiedAnnotation

        return VerifiedAnnotation(
            status=status, claimed_go=claimed_go, claimed_concept="c", real_identity="r",
        )

    def test_joins_matching_claim_and_maps_status(self):
        from biolens.eval.statistics import build_sweep_rows

        sweep_results = {
            20: [{"go_id": "GO:A", "best_feature_idx": 5}],
            100: [{"go_id": "GO:B", "best_feature_idx": 7}],
        }
        verified = {
            5: self._annotation("confirmed", ["GO:A"]),
            7: self._annotation("spurious", ["GO:B"]),
        }

        mp, target, feat = build_sweep_rows(sweep_results, verified)
        assert list(mp) == [20.0, 100.0]
        assert list(target) == [1.0, 0.0]
        assert list(feat) == [5, 7]

    def test_plausible_maps_to_half(self):
        from biolens.eval.statistics import build_sweep_rows

        sweep_results = {50: [{"go_id": "GO:A", "best_feature_idx": 1}]}
        verified = {1: self._annotation("plausible", ["GO:A"])}

        _, target, _ = build_sweep_rows(sweep_results, verified)
        assert list(target) == [0.5]

    def test_unverifiable_excluded(self):
        from biolens.eval.statistics import build_sweep_rows

        sweep_results = {50: [{"go_id": "GO:A", "best_feature_idx": 1}]}
        verified = {1: self._annotation("unverifiable", ["GO:A"])}

        mp, target, feat = build_sweep_rows(sweep_results, verified)
        assert len(mp) == 0

    def test_unregistered_feature_excluded(self):
        """A top-hit feature never checked against the registry contributes
        no row — the fit must not silently guess a status for it."""
        from biolens.eval.statistics import build_sweep_rows

        sweep_results = {50: [{"go_id": "GO:A", "best_feature_idx": 999}]}
        verified = {1: self._annotation("confirmed", ["GO:A"])}

        mp, target, feat = build_sweep_rows(sweep_results, verified)
        assert len(mp) == 0

    def test_go_id_not_in_claimed_go_excluded(self):
        """The feature IS registered, but for a DIFFERENT GO claim than the
        one in this row — must not be silently reused for an unrelated claim."""
        from biolens.eval.statistics import build_sweep_rows

        sweep_results = {50: [{"go_id": "GO:OTHER", "best_feature_idx": 1}]}
        verified = {1: self._annotation("confirmed", ["GO:A"])}

        mp, target, feat = build_sweep_rows(sweep_results, verified)
        assert len(mp) == 0

    def test_feature_hit_for_multiple_go_terms_produces_multiple_rows(self):
        """A single feature that's the top hit for several claimed GO terms
        (the exact clustering scenario fit_verified_rate_sweep clusters on)
        must produce one row per claim, all sharing the same feature_idx."""
        from biolens.eval.statistics import build_sweep_rows

        sweep_results = {
            20: [
                {"go_id": "GO:A", "best_feature_idx": 5},
                {"go_id": "GO:B", "best_feature_idx": 5},
            ],
        }
        verified = {5: self._annotation("confirmed", ["GO:A", "GO:B"])}

        mp, target, feat = build_sweep_rows(sweep_results, verified)
        assert len(mp) == 2
        assert list(feat) == [5, 5]

    def test_output_feeds_directly_into_fit(self):
        """End-to-end: build_sweep_rows' output shape/dtype must work
        directly as fit_verified_rate_sweep's input without adaptation."""
        from biolens.eval.statistics import build_sweep_rows, fit_verified_rate_sweep

        sweep_results = {
            mp: [{"go_id": f"GO:{i}", "best_feature_idx": i} for i in range(15)]
            for mp in [10, 20, 50, 100]
        }
        verified = {
            i: self._annotation("confirmed" if i % 2 == 0 else "spurious", [f"GO:{i}"])
            for i in range(15)
        }

        mp, target, feat = build_sweep_rows(sweep_results, verified)
        result = fit_verified_rate_sweep(mp, target, feat, random_state=1)
        assert result.n_rows == 60  # 15 features x 4 sweep points


class TestSweepFitResultStr:
    def test_str_output_includes_key_fields(self):
        from biolens.eval.statistics import SweepFitResult

        result = SweepFitResult(
            intercept=-1.0, slope=0.5,
            intercept_se_cluster=0.1, slope_se_cluster=0.15,
            slope_ci_cluster=(0.2, 0.8), slope_p_value_cluster=0.01,
            slope_ci_bootstrap=(0.18, 0.82), n_bootstrap_successful=1900,
            n_rows=500, n_clusters=80,
        )
        s = str(result)
        assert "n_clusters=80" in s
        assert "cluster-bootstrap" in s


class TestBootstrapMeanAurocInflation:
    def test_ci_excludes_zero_for_clearly_positive_inflation(self):
        """A real, non-degenerate positive mean inflation should get a CI
        that excludes zero, given enough independent-ish clusters."""
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        rng = np.random.default_rng(0)
        n = 300
        auroc_inflation = rng.normal(loc=0.05, scale=0.02, size=n)
        feature_idx = np.arange(n)  # every row its own cluster

        result = bootstrap_mean_auroc_inflation(auroc_inflation, feature_idx, random_state=1)

        assert result.mean == pytest.approx(np.mean(auroc_inflation))
        lo, hi = result.ci
        assert lo > 0, f"95% CI should exclude 0 for clearly positive inflation: [{lo}, {hi}]"
        assert result.n_rows == n
        assert result.n_clusters == n

    def test_ci_includes_zero_for_noise_centered_at_zero(self):
        """Inflation values centered at zero (e.g. the permuted-label
        control) should plausibly get a CI spanning zero."""
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        rng = np.random.default_rng(3)
        n = 300
        auroc_inflation = rng.normal(loc=0.0, scale=0.02, size=n)
        feature_idx = np.arange(n)

        result = bootstrap_mean_auroc_inflation(auroc_inflation, feature_idx, random_state=2)
        lo, hi = result.ci
        assert lo < 0 < hi, f"No real effect — CI should span 0: [{lo}, {hi}]"

    def test_clustering_widens_ci_under_intra_cluster_correlation(self):
        """The same non-independence property fit_verified_rate_sweep's
        cluster-robust SE relies on: if many rows share a feature_idx and
        that shared feature drives correlated inflation values, clustering
        must produce a wider (or equal) CI than treating every row as its
        own independent cluster — otherwise the clustering isn't doing
        anything."""
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        rng = np.random.default_rng(3)
        n_clusters = 15
        rows_per_cluster = 10
        cluster_effects = rng.normal(loc=0.05, scale=0.03, size=n_clusters)
        feature_idx = np.repeat(np.arange(n_clusters), rows_per_cluster)
        auroc_inflation = np.repeat(cluster_effects, rows_per_cluster) + rng.normal(
            scale=0.005, size=n_clusters * rows_per_cluster
        )
        independent_idx = np.arange(len(auroc_inflation))

        clustered = bootstrap_mean_auroc_inflation(auroc_inflation, feature_idx, random_state=4)
        unclustered = bootstrap_mean_auroc_inflation(
            auroc_inflation, independent_idx, random_state=4
        )

        clustered_width = clustered.ci[1] - clustered.ci[0]
        unclustered_width = unclustered.ci[1] - unclustered.ci[0]
        assert clustered_width > unclustered_width, (
            f"Clustered CI ({clustered_width:.4f}) should be wider than treating "
            f"correlated rows as independent ({unclustered_width:.4f})"
        )
        assert clustered.n_clusters == n_clusters
        assert unclustered.n_clusters == len(auroc_inflation)

    def test_mismatched_lengths_raise(self):
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        with pytest.raises(ValueError):
            bootstrap_mean_auroc_inflation(np.array([0.1, 0.2]), np.array([0]))

    def test_empty_input_raises(self):
        from biolens.eval.statistics import bootstrap_mean_auroc_inflation

        with pytest.raises(ValueError):
            bootstrap_mean_auroc_inflation(np.array([]), np.array([]))

    def test_str_output_includes_key_fields(self):
        from biolens.eval.statistics import InflationBootstrapResult

        result = InflationBootstrapResult(
            mean=0.05, ci=(0.02, 0.08), n_bootstrap_successful=1950, n_rows=200, n_clusters=40,
        )
        s = str(result)
        assert "0.0500" in s
        assert "n_clusters=40" in s


class TestFitPooledScaleInteraction:
    """Tests for the cross-model verified_rate ~ log(min_positives) *
    model_scale pooled fit — the replacement for informally comparing N
    separate per-model SweepFitResults."""

    def _make_scale(self, rng, n, intercept, slope, cluster_offset=0):
        """Simulate one model's rows: n_clusters = n // 3 features, each the
        top hit for a handful of GO terms at various min_positives."""
        min_positives = rng.choice([10, 20, 50, 100, 150, 200, 300], size=n)
        logit = intercept + slope * np.log(min_positives)
        p = 1 / (1 + np.exp(-logit))
        verified = (rng.random(n) < p).astype(float)
        # every row its own cluster, offset so raw indices can collide across
        # scales without actually being the same feature
        feature_idx = np.arange(n) + cluster_offset
        return min_positives.astype(float), verified, feature_idx

    def test_detects_real_slope_difference_between_scales(self):
        """Two scales with genuinely different true slopes should produce a
        significant interaction test and recover each scale's own slope."""
        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(0)
        n = 500
        mp_a, v_a, f_a = self._make_scale(rng, n, intercept=-1.5, slope=0.05, cluster_offset=0)
        mp_b, v_b, f_b = self._make_scale(rng, n, intercept=-1.5, slope=0.9, cluster_offset=10_000)

        min_positives = np.concatenate([mp_a, mp_b])
        verified = np.concatenate([v_a, v_b])
        feature_idx = np.concatenate([f_a, f_b])
        model_scale = np.array(["scale_a"] * n + ["scale_b"] * n, dtype=object)

        result = fit_pooled_scale_interaction(
            min_positives, verified, feature_idx, model_scale, random_state=1,
        )

        assert result.interaction_p_value < 0.05, (
            f"Genuinely different slopes (0.05 vs 0.9) should yield a significant "
            f"interaction test, got p={result.interaction_p_value}"
        )
        assert result.slope["scale_a"] < result.slope["scale_b"]

    def test_no_interaction_when_slopes_match(self):
        """Two scales sharing the same true slope should NOT produce a
        significant interaction test — the pooled model shouldn't manufacture
        a scale difference that isn't there."""
        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(2)
        n = 500
        mp_a, v_a, f_a = self._make_scale(rng, n, intercept=-1.0, slope=0.4, cluster_offset=0)
        mp_b, v_b, f_b = self._make_scale(rng, n, intercept=-1.0, slope=0.4, cluster_offset=10_000)

        min_positives = np.concatenate([mp_a, mp_b])
        verified = np.concatenate([v_a, v_b])
        feature_idx = np.concatenate([f_a, f_b])
        model_scale = np.array(["scale_a"] * n + ["scale_b"] * n, dtype=object)

        result = fit_pooled_scale_interaction(
            min_positives, verified, feature_idx, model_scale, random_state=3,
        )

        assert result.interaction_p_value > 0.05, (
            f"Identical true slopes should not yield a significant interaction "
            f"test, got p={result.interaction_p_value}"
        )

    def test_colliding_raw_feature_indices_across_scales_not_merged(self):
        """Feature index 0 in scale_a and feature index 0 in scale_b are
        unrelated features (different SAEs) — they must not be pooled into
        one cluster just because the raw integer matches."""
        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(4)
        n = 300
        # cluster_offset=0 for BOTH scales on purpose: identical raw indices
        mp_a, v_a, f_a = self._make_scale(rng, n, intercept=-1.0, slope=0.3, cluster_offset=0)
        mp_b, v_b, f_b = self._make_scale(rng, n, intercept=-1.0, slope=0.3, cluster_offset=0)

        min_positives = np.concatenate([mp_a, mp_b])
        verified = np.concatenate([v_a, v_b])
        feature_idx = np.concatenate([f_a, f_b])
        model_scale = np.array(["scale_a"] * n + ["scale_b"] * n, dtype=object)

        result = fit_pooled_scale_interaction(
            min_positives, verified, feature_idx, model_scale, random_state=5,
        )

        # n distinct raw feature_idx values per scale (every row its own
        # cluster) — if scale/feature were wrongly merged, n_clusters_total
        # would be n (the union of identical raw indices) instead of 2n.
        assert result.n_clusters_per_scale["scale_a"] == n
        assert result.n_clusters_per_scale["scale_b"] == n
        assert result.n_clusters_total == 2 * n

    def test_reference_scale_selection(self):
        """reference_scale controls parameterization only — every scale still
        gets its own slope/CI/p-value regardless of which is the reference."""
        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(6)
        n = 300
        mp_a, v_a, f_a = self._make_scale(rng, n, intercept=-1.0, slope=0.3, cluster_offset=0)
        mp_b, v_b, f_b = self._make_scale(rng, n, intercept=-1.0, slope=0.6, cluster_offset=10_000)

        min_positives = np.concatenate([mp_a, mp_b])
        verified = np.concatenate([v_a, v_b])
        feature_idx = np.concatenate([f_a, f_b])
        model_scale = np.array(["scale_a"] * n + ["scale_b"] * n, dtype=object)

        result_a_ref = fit_pooled_scale_interaction(
            min_positives, verified, feature_idx, model_scale,
            reference_scale="scale_a", random_state=7, n_bootstrap=0,
        )
        result_b_ref = fit_pooled_scale_interaction(
            min_positives, verified, feature_idx, model_scale,
            reference_scale="scale_b", random_state=7, n_bootstrap=0,
        )

        # Same underlying model — marginal slope estimates must match
        # regardless of which scale was chosen as the reference.
        assert result_a_ref.slope["scale_a"] == pytest.approx(result_b_ref.slope["scale_a"], abs=1e-6)
        assert result_a_ref.slope["scale_b"] == pytest.approx(result_b_ref.slope["scale_b"], abs=1e-6)
        assert result_a_ref.interaction_p_value == pytest.approx(
            result_b_ref.interaction_p_value, abs=1e-6
        )

    def test_invalid_reference_scale_raises(self):
        from biolens.eval.statistics import fit_pooled_scale_interaction

        n = 20
        min_positives = np.full(n, 10.0)
        verified = np.zeros(n)
        feature_idx = np.arange(n)
        model_scale = np.array(["a"] * 10 + ["b"] * 10, dtype=object)

        with pytest.raises(ValueError):
            fit_pooled_scale_interaction(
                min_positives, verified, feature_idx, model_scale, reference_scale="c",
            )

    def test_single_scale_raises(self):
        from biolens.eval.statistics import fit_pooled_scale_interaction

        n = 20
        min_positives = np.full(n, 10.0)
        verified = np.zeros(n)
        feature_idx = np.arange(n)
        model_scale = np.array(["a"] * n, dtype=object)

        with pytest.raises(ValueError):
            fit_pooled_scale_interaction(min_positives, verified, feature_idx, model_scale)

    def test_mismatched_lengths_raise(self):
        from biolens.eval.statistics import fit_pooled_scale_interaction

        with pytest.raises(ValueError):
            fit_pooled_scale_interaction(
                np.array([10.0, 20.0]), np.array([0.0]), np.array([1, 2]),
                np.array(["a", "b"], dtype=object),
            )

    def test_empty_input_raises(self):
        from biolens.eval.statistics import fit_pooled_scale_interaction

        with pytest.raises(ValueError):
            fit_pooled_scale_interaction(
                np.array([]), np.array([]), np.array([]), np.array([], dtype=object),
            )

    def test_three_scales_joint_interaction(self):
        """Three scales at once — the joint Wald test must have the right
        degrees of freedom (2 interaction terms) and still detect a genuine
        difference among them."""
        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(8)
        n = 400
        mp_a, v_a, f_a = self._make_scale(rng, n, intercept=-1.0, slope=0.1, cluster_offset=0)
        mp_b, v_b, f_b = self._make_scale(rng, n, intercept=-1.0, slope=0.5, cluster_offset=10_000)
        mp_c, v_c, f_c = self._make_scale(rng, n, intercept=-1.0, slope=0.9, cluster_offset=20_000)

        min_positives = np.concatenate([mp_a, mp_b, mp_c])
        verified = np.concatenate([v_a, v_b, v_c])
        feature_idx = np.concatenate([f_a, f_b, f_c])
        model_scale = np.array(["a"] * n + ["b"] * n + ["c"] * n, dtype=object)

        result = fit_pooled_scale_interaction(
            min_positives, verified, feature_idx, model_scale, random_state=9,
        )

        assert set(result.scales) == {"a", "b", "c"}
        assert result.interaction_p_value < 0.05
        assert result.slope["a"] < result.slope["b"] < result.slope["c"]

    def test_bootstrap_ci_present_with_enough_clusters(self):
        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(10)
        n = 200
        mp_a, v_a, f_a = self._make_scale(rng, n, intercept=-1.0, slope=0.3, cluster_offset=0)
        mp_b, v_b, f_b = self._make_scale(rng, n, intercept=-1.0, slope=0.3, cluster_offset=10_000)

        min_positives = np.concatenate([mp_a, mp_b])
        verified = np.concatenate([v_a, v_b])
        feature_idx = np.concatenate([f_a, f_b])
        model_scale = np.array(["scale_a"] * n + ["scale_b"] * n, dtype=object)

        result = fit_pooled_scale_interaction(
            min_positives, verified, feature_idx, model_scale,
            random_state=11, n_bootstrap=200,
        )

        assert result.slope_ci_bootstrap is not None
        assert "scale_a" in result.slope_ci_bootstrap
        assert "scale_b" in result.slope_ci_bootstrap
        assert result.n_bootstrap_successful > 0

    def test_str_output_includes_key_fields(self):
        from biolens.eval.statistics import PooledScaleFitResult

        result = PooledScaleFitResult(
            reference_scale="scale_a",
            scales=["scale_a", "scale_b"],
            slope={"scale_a": 0.2, "scale_b": 0.8},
            slope_se_cluster={"scale_a": 0.1, "scale_b": 0.15},
            slope_ci_cluster={"scale_a": (0.0, 0.4), "scale_b": (0.5, 1.1)},
            slope_p_value_cluster={"scale_a": 0.04, "scale_b": 0.001},
            slope_ci_bootstrap=None,
            interaction_wald_stat=6.5,
            interaction_p_value=0.011,
            n_rows=400,
            n_clusters_total=90,
            n_clusters_per_scale={"scale_a": 40, "scale_b": 50},
            n_bootstrap_successful=0,
        )
        s = str(result)
        assert "scale_a" in s and "scale_b" in s
        assert "n_clusters_total=90" in s
        assert "interaction" in s.lower()

    def test_trend_fields_none_when_scale_param_counts_not_given(self):
        """Backward compatibility: existing callers that don't pass
        scale_param_counts must see no behavior change."""
        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(20)
        n = 200
        mp_a, v_a, f_a = self._make_scale(rng, n, intercept=-1.0, slope=0.3, cluster_offset=0)
        mp_b, v_b, f_b = self._make_scale(rng, n, intercept=-1.0, slope=0.3, cluster_offset=10_000)
        min_positives = np.concatenate([mp_a, mp_b])
        verified = np.concatenate([v_a, v_b])
        feature_idx = np.concatenate([f_a, f_b])
        model_scale = np.array(["scale_a"] * n + ["scale_b"] * n, dtype=object)

        result = fit_pooled_scale_interaction(
            min_positives, verified, feature_idx, model_scale, random_state=21, n_bootstrap=0,
        )
        assert result.scale_trend_slope is None
        assert result.scale_trend_ci_bootstrap is None
        assert result.scale_trend_p_bootstrap is None

    def test_scale_param_counts_missing_entry_raises(self):
        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(22)
        n = 100
        mp_a, v_a, f_a = self._make_scale(rng, n, intercept=-1.0, slope=0.3, cluster_offset=0)
        mp_b, v_b, f_b = self._make_scale(rng, n, intercept=-1.0, slope=0.3, cluster_offset=10_000)
        min_positives = np.concatenate([mp_a, mp_b])
        verified = np.concatenate([v_a, v_b])
        feature_idx = np.concatenate([f_a, f_b])
        model_scale = np.array(["scale_a"] * n + ["scale_b"] * n, dtype=object)

        with pytest.raises(ValueError, match="missing"):
            fit_pooled_scale_interaction(
                min_positives, verified, feature_idx, model_scale,
                n_bootstrap=0, scale_param_counts={"scale_a": 8e6},  # scale_b missing
            )

    def test_trend_point_estimate_uses_precision_weighted_centering(self):
        """Regression test for the exact bug an external review caught and
        reproduced (2026-08): centering x on the UNWEIGHTED mean while still
        weighting the cross-products gives a different, wrong slope, not
        just a different SE. Four synthetic scales with known slopes and
        known (unequal) precisions must recover the precision-weighted WLS
        answer, computed independently here via the textbook formula, not
        the buggy one."""
        import math

        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(23)
        # Four scales, deliberately UNEQUAL sample sizes -> unequal precision,
        # exactly the regime where weighted vs unweighted centering diverge.
        specs = [
            ("s1", 1e6, 600, -1.0, 0.8),
            ("s2", 1e7, 300, -1.0, 0.5),
            ("s3", 1e8, 150, -1.0, 0.3),
            ("s4", 1e9, 80, -1.0, -0.1),
        ]
        mp_all, v_all, f_all, scale_all = [], [], [], []
        offset = 0
        for name, _size, n, intercept, true_slope in specs:
            mp, v, f = self._make_scale(rng, n, intercept, true_slope, cluster_offset=offset)
            mp_all.append(mp); v_all.append(v); f_all.append(f)
            scale_all.append(np.full(n, name, dtype=object))
            offset += 100_000

        result = fit_pooled_scale_interaction(
            np.concatenate(mp_all), np.concatenate(v_all), np.concatenate(f_all),
            np.concatenate(scale_all), random_state=24, n_bootstrap=0,
            scale_param_counts={name: size for name, size, *_ in specs},
        )

        # Independently compute the precision-weighted WLS slope from the
        # result's OWN reported per-scale slope/SE (not re-deriving from raw
        # data) -- this isolates exactly the centering-formula correctness
        # this test exists to guard, matching the textbook WLS formula.
        x = np.array([math.log10(size) for _, size, *_ in specs])
        y = np.array([result.slope[name] for name, *_ in specs])
        w = np.array([1.0 / result.slope_se_cluster[name] ** 2 for name, *_ in specs])
        xbar_w = np.sum(w * x) / np.sum(w)
        ybar_w = np.sum(w * y) / np.sum(w)
        expected = np.sum(w * (x - xbar_w) * (y - ybar_w)) / np.sum(w * (x - xbar_w) ** 2)

        assert result.scale_trend_slope == pytest.approx(expected, abs=1e-9)

        # And confirm it's NOT the buggy (unweighted-centered) value, which
        # would differ whenever the true slopes/SEs are as unequal as here.
        xbar_unweighted = np.mean(x)
        buggy = np.sum(w * (x - xbar_unweighted) * y) / np.sum(w * (x - xbar_unweighted) ** 2)
        assert result.scale_trend_slope != pytest.approx(buggy, abs=1e-6)

    def test_trend_bootstrap_ci_present_and_correlation_correct(self):
        """The bootstrap CI must actually be populated, and -- the entire
        point of computing this inside the existing stratified bootstrap
        rather than as a second regression on the four point estimates --
        it must come from real per-replicate draws, not an analytic formula
        assuming independence."""
        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(25)
        specs = [("s1", 1e6, 200, -1.0, 0.9), ("s2", 1e8, 200, -1.0, 0.1)]
        mp_all, v_all, f_all, scale_all = [], [], [], []
        offset = 0
        for name, _size, n, intercept, true_slope in specs:
            mp, v, f = self._make_scale(rng, n, intercept, true_slope, cluster_offset=offset)
            mp_all.append(mp); v_all.append(v); f_all.append(f)
            scale_all.append(np.full(n, name, dtype=object))
            offset += 100_000

        result = fit_pooled_scale_interaction(
            np.concatenate(mp_all), np.concatenate(v_all), np.concatenate(f_all),
            np.concatenate(scale_all), random_state=26, n_bootstrap=200,
            scale_param_counts={name: size for name, size, *_ in specs},
        )

        assert result.scale_trend_slope is not None
        assert result.scale_trend_ci_bootstrap is not None
        lo, hi = result.scale_trend_ci_bootstrap
        assert lo < result.scale_trend_slope < hi
        assert 0.0 <= result.scale_trend_p_bootstrap <= 1.0

    def test_trend_point_estimate_not_equal_weight(self):
        """Regression test for the estimand-mismatch bug caught via external
        review (2026-08): an earlier version harvested the bootstrap trend
        with EQUAL weights while the point estimate used PRECISION weights
        -- two different quantities differing by ~2% on real data, silently
        reported together as if they were one estimate+CI. Confirms the
        point estimate is genuinely precision-weighted, not the naive
        unweighted trend the old bug's bootstrap half actually computed
        (the deterministic half of the regression -- see
        test_trend_bootstrap_uses_caller_supplied_contrast_exactly below
        for the half that guards the bootstrap loop itself)."""
        import math

        from biolens.eval.statistics import fit_pooled_scale_interaction

        rng = np.random.default_rng(27)
        specs = [
            ("s1", 1e6, 500, -1.0, 0.8),
            ("s2", 1e7, 250, -1.0, 0.5),
            ("s3", 1e8, 120, -1.0, 0.3),
            ("s4", 1e9, 60, -1.0, -0.1),
        ]
        mp_all, v_all, f_all, scale_all = [], [], [], []
        offset = 0
        for name, _size, n, intercept, true_slope in specs:
            mp, v, f = self._make_scale(rng, n, intercept, true_slope, cluster_offset=offset)
            mp_all.append(mp); v_all.append(v); f_all.append(f)
            scale_all.append(np.full(n, name, dtype=object))
            offset += 100_000

        result = fit_pooled_scale_interaction(
            np.concatenate(mp_all), np.concatenate(v_all), np.concatenate(f_all),
            np.concatenate(scale_all), random_state=28, n_bootstrap=400,
            scale_param_counts={name: size for name, size, *_ in specs},
        )

        x = np.array([math.log10(size) for _, size, *_ in specs])
        y = np.array([result.slope[name] for name, *_ in specs])
        xc_unweighted = x - x.mean()
        unweighted_trend = float(np.sum(xc_unweighted * y) / np.sum(xc_unweighted**2))
        assert result.scale_trend_slope != pytest.approx(unweighted_trend, rel=1e-3)

    def test_trend_bootstrap_uses_caller_supplied_contrast_exactly(self):
        """Deterministic regression test for the estimand-mismatch bug
        (external review, 2026-08): the original defect was that
        `_pooled_cluster_bootstrap_ci` computed its OWN internal contrast
        weights (equal-weight) instead of using the fixed precision-weighted
        contrast the caller already computed for the point estimate -- two
        different estimands silently reported together as one estimate+CI.

        An earlier version of this test tried to catch that statistically
        (comparing the bootstrap draw mean to the point estimate within a
        few bootstrap standard errors of the mean). Verified directly, by
        hand-reconstructing the old buggy contrast and re-running the
        bootstrap, that this approach does NOT reliably discriminate the
        fixed implementation from the buggy one -- the mismatch this bug
        produces is too small relative to Monte Carlo noise in the draw
        mean, even after using the true per-replicate draws (not a CI-range
        approximation) and even at n_bootstrap=400.

        This test instead exploits a property that holds EXACTLY, with no
        sampling tolerance needed: pass a ONE-HOT contrast (weight 1.0 on a
        single scale, 0.0 on the rest). If the bootstrap loop genuinely uses
        the caller's contrast dict as-is, every replicate's trend value must
        equal exactly that scale's own resampled slope on that replicate --
        so the trend draws' percentile CI must exactly equal that scale's
        own independently-tracked per-scale bootstrap CI (both come from the
        same underlying per-replicate values). If the loop instead
        recomputes its own weights internally (the original bug's shape),
        this equality breaks even though the caller passed a one-hot
        contrast, since the loop's own weights would not be one-hot."""
        from biolens.eval.statistics import _pooled_cluster_bootstrap_ci

        rng = np.random.default_rng(31)
        specs = [
            ("s1", 500, -1.0, 0.8),
            ("s2", 250, -1.0, 0.5),
            ("s3", 120, -1.0, 0.3),
        ]
        mp_all, v_all, f_all, scale_all = [], [], [], []
        offset = 0
        for name, n, intercept, true_slope in specs:
            mp, v, f = self._make_scale(rng, n, intercept, true_slope, cluster_offset=offset)
            mp_all.append(mp); v_all.append(v); f_all.append(f)
            scale_all.append(np.full(n, name, dtype=object))
            offset += 100_000

        min_positives = np.concatenate(mp_all)
        verified = np.concatenate(v_all)
        feature_idx = np.concatenate(f_all)
        model_scale = np.concatenate(scale_all)
        all_scales = [name for name, *_ in specs]

        for chosen in all_scales:
            one_hot = {s: (1.0 if s == chosen else 0.0) for s in all_scales}
            ci_out, n_successful, trend_draws = _pooled_cluster_bootstrap_ci(
                min_positives, verified, feature_idx, model_scale,
                all_scales[0], all_scales[1:],
                n_bootstrap=200, confidence=0.95, random_state=32,
                scale_trend_contrast=one_hot,
            )
            assert n_successful >= 50
            trend_lo, trend_hi = np.percentile(trend_draws, [2.5, 97.5])
            scale_lo, scale_hi = ci_out[chosen]
            assert trend_lo == pytest.approx(scale_lo, abs=1e-9), (
                f"One-hot trend contrast on {chosen!r} must reproduce that scale's own "
                f"bootstrap CI exactly -- got trend_lo={trend_lo}, scale_lo={scale_lo}. "
                f"A mismatch means the bootstrap loop is not using the caller-supplied "
                f"contrast as-is (the estimand-mismatch bug's shape)."
            )
            assert trend_hi == pytest.approx(scale_hi, abs=1e-9)


class TestBuildPooledSweepRows:
    def _annotation(self, status, claimed_go):
        from biolens.eval.feature_inspection import VerifiedAnnotation

        return VerifiedAnnotation(
            status=status, claimed_go=claimed_go, claimed_concept="c", real_identity="r",
        )

    def test_combines_multiple_models_with_scale_labels(self):
        from biolens.eval.statistics import build_pooled_sweep_rows

        per_model_sweep_results = {
            "esm2_8m": {20: [{"go_id": "GO:A", "best_feature_idx": 5}]},
            "esm2_35m": {50: [{"go_id": "GO:B", "best_feature_idx": 5}]},
        }
        per_model_verified = {
            "esm2_8m": {5: self._annotation("confirmed", ["GO:A"])},
            "esm2_35m": {5: self._annotation("spurious", ["GO:B"])},
        }

        mp, target, feat, scale = build_pooled_sweep_rows(
            per_model_sweep_results, per_model_verified
        )

        assert len(mp) == 2
        assert list(scale) == ["esm2_8m", "esm2_35m"]
        # raw feature_idx collides (both 5) but scale labels disambiguate them
        assert list(feat) == [5, 5]
        assert list(target) == [1.0, 0.0]

    def test_output_feeds_directly_into_pooled_fit(self):
        """End-to-end: build_pooled_sweep_rows' output shape/dtype must work
        directly as fit_pooled_scale_interaction's input without adaptation.

        Status is drawn probabilistically as a function of min_positives
        (not a fixed i%2/i%3 pattern independent of mp) -- a deterministic,
        mp-independent target produced a near-perfectly-separated per-scale
        design, which made statsmodels' cluster-robust sandwich covariance
        numerically negative on one contrast (NaN SE) purely as a synthetic-
        fixture artifact, unrelated to what this test actually checks
        (shape/dtype compatibility). Caught via external codebase audit,
        2026-08-27, after _extract_per_scale_slopes was given a NaN-SE
        guard that (correctly) started rejecting this fixture."""
        from biolens.eval.statistics import build_pooled_sweep_rows, fit_pooled_scale_interaction

        rng = np.random.default_rng(0)
        mps = [10, 20, 50, 100]

        def make_verified(intercept, slope, n=15):
            # status is a property of the feature (not of mp) in the real
            # registry -- draw it once per feature, probabilistically, using
            # the mid-sweep mp for a representative probability.
            p_mid = 1 / (1 + np.exp(-(intercept + slope * np.log(mps[len(mps) // 2]))))
            verified = {}
            for i in range(n):
                status = "confirmed" if rng.random() < p_mid else "spurious"
                verified[i] = self._annotation(status, [f"GO:{i}"])
            return verified

        per_model_sweep_results = {
            "esm2_8m": {
                mp: [{"go_id": f"GO:{i}", "best_feature_idx": i} for i in range(15)]
                for mp in mps
            },
            "esm2_35m": {
                mp: [{"go_id": f"GO:{i}", "best_feature_idx": i} for i in range(15)]
                for mp in mps
            },
        }
        per_model_verified = {
            "esm2_8m": make_verified(intercept=-0.5, slope=0.3),
            "esm2_35m": make_verified(intercept=-0.3, slope=0.15),
        }

        mp, target, feat, scale = build_pooled_sweep_rows(
            per_model_sweep_results, per_model_verified
        )
        result = fit_pooled_scale_interaction(mp, target, feat, scale, random_state=1, n_bootstrap=0)
        assert result.n_rows == 120  # 2 models x 15 features x 4 sweep points
        assert set(result.scales) == {"esm2_8m", "esm2_35m"}

    def test_nan_cluster_robust_se_raises_instead_of_silently_propagating(self):
        """Regression test for a real robustness gap (external codebase
        audit, 2026-08-27): a near-perfectly-separated per-scale design can
        make statsmodels' cluster-robust sandwich covariance numerically
        negative on one contrast, so `sqrt(negative)` silently yields a NaN
        SE (and a NaN CI/p-value) instead of raising. No real pooled fit in
        this project has ever hit this (checked: all real saved outputs are
        finite), but a future run on a genuinely degenerate design should
        fail loudly rather than silently publish a NaN interval. Uses the
        deterministic, mp-independent i%2/i%3 status assignment this same
        test class's sibling test used before it was fixed to avoid exactly
        this degenerate design -- confirmed via the RuntimeWarning
        statsmodels itself emitted on this input before the guard existed."""
        from biolens.eval.statistics import build_pooled_sweep_rows, fit_pooled_scale_interaction

        per_model_sweep_results = {
            "esm2_8m": {
                mp: [{"go_id": f"GO:{i}", "best_feature_idx": i} for i in range(15)]
                for mp in [10, 20, 50, 100]
            },
            "esm2_35m": {
                mp: [{"go_id": f"GO:{i}", "best_feature_idx": i} for i in range(15)]
                for mp in [10, 20, 50, 100]
            },
        }
        per_model_verified = {
            "esm2_8m": {
                i: self._annotation("confirmed" if i % 2 == 0 else "spurious", [f"GO:{i}"])
                for i in range(15)
            },
            "esm2_35m": {
                i: self._annotation("confirmed" if i % 3 == 0 else "spurious", [f"GO:{i}"])
                for i in range(15)
            },
        }

        mp, target, feat, scale = build_pooled_sweep_rows(
            per_model_sweep_results, per_model_verified
        )
        with pytest.raises(ValueError, match="NaN"):
            fit_pooled_scale_interaction(mp, target, feat, scale, random_state=1, n_bootstrap=0)


class TestApplyMultiplicityCorrection:
    """Real gap found by external methodological review, 2026-08-05: no
    FDR/Bonferroni correction existed anywhere in the codebase despite
    testing multiple loci (26 in the current dataset, hundreds planned) in
    the same run."""

    def test_bh_correction_matches_statsmodels_directly(self):
        """The whole point of this function is to be a thin, correct wrapper
        around statsmodels' own tested implementation -- cross-check against
        calling multipletests directly."""
        from statsmodels.stats.multitest import multipletests

        from biolens.eval.statistics import apply_multiplicity_correction

        p_values = [0.001, 0.02, 0.03, 0.4, 0.5, 0.6, 0.8, 0.9]
        result = apply_multiplicity_correction(p_values, method="fdr_bh", alpha=0.05)

        expected_rejected, expected_adjusted, _, _ = multipletests(
            p_values, alpha=0.05, method="fdr_bh"
        )
        assert list(result.rejected) == list(expected_rejected)
        assert result.adjusted_p_values == pytest.approx(expected_adjusted)

    def test_all_null_p_values_mostly_survive_uncorrected(self):
        """A family of large, non-significant p-values should mostly remain
        non-significant after BH correction (BH is less conservative than
        Bonferroni, but a set of genuinely null p-values shouldn't manufacture
        significant results out of correction alone)."""
        from biolens.eval.statistics import apply_multiplicity_correction

        rng = np.random.default_rng(0)
        p_values = rng.uniform(0.3, 1.0, size=30)  # nowhere near significant
        result = apply_multiplicity_correction(p_values, alpha=0.05)

        assert result.n_significant == 0
        assert not result.rejected.any()

    def test_strong_signal_across_all_tests_survives_correction(self):
        """A family where every single test is genuinely, strongly
        significant should mostly survive BH correction (this is the real
        Geuvadis use case -- ADE/total-effect p-values, which are near-zero
        at every locus, should not get wiped out by correcting across ~30
        loci)."""
        from biolens.eval.statistics import apply_multiplicity_correction

        p_values = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4] * 6  # 30 strongly significant tests
        result = apply_multiplicity_correction(p_values, alpha=0.05)

        assert result.n_significant == 30
        assert result.rejected.all()

    def test_adjusted_p_values_never_smaller_than_raw(self):
        """BH-adjusted p-values must be >= the raw p-values, by construction
        of the correction -- a violation here would mean the wrapper is
        misusing statsmodels' output."""
        from biolens.eval.statistics import apply_multiplicity_correction

        rng = np.random.default_rng(1)
        p_values = rng.uniform(0, 1, size=50)
        result = apply_multiplicity_correction(p_values)

        assert np.all(result.adjusted_p_values >= result.raw_p_values - 1e-12)

    def test_empty_input_raises(self):
        from biolens.eval.statistics import apply_multiplicity_correction

        with pytest.raises(ValueError, match="zero p-values"):
            apply_multiplicity_correction([])

    def test_n_tests_and_n_significant_counts(self):
        from biolens.eval.statistics import apply_multiplicity_correction

        p_values = [1e-10, 1e-10, 1e-10, 0.9, 0.95]
        result = apply_multiplicity_correction(p_values, alpha=0.05)

        assert result.n_tests == 5
        assert result.n_significant == int(result.rejected.sum())

    def test_str_output(self):
        from biolens.eval.statistics import apply_multiplicity_correction

        result = apply_multiplicity_correction([0.001, 0.5, 0.6], alpha=0.05)
        s = str(result)
        assert "fdr_bh" in s
        assert "3 tests" in s

    def test_bonferroni_method_also_supported(self):
        """method= is passed straight through to statsmodels -- confirm a
        non-default method (Bonferroni, stricter FWER control) also works,
        not just the default fdr_bh."""
        from statsmodels.stats.multitest import multipletests

        from biolens.eval.statistics import apply_multiplicity_correction

        p_values = [0.001, 0.01, 0.02, 0.3, 0.9]
        result = apply_multiplicity_correction(p_values, method="bonferroni", alpha=0.05)
        _, expected_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="bonferroni")
        assert result.adjusted_p_values == pytest.approx(expected_adjusted)
        assert result.method == "bonferroni"


class TestTostEquivalenceTest:
    """Real gap found by external review, 2026-08-29: the Geuvadis case
    study's headline equivalence claim (abstract/§11: mean mediated fraction
    equivalence-bounded within +/-10% of zero, z=3.79, p=0.00008) had no
    function anywhere in the codebase computing it -- a grep for "tost" and
    "equivalence" across every .py file, tracked and untracked, returned
    zero hits before tost_equivalence_test was added."""

    def test_reproduces_the_geuvadis_headline_number_exactly(self):
        """The actual 18 reportable naive_proportion_mediated values from
        the Geuvadis case study results, hardcoded here so this test
        doesn't depend on that file's continued existence/content -- pinning
        the exact numbers this project's results cite (z=3.79, p=0.00008,
        95% CI +/-5.2%), independently reproduced by hand before this
        function existed."""
        from biolens.eval.statistics import tost_equivalence_test

        values = [
            -0.0534, 0.0309, -0.236, 0.0095, 0.0049, -0.011, -0.0231, -0.1586,
            0.118, 0.0146, 0.0123, 0.2682, 0.0025, 0.0769, 0.0169, 0.0799,
            -0.1686, 0.0177,
        ]
        result = tost_equivalence_test(values, margin=0.10)

        assert result.n == 18
        assert result.mean == pytest.approx(0.0, abs=1e-3)  # near-zero, values rounded to 4dp
        assert min(result.z_lower, result.z_upper) == pytest.approx(3.79, abs=0.01)
        assert result.p_value == pytest.approx(0.00008, abs=1e-5)
        assert result.ci95[0] == pytest.approx(-0.0516, abs=1e-3)
        assert result.ci95[1] == pytest.approx(0.0518, abs=1e-3)
        assert result.equivalent_at_alpha[0.05] is True

    def test_mean_far_outside_margin_is_not_equivalent(self):
        from biolens.eval.statistics import tost_equivalence_test

        values = [0.5, 0.6, 0.55, 0.52, 0.58]  # nowhere near zero
        result = tost_equivalence_test(values, margin=0.10)

        assert result.equivalent_at_alpha[0.05] is False
        assert result.p_value > 0.05

    def test_binding_p_value_is_the_larger_one_sided_p(self):
        """TOST convention: equivalence requires BOTH one-sided nulls
        rejected, so the reported p-value must be the max (the weaker,
        binding test), not the min."""
        from biolens.eval.statistics import tost_equivalence_test

        rng = np.random.default_rng(3)
        values = rng.normal(loc=0.02, scale=0.03, size=25)  # asymmetric around 0
        result = tost_equivalence_test(values, margin=0.10)

        assert result.p_value == pytest.approx(max(result.p_lower, result.p_upper))

    def test_symmetric_values_give_symmetric_ci_around_mean(self):
        from biolens.eval.statistics import tost_equivalence_test

        values = [-0.01, -0.005, 0.0, 0.005, 0.01]
        result = tost_equivalence_test(values, margin=0.10)

        width_lower = result.mean - result.ci95[0]
        width_upper = result.ci95[1] - result.mean
        assert width_lower == pytest.approx(width_upper)

    def test_zero_margin_raises(self):
        from biolens.eval.statistics import tost_equivalence_test

        with pytest.raises(ValueError, match="margin"):
            tost_equivalence_test([0.01, 0.02, -0.01], margin=0.0)

    def test_negative_margin_raises(self):
        from biolens.eval.statistics import tost_equivalence_test

        with pytest.raises(ValueError, match="margin"):
            tost_equivalence_test([0.01, 0.02, -0.01], margin=-0.05)

    def test_single_value_raises(self):
        """SE is undefined with fewer than 2 observations."""
        from biolens.eval.statistics import tost_equivalence_test

        with pytest.raises(ValueError, match="at least 2"):
            tost_equivalence_test([0.01], margin=0.10)

    def test_str_output(self):
        from biolens.eval.statistics import tost_equivalence_test

        result = tost_equivalence_test([-0.01, 0.0, 0.01, -0.005, 0.005], margin=0.10)
        s = str(result)
        assert "TOST" in s
        assert "10.00%" in s


class TestHanleyMcNeilAurocVariance:
    """Cross-checked against a direct Monte Carlo simulation of the actual
    production AUROC function (biolens.eval.probing._vectorized_auroc),
    not just internal consistency of the closed-form formula against
    itself -- the same discipline used throughout this project's stats
    code (see winners_curse_simulation.py's module docstring)."""

    def test_matches_monte_carlo_simulation_at_null(self):
        """Deliberately uses raw sklearn.roc_auc_score, NOT
        biolens.eval.probing._vectorized_auroc -- the production function
        reflects AUROC < 0.5 up to [0.5, 1.0] (see its docstring), which
        would fold this exact distribution and understate its variance
        relative to what the (unreflected) Hanley-McNeil formula predicts.
        gaussian_max_inflation_prediction's own `folded=` argument is where
        that reflection is accounted for, separately and explicitly."""
        from sklearn.metrics import roc_auc_score

        from biolens.eval.statistics import hanley_mcneil_auroc_variance

        rng = np.random.default_rng(0)
        n_pos, n_neg = 50, 950
        n_total = n_pos + n_neg
        n_trials = 3000

        labels = np.zeros(n_total, dtype=int)
        labels[:n_pos] = 1
        aurocs = np.empty(n_trials)
        for t in range(n_trials):
            scores = rng.standard_normal(n_total)
            aurocs[t] = roc_auc_score(labels, scores)

        empirical_var = float(np.var(aurocs))
        predicted_var = hanley_mcneil_auroc_variance(0.5, n_pos, n_neg)

        # Monte Carlo variance estimate itself has sampling noise; a loose
        # relative tolerance confirms the formula, not exact equality.
        assert empirical_var == pytest.approx(predicted_var, rel=0.15)

    def test_symmetric_at_auc_half(self):
        """At AUC=0.5, Q1 == Q2, so the formula must be symmetric under
        swapping n_positive and n_negative."""
        from biolens.eval.statistics import hanley_mcneil_auroc_variance

        v1 = hanley_mcneil_auroc_variance(0.5, 30, 500)
        v2 = hanley_mcneil_auroc_variance(0.5, 500, 30)
        assert v1 == pytest.approx(v2)

    def test_variance_shrinks_as_n_positive_grows(self):
        from biolens.eval.statistics import hanley_mcneil_auroc_variance

        variances = [
            hanley_mcneil_auroc_variance(0.5, n, 10000 - n)
            for n in (10, 100, 1000, 5000)
        ]
        assert variances == sorted(variances, reverse=True)


class TestFitRandomEffectsMeta:
    """Cross-checked against an independent hand-derivation during external
    review (2026-08): standard DerSimonian-Laird pooling of the four ESM2
    scales' AUROC-inflation-vs-n and permuted-inflation-vs-n slopes,
    reproducing Q/I^2/tau^2/pooled-CI to 4 decimal places against numbers
    computed completely independently outside this codebase."""

    def test_matches_independently_verified_real_data_numbers(self):
        """Route 1 (AUROC, real-label) from the experiment_auroc_
        inflation_vs_n.json results' per_model_fits, as of the 2026-07-30
        run. Target numbers independently re-derived from first principles
        during external review, 2026-08, and matched to 4 decimals."""
        from biolens.eval.statistics import fit_random_effects_meta

        thetas = [-0.8398398196469706, -0.8383409142708652, -0.7536480358971068, -0.4555700611362833]
        # SEs back-derived from each model's own 95% CI (t(.975, df=4), n_points=6)
        cis = [
            (-1.4152943658295376, -0.26438527346440366),
            (-1.3821310250431047, -0.2945508034986257),
            (-1.2491893507975507, -0.25810672099666304),
            (-0.7381358288068831, -0.17300429346568347),
        ]
        t_crit = 2.7764  # t(.975, df=4)
        ses = [(hi - lo) / 2 / t_crit for lo, hi in cis]

        result = fit_random_effects_meta(thetas, ses)

        assert result.theta_re == pytest.approx(-0.6747, abs=1e-3)
        assert result.ci[0] == pytest.approx(-0.8933, abs=1e-3)
        assert result.ci[1] == pytest.approx(-0.4561, abs=1e-3)
        assert result.q_statistic == pytest.approx(5.5346, abs=1e-3)
        assert result.i_squared == pytest.approx(45.8, abs=0.1)
        # The random-effects CI covers the theoretical -0.5; the naive
        # row-level pooled fit (-0.722, [-0.919, -0.524]) does not -- the
        # entire point of this estimator.
        assert result.ci[0] <= -0.5 <= result.ci[1]

        # Knapp-Hartung small-sample correction: at k=4 with real
        # heterogeneity (tau^2>0 here), a 40k-replicate Monte Carlo on this
        # exact SE vector (external review, 2026-08, independently
        # reproduced) found the normal-quantile `ci` undercovers (~90% actual
        # for a nominal 95%), while `ci_kh` stays close to nominal (~94%).
        # `ci_kh` must be WIDER here and still cover -0.5. (Numbers below
        # reflect the q_kh>=1.0 floor added 2026-08-27 -- a separate real
        # bug caught by external audit: this route's own q_kh is 0.809,
        # below 1, so flooring widens ci_kh further than the unfloored
        # value this test originally pinned.)
        assert result.ci_kh[0] == pytest.approx(-1.0296, abs=1e-3)
        assert result.ci_kh[1] == pytest.approx(-0.3198, abs=1e-3)
        assert (result.ci_kh[1] - result.ci_kh[0]) > (result.ci[1] - result.ci[0])
        assert result.ci_kh[0] <= -0.5 <= result.ci_kh[1]

    def test_zero_heterogeneity_reduces_to_fixed_effect(self):
        """Route 2 (permuted-label) real data: Q < k-1, so tau^2 is floored
        at 0 and the random-effects estimate must exactly equal the
        fixed-effect (inverse-variance-only) estimate."""
        from biolens.eval.statistics import fit_random_effects_meta

        thetas = [-0.5067395455460497, -0.7715164528222567, -0.5109737178332189, -0.4888590047428928]
        cis = [
            (-0.7620212728631736, -0.2514578182289259),
            (-1.3298440351238454, -0.21318887052066793),
            (-0.7689270135716585, -0.25302042209477926),
            (-0.7274975971788922, -0.2502204123068934),
        ]
        t_crit = 2.7764
        ses = [(hi - lo) / 2 / t_crit for lo, hi in cis]

        result = fit_random_effects_meta(thetas, ses)

        assert result.tau_squared == 0.0
        assert result.i_squared == 0.0
        assert result.theta_re == pytest.approx(result.theta_fixed_effect, abs=1e-10)
        assert result.theta_re == pytest.approx(-0.5185, abs=1e-3)

        # q_kh here is 0.575 (studies agree more than their own variances
        # predict) -- floored at 1.0 (external audit, 2026-08: unfloored,
        # this used to NARROW ci_kh below the normal-quantile ci, the
        # opposite of Knapp-Hartung's purpose; a prior version of this test
        # asserted that narrowed interval as expected behavior, which was
        # itself the bug). With the floor, ci_kh must be at least as wide
        # as ci, unconditionally.
        assert result.ci_kh[0] == pytest.approx(-0.6787, abs=1e-3)
        assert result.ci_kh[1] == pytest.approx(-0.3582, abs=1e-3)
        assert (result.ci_kh[1] - result.ci_kh[0]) >= (result.ci[1] - result.ci[0])
        assert result.ci_kh[0] <= -0.5 <= result.ci_kh[1]

    def test_identical_studies_recover_the_shared_value_with_shrunk_se(self):
        """k identical (theta, se) pairs: theta_re must equal that shared
        theta exactly, and pooling must strictly shrink the SE relative to
        any single study's own SE (the basic point of pooling)."""
        from biolens.eval.statistics import fit_random_effects_meta

        result = fit_random_effects_meta([1.5, 1.5, 1.5, 1.5], [0.2, 0.2, 0.2, 0.2])

        assert result.theta_re == pytest.approx(1.5)
        assert result.q_statistic == pytest.approx(0.0, abs=1e-9)
        assert result.tau_squared == 0.0
        assert result.se_re < 0.2

    def test_more_heterogeneous_studies_widen_the_ci(self):
        """Holding each study's own SE fixed, spreading the point estimates
        further apart must increase tau^2 and widen the pooled CI relative
        to the homogeneous case -- heterogeneity is not silently absorbed."""
        from biolens.eval.statistics import fit_random_effects_meta

        homogeneous = fit_random_effects_meta([1.0, 1.05, 0.95, 1.0], [0.15, 0.15, 0.15, 0.15])
        heterogeneous = fit_random_effects_meta([1.0, 2.0, 0.0, 1.5], [0.15, 0.15, 0.15, 0.15])

        assert heterogeneous.tau_squared > homogeneous.tau_squared
        homog_width = homogeneous.ci[1] - homogeneous.ci[0]
        heterog_width = heterogeneous.ci[1] - heterogeneous.ci[0]
        assert heterog_width > homog_width

    def test_rejects_mismatched_lengths(self):
        from biolens.eval.statistics import fit_random_effects_meta

        with pytest.raises(ValueError):
            fit_random_effects_meta([1.0, 2.0], [0.1, 0.2, 0.3])

    def test_rejects_fewer_than_two_studies(self):
        from biolens.eval.statistics import fit_random_effects_meta

        with pytest.raises(ValueError):
            fit_random_effects_meta([1.0], [0.1])

    def test_rejects_nonpositive_se(self):
        from biolens.eval.statistics import fit_random_effects_meta

        with pytest.raises(ValueError):
            fit_random_effects_meta([1.0, 2.0], [0.1, 0.0])

    def test_knapp_hartung_q_is_floored_at_one(self):
        """Regression test for a real bug (external codebase audit, 2026-08):
        q_kh (the weighted dispersion of studies around theta_re) was not
        floored at 1.0, so when studies happen to agree more than their own
        variances predict (q_kh<1), se_kh came out SMALLER than se_re --
        narrowing the interval, the opposite of Knapp-Hartung's entire
        purpose as a conservative small-k correction. Constructs 4 studies
        with identical point estimates (so q_kh is exactly 0, the most
        extreme under-1 case possible) and confirms se_kh is not shrunk
        below se_re."""
        from biolens.eval.statistics import fit_random_effects_meta

        result = fit_random_effects_meta([2.0, 2.0, 2.0, 2.0], [0.3, 0.25, 0.2, 0.35])

        assert result.se_kh >= result.se_re
        ci_width = result.ci[1] - result.ci[0]
        ci_kh_width = result.ci_kh[1] - result.ci_kh[0]
        assert ci_kh_width >= ci_width

    def test_str_output_includes_key_fields(self):
        from biolens.eval.statistics import fit_random_effects_meta

        result = fit_random_effects_meta([1.0, 1.2, 0.8, 1.1], [0.2, 0.25, 0.15, 0.22])
        s = str(result)
        assert "k=4" in s
        assert "I^2" in s
        assert "95% CI" in s
        assert "Knapp-Hartung" in s


class TestGaussianMaxInflationPrediction:
    def test_scales_linearly_with_sigma(self):
        from biolens.eval.statistics import gaussian_max_inflation_prediction

        base = gaussian_max_inflation_prediction(sigma=1.0, d=2560)
        doubled = gaussian_max_inflation_prediction(sigma=2.0, d=2560)
        assert doubled == pytest.approx(2 * base)

    def test_increases_with_d(self):
        from biolens.eval.statistics import gaussian_max_inflation_prediction

        small_d = gaussian_max_inflation_prediction(sigma=0.1, d=10)
        large_d = gaussian_max_inflation_prediction(sigma=0.1, d=10000)
        assert large_d > small_d

    def test_folded_equals_unfolded_at_double_the_candidates(self):
        """Regression test for a real bug (external codebase audit, 2026-08):
        an earlier version rescaled sigma by the folded-normal SD factor
        sqrt(1-2/pi) for folded=True -- a category error, since extreme-value
        scale is governed by tail behavior, not by the folded variable's
        overall variance. max_i|eps_i| over d draws equals exactly the max
        over the 2d signed values {eps_1,-eps_1,...,eps_d,-eps_d}, and a
        half-normal's tail is exactly twice a normal's tail at the same y --
        so reflection corresponds to doubling the effective sample count in
        the max-of-iid-Gaussians asymptotic, not to shrinking sigma. This is
        now implemented directly: folded=True must equal folded=False
        evaluated at exactly 2*d, with NO other transformation."""
        from biolens.eval.statistics import gaussian_max_inflation_prediction

        sigma, d = 0.7, 2560
        folded = gaussian_max_inflation_prediction(sigma, d, folded=True)
        unfolded_at_2d = gaussian_max_inflation_prediction(sigma, 2 * d, folded=False)
        assert folded == pytest.approx(unfolded_at_2d)

        # And it must NOT match the old (wrong) sigma-rescaling formula --
        # a regression back to that bug must fail this test.
        import math
        old_wrong_formula = (
            sigma * math.sqrt(1 - 2 / math.pi) * math.sqrt(2 * math.log(d))
        )
        assert folded != pytest.approx(old_wrong_formula, rel=1e-3)

    def test_matches_monte_carlo_simulation_of_reflected_max(self):
        """The decisive correctness check: simulate max_i|eps_i| directly
        (exactly what biolens.eval.probing._vectorized_auroc's reflection
        step computes under the null) and confirm the folded prediction
        lands close to it -- not just internally self-consistent, but
        actually correct for the quantity it claims to predict. The
        superseded formula (sigma rescaled by sqrt(1-2/pi)) underestimated
        this simulated value by roughly 35% at realistic (sigma, d); the
        current formula should be within a few percent."""
        from biolens.eval.statistics import gaussian_max_inflation_prediction

        rng = np.random.default_rng(3)
        sigma, d = 0.09179, 2560
        n_reps = 8000
        maxes = np.array([
            np.abs(rng.normal(0, sigma, size=d)).max() for _ in range(n_reps)
        ])
        empirical = maxes.mean()
        predicted = gaussian_max_inflation_prediction(sigma, d, folded=True)

        assert predicted == pytest.approx(empirical, rel=0.05)


class TestResolveConsensus:
    def test_agreement_returns_the_shared_label(self):
        from biolens.eval.statistics import resolve_consensus

        label, reason = resolve_consensus("confirmed", "high", "confirmed", "medium")
        assert label == "confirmed"
        assert reason == "agreement"

    def test_higher_confidence_breaks_the_tie(self):
        from biolens.eval.statistics import resolve_consensus

        label, reason = resolve_consensus("confirmed", "high", "spurious", "low")
        assert label == "confirmed"
        assert "A's higher confidence" in reason

        label, reason = resolve_consensus("confirmed", "low", "spurious", "high")
        assert label == "spurious"
        assert "B's higher confidence" in reason

    def test_both_high_confidence_disagreement_is_excluded_not_broken(self):
        """The pre-registered protocol is explicit: nobody unilaterally
        breaks a high-vs-high disagreement -- that would reintroduce the
        non-independence problem two annotators exist to solve."""
        from biolens.eval.statistics import resolve_consensus

        label, reason = resolve_consensus("confirmed", "high", "spurious", "high")
        assert label is None
        assert "excluded" in reason

    def test_tied_non_high_confidence_disagreement_is_also_excluded(self):
        """Extension beyond the literal both-high case in the written
        protocol, for the identical stated reason -- see
        resolve_consensus's docstring."""
        from biolens.eval.statistics import resolve_consensus

        label, reason = resolve_consensus("plausible", "medium", "spurious", "medium")
        assert label is None
        label, reason = resolve_consensus("confirmed", "low", "plausible", "low")
        assert label is None


class TestComputeAgreement:
    def test_perfect_agreement(self):
        from biolens.eval.statistics import compute_agreement

        labels = ["confirmed", "plausible", "spurious", "spurious"]
        confidences = ["high", "medium", "high", "low"]
        result = compute_agreement(labels, confidences, labels, confidences)

        assert result.percent_agreement == 1.0
        assert result.percent_agreement_on_scale == 1.0
        assert result.n_excluded_tied_disagreement == 0

    def test_weighted_kappa_penalizes_far_disagreements_more(self):
        """confirmed-vs-spurious (ordinal distance 2) should hurt the
        weighted kappa more than confirmed-vs-plausible (distance 1) --
        the entire point of using a WEIGHTED kappa on quasi-ordinal
        categories instead of unweighted. Both
        sequences need real variability in both raters' marginals -- Cohen's
        kappa is degenerately 0 whenever either rater's labels are constant
        (no variability to correct for chance agreement against), so a
        constant-label test wouldn't actually exercise this property."""
        from biolens.eval.statistics import compute_agreement

        labels_a = (
            ["confirmed"] * 5 + ["spurious"] * 5 + ["plausible"] * 5
        )  # real variability in A: 5/5/5 split
        # Both near and far agree on 13/15 items; only the two disagreements'
        # ordinal DISTANCE differs between them, isolating the effect.
        labels_b_near = (
            ["confirmed"] * 4 + ["plausible"]       # confirmed[4] -> plausible (distance 1)
            + ["spurious"] * 4 + ["plausible"]       # spurious[4] -> plausible (distance 1)
            + ["plausible"] * 5
        )
        labels_b_far = (
            ["confirmed"] * 4 + ["spurious"]         # confirmed[4] -> spurious (distance 2)
            + ["spurious"] * 4 + ["confirmed"]        # spurious[4] -> confirmed (distance 2)
            + ["plausible"] * 5
        )
        confidences = ["high"] * 15

        near = compute_agreement(labels_a, confidences, labels_b_near, confidences)
        far = compute_agreement(labels_a, confidences, labels_b_far, confidences)

        assert near.percent_agreement == far.percent_agreement  # same # of disagreements
        assert near.weighted_kappa is not None and far.weighted_kappa is not None
        assert far.weighted_kappa < near.weighted_kappa

    def test_unverifiable_excluded_from_on_scale_metrics(self):
        from biolens.eval.statistics import compute_agreement

        labels_a = ["confirmed", "unverifiable", "spurious", "plausible"]
        labels_b = ["confirmed", "spurious", "spurious", "plausible"]
        confidences = ["high", "medium", "high", "high"]

        result = compute_agreement(labels_a, confidences, labels_b, confidences)

        assert result.n_items == 4
        assert result.n_unverifiable_by_either == 1
        # on-scale agreement computed only over the 3 non-unverifiable-involving items
        assert result.percent_agreement_on_scale == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self):
        from biolens.eval.statistics import compute_agreement

        with pytest.raises(ValueError, match="same length"):
            compute_agreement(["confirmed"], ["high"], ["confirmed", "spurious"], ["high", "low"])

    def test_empty_input_raises(self):
        from biolens.eval.statistics import compute_agreement

        with pytest.raises(ValueError, match="zero items"):
            compute_agreement([], [], [], [])

    def test_single_category_present_gives_none_kappa_not_a_crash(self):
        """cohen_kappa_score is undefined with <2 distinct categories in
        play -- must degrade to None, not raise or silently return a
        meaningless number."""
        from biolens.eval.statistics import compute_agreement

        labels = ["spurious"] * 5
        confidences = ["high"] * 5
        result = compute_agreement(labels, confidences, labels, confidences)
        assert result.weighted_kappa is None
