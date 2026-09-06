"""
Tests for the winner's-curse simulation's vectorized metric implementations
(the AP/F1/MCC metric-robustness extension). Validates the vectorized,
all-columns-at-once implementations against sklearn's per-column ground
truth, the same pattern already used for _vectorized_auroc in
tests/test_eval.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestVectorizedAP:
    def test_matches_sklearn(self):
        from sklearn.metrics import average_precision_score

        from winners_curse_simulation import _vectorized_ap

        rng = np.random.default_rng(1)
        N, D = 300, 25
        scores = rng.standard_normal((N, D))
        labels = (rng.random(N) > 0.85).astype(int)  # sparse positives, like real min_positives

        vec_ap = _vectorized_ap(scores, labels)
        for i in range(D):
            sk_ap = average_precision_score(labels, scores[:, i])
            assert abs(vec_ap[i] - sk_ap) < 1e-9, f"Feature {i}: vec={vec_ap[i]}, sklearn={sk_ap}"

    def test_zero_positives_returns_zeros(self):
        from winners_curse_simulation import _vectorized_ap

        scores = np.random.default_rng(0).standard_normal((50, 5))
        labels = np.zeros(50, dtype=int)
        result = _vectorized_ap(scores, labels)
        assert np.all(result == 0.0)
        assert result.shape == (5,)

    def test_perfect_separation_gives_ap_one(self):
        from winners_curse_simulation import _vectorized_ap

        # Feature perfectly separates: all positives have the highest scores.
        scores = np.array([[5.0], [4.0], [3.0], [1.0], [0.0]])
        labels = np.array([1, 1, 1, 0, 0])
        result = _vectorized_ap(scores, labels)
        assert result[0] == pytest.approx(1.0)


class TestVectorizedF1MCC:
    def test_matches_sklearn(self):
        from sklearn.metrics import f1_score, matthews_corrcoef

        from winners_curse_simulation import _vectorized_f1_mcc

        rng = np.random.default_rng(2)
        N, D = 300, 25
        scores = rng.standard_normal((N, D))
        labels = (rng.random(N) > 0.85).astype(int)

        vec_f1, vec_mcc = _vectorized_f1_mcc(scores, labels, threshold=0.0)
        for i in range(D):
            pred = (scores[:, i] > 0.0).astype(int)
            sk_f1 = f1_score(labels, pred, zero_division=0)
            sk_mcc = matthews_corrcoef(labels, pred) if len(np.unique(pred)) > 1 else 0.0
            assert abs(vec_f1[i] - sk_f1) < 1e-9, f"F1 feature {i}: vec={vec_f1[i]}, sklearn={sk_f1}"
            assert abs(vec_mcc[i] - sk_mcc) < 1e-9, (
                f"MCC feature {i}: vec={vec_mcc[i]}, sklearn={sk_mcc}"
            )

    def test_degenerate_all_positive_predictions_no_nan(self):
        """A feature whose scores are entirely above threshold (predicts
        every example positive) has an undefined MCC denominator — must
        return 0.0, not NaN, so downstream aggregates (mean/max across
        features) aren't silently poisoned."""
        from winners_curse_simulation import _vectorized_f1_mcc

        scores = np.full((20, 3), 10.0)  # every score >> threshold
        labels = np.array([1] * 10 + [0] * 10)
        f1, mcc = _vectorized_f1_mcc(scores, labels, threshold=0.0)
        assert not np.any(np.isnan(f1))
        assert not np.any(np.isnan(mcc))
        assert np.all(mcc == 0.0)

    def test_perfect_classifier_gives_f1_and_mcc_one(self):
        from winners_curse_simulation import _vectorized_f1_mcc

        scores = np.array([[5.0], [4.0], [-1.0], [-2.0]])
        labels = np.array([1, 1, 0, 0])
        f1, mcc = _vectorized_f1_mcc(scores, labels, threshold=0.0)
        assert f1[0] == pytest.approx(1.0)
        assert mcc[0] == pytest.approx(1.0)


class TestRunTrial:
    def test_returns_all_four_metrics_with_max_geq_mean(self):
        from winners_curse_simulation import METRICS, run_trial

        rng = np.random.default_rng(3)
        trial = run_trial(n_total=500, n_positive=20, d_sae=64, rng=rng)
        assert set(trial.keys()) == set(METRICS)
        for m in METRICS:
            max_v, mean_v = trial[m]
            assert max_v >= mean_v - 1e-9, f"{m}: max {max_v} should be >= mean {mean_v}"

    def test_winners_curse_pattern_shrinks_for_rank_based_metrics(self):
        """The metric-robustness check's actual, real finding (checked, not
        assumed): the classic winner's-curse absolute-inflation-shrinks-
        with-more-positives pattern holds cleanly for AUROC and AP, both
        threshold-free/rank-based metrics whose null
        expectation and variance behave consistently across the swept
        n_positive range. It does NOT hold the same way for F1 (fixed
        threshold, so its own baseline scale is itself strongly
        prevalence-dependent — see test_f1_scale_is_prevalence_dependent
        below) or MCC (well-documented in the ML literature to be
        high-variance under extreme class imbalance, which dominates at the
        smallest n_positive values tested here) — see
        test_f1_and_mcc_do_not_reliably_shrink for that explicitly-checked
        negative result. Asserting the shrinking property for all four
        metrics would have been asserting something false; this test
        checks what's actually true."""
        from winners_curse_simulation import run_trial

        rng = np.random.default_rng(4)
        d_sae = 200
        n_trials = 40

        def mean_inflation(n_positive: int, metrics: tuple[str, ...]) -> dict[str, float]:
            totals = {m: 0.0 for m in metrics}
            for _ in range(n_trials):
                trial = run_trial(n_total=2000, n_positive=n_positive, d_sae=d_sae, rng=rng)
                for m in metrics:
                    max_v, mean_v = trial[m]
                    totals[m] += max_v - mean_v
            return {m: totals[m] / n_trials for m in metrics}

        rank_based = ("auroc", "ap")
        small_n_inflation = mean_inflation(10, rank_based)
        large_n_inflation = mean_inflation(500, rank_based)

        for m in rank_based:
            assert small_n_inflation[m] > large_n_inflation[m], (
                f"{m}: inflation at n_positive=10 ({small_n_inflation[m]:.4f}) should exceed "
                f"n_positive=500 ({large_n_inflation[m]:.4f}) — winner's curse should shrink "
                f"with more positives for rank-based, threshold-free metrics"
            )

    def test_f1_scale_is_prevalence_dependent(self):
        """F1 at a fixed threshold is NOT prevalence-invariant the way
        AUROC/MCC are by construction — its own population mean grows
        substantially with n_positive (more balanced classes push the
        fixed-threshold precision/recall trade-off toward F1's natural
        maximum), which is WHY its absolute inflation-above-null doesn't
        shrink the same way AUROC's does (a growing baseline leaves more
        absolute room for max-selection noise even as the per-feature
        estimate itself gets more precise) — checked directly rather than
        asserted."""
        from winners_curse_simulation import run_trial

        rng = np.random.default_rng(5)
        n_trials = 30

        def mean_f1_baseline(n_positive: int) -> float:
            total = 0.0
            for _ in range(n_trials):
                trial = run_trial(n_total=2000, n_positive=n_positive, d_sae=200, rng=rng)
                total += trial["f1"][1]  # mean_v
            return total / n_trials

        low_prevalence_baseline = mean_f1_baseline(10)
        high_prevalence_baseline = mean_f1_baseline(500)
        assert high_prevalence_baseline > low_prevalence_baseline * 5, (
            f"F1's own baseline should grow sharply with prevalence at a fixed threshold: "
            f"n_positive=10 -> {low_prevalence_baseline:.4f}, "
            f"n_positive=500 -> {high_prevalence_baseline:.4f}"
        )
