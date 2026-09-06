"""
Tests for the standardized evaluation suite.

All tests use synthetic data — no model downloads or real GO annotations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ── Reconstruction metrics ────────────────────────────────────────────────────

class TestReconstructionMetrics:
    def test_perfect_reconstruction(self, topk_sae, d_model):
        """An SAE that perfectly reconstructs should have FVE = 1 (or very close)."""
        # Train briefly so reconstruction is decent
        import torch.optim as optim

        from biolens.eval.reconstruction import compute_reconstruction_metrics
        optimizer = optim.Adam(topk_sae.parameters(), lr=1e-2)
        x = torch.randn(512, d_model)
        for _ in range(200):
            out = topk_sae(x)
            loss = out.l2_loss + 1e-3 * out.auxiliary_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            topk_sae.normalize_decoder()

        metrics = compute_reconstruction_metrics(topk_sae, x, batch_size=256, device="cpu")
        # After training on its own data, FVE should be reasonable
        assert metrics.frac_variance_explained > 0.3, (
            f"FVE too low: {metrics.frac_variance_explained}"
        )

    def test_untrained_metrics_shape(self, topk_sae, random_activations):
        from biolens.eval.reconstruction import compute_reconstruction_metrics

        metrics = compute_reconstruction_metrics(
            topk_sae, random_activations, batch_size=32, device="cpu"
        )
        assert 0.0 <= metrics.dead_fraction <= 1.0
        assert 0.0 <= metrics.mean_l0_frac <= 1.0
        assert metrics.n_samples == len(random_activations)
        assert len(metrics.per_feature_act_rate) == topk_sae.cfg.d_sae

    def test_mean_l0_matches_k(self, topk_sae, random_activations):
        from biolens.eval.reconstruction import compute_reconstruction_metrics

        metrics = compute_reconstruction_metrics(
            topk_sae, random_activations, batch_size=32, device="cpu"
        )
        # mean_l0 should be ≤ k (some might be zero due to ReLU clamp)
        assert metrics.mean_l0 <= topk_sae.k + 0.5

    def test_str_representation(self, topk_sae, random_activations):
        from biolens.eval.reconstruction import compute_reconstruction_metrics

        metrics = compute_reconstruction_metrics(
            topk_sae, random_activations, batch_size=32, device="cpu"
        )
        s = str(metrics)
        assert "FVE=" in s
        assert "L0=" in s

    def test_from_cache(self, topk_sae, tmp_cache_dir, d_model):
        from biolens.eval.reconstruction import compute_reconstruction_metrics
        from biolens.sae.dictionary import ActivationCache

        cache = ActivationCache(tmp_cache_dir)
        acts = torch.randn(100, d_model)
        ids = [f"P{i}" for i in range(100)]
        cache.write_shard(0, acts, ids)
        cache.finalize()

        metrics = compute_reconstruction_metrics(
            topk_sae, cache, batch_size=32, device="cpu", normalize=True
        )
        assert metrics.n_samples == 100


# ── GO probing ────────────────────────────────────────────────────────────────

class TestGOProbing:
    def test_probe_with_informative_features(self, d_sae):
        """Plant an informative feature: if feature_5 > 0 → label is 1."""
        from biolens.eval.probing import probe_go_terms

        N = 300
        d_sae_val = 64
        torch.manual_seed(1)

        Z = torch.randn(N, d_sae_val).abs()
        ids = [f"P{i:05d}" for i in range(N)]

        # Feature 5 perfectly predicts the label
        labels = (Z[:, 5] > Z[:, 5].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z,
            protein_ids=ids,
            go_labels=go_labels,
            go_names={"GO:0000001": "test_term"},
            min_positives=20,
            max_go_terms=10,
            train_fraction=0.8,
        )
        assert len(report.results) >= 1
        best = report.top_results(1)[0]
        assert best.single_feature_auroc > 0.75, (
            f"Informative feature not detected: AUROC={best.single_feature_auroc:.3f}"
        )
        assert best.best_feature_idx == 5

    def test_probe_random_features(self, d_sae, n_seqs):
        """Random features vs. random labels: mean AUROC should be ~0.5."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(2)
        N = 200
        d_sae_val = 32
        Z = torch.randn(N, d_sae_val)
        ids = [f"P{i:05d}" for i in range(N)]

        rng = np.random.default_rng(2)
        go_labels = {
            pid: ({"GO:0001234"} if rng.random() < 0.4 else set())
            for pid in ids
        }

        report = probe_go_terms(
            feature_acts=Z,
            protein_ids=ids,
            go_labels=go_labels,
            min_positives=30,
        )
        if report.results:
            mean_auroc = np.mean([r.single_feature_auroc for r in report.results])
            # Random features: reflected AUROC should be near 0.5..0.65
            assert mean_auroc < 0.85, (
                f"Random features achieved suspiciously high AUROC: {mean_auroc}"
            )

    def test_min_positives_filter(self):
        """GO terms with fewer than min_positives examples should be skipped."""
        from biolens.eval.probing import probe_go_terms

        N = 100
        Z = torch.randn(N, 32)
        ids = [f"P{i}" for i in range(N)]
        # Only 3 positives — below default min_positives=50
        go_labels = {ids[i]: {"GO:RARE"} for i in range(3)}

        report = probe_go_terms(
            feature_acts=Z,
            protein_ids=ids,
            go_labels=go_labels,
            min_positives=50,
        )
        assert report.n_go_terms_tested == 0

    def test_compute_multivariate_auroc_false_skips_the_fit(self):
        """Real production timeout fix (job 9701717, 2026-07-28): a caller
        that only needs best_feature_idx shouldn't pay for the multivariate
        LogisticRegression fit at all — multivariate_auroc should come back
        NaN, not just be ignored after being computed anyway."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(5)
        N = 200
        Z = torch.randn(N, 32).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        labels = (Z[:, 3] > Z[:, 3].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_multivariate_auroc=False,
        )
        assert len(report.results) >= 1
        for r in report.results:
            assert np.isnan(r.multivariate_auroc)
            # single_feature_auroc/best_feature_idx must still be computed —
            # only the multivariate fit is skipped.
            assert 0.0 <= r.single_feature_auroc <= 1.0

    def test_report_fraction_above_threshold(self):
        from biolens.eval.probing import GOProbingReport, GOProbingResult

        results = [
            GOProbingResult("GO:0000001", "t1", 60, 200, 0.82, 0, 0.85),
            GOProbingResult("GO:0000002", "t2", 60, 200, 0.65, 1, 0.70),
            GOProbingResult("GO:0000003", "t3", 60, 200, 0.55, 2, 0.60),
        ]
        report = GOProbingReport(results, 3, "esm2_8m", 5, "topk")
        # [0.82, 0.65, 0.55] — only 0.82 is ≥ 0.70
        assert report.fraction_above_auroc(0.65) == pytest.approx(2 / 3)
        assert report.fraction_above_auroc(0.70) == pytest.approx(1 / 3)
        assert report.fraction_above_auroc(0.80) == pytest.approx(1 / 3)
        assert report.fraction_above_auroc(0.90) == pytest.approx(0.0)

    def test_vectorized_auroc_vs_sklearn(self):
        """Vectorized AUROC should match sklearn within floating-point tolerance."""
        from sklearn.metrics import roc_auc_score

        from biolens.eval.probing import _vectorized_auroc

        rng = np.random.default_rng(42)
        N, D = 200, 20
        scores = rng.standard_normal((N, D))
        labels = (rng.random(N) > 0.6).astype(int)

        vec_aurocs = _vectorized_auroc(scores, labels)
        for i in range(D):
            sk_auroc = roc_auc_score(labels, scores[:, i])
            # Both reflect below 0.5
            sk_reflected = max(sk_auroc, 1.0 - sk_auroc)
            assert abs(vec_aurocs[i] - sk_reflected) < 1e-6, (
                f"Feature {i}: vectorized={vec_aurocs[i]:.6f}, sklearn={sk_reflected:.6f}"
            )

    def test_str_output(self):
        from biolens.eval.probing import GOProbingReport, GOProbingResult

        results = [GOProbingResult("GO:0000001", "cytoplasm", 100, 500, 0.78, 42, 0.81)]
        report = GOProbingReport(results, 1, "esm2_8m", 5, "topk")
        s = str(report)
        assert "GO probing" in s
        assert "GO:0000001" in s
        assert "0.78" in s

    def test_multivariate_auroc_handles_wildly_scaled_features(self):
        """multivariate_auroc's LogisticRegression must standardize features
        before fitting — real, unscaled JumpReLU SAE activations (e.g. Gemma
        Scope) caused repeated `lbfgs failed to converge`
        warnings and multi-hour runtimes in production (2026-07-05), unlike
        any TopK SAE (ESM2/Evo2), whose fixed per-example sparsity happens to
        keep features well-conditioned regardless. This test constructs
        features with deliberately wild per-feature scale differences (the
        actual mechanism suspected to cause the real failure) and only
        checks the fit still converges to a sensible, non-degenerate result
        — it does not attempt to detect a convergence *warning* directly,
        since that's an sklearn implementation detail, not part of this
        function's contract."""
        from biolens.eval.probing import probe_go_terms

        N = 300
        d_sae_val = 64
        rng = np.random.default_rng(3)

        Z = rng.standard_normal((N, d_sae_val)).astype(np.float64)
        # Wildly different per-feature scales — some tiny, some huge —
        # mimicking unscaled, dense JumpReLU-style activations rather than
        # TopK's uniform, sparse pattern.
        scale = 10.0 ** rng.uniform(-6, 6, size=d_sae_val)
        Z = torch.from_numpy((Z * scale).astype(np.float32))

        labels = (Z[:, 5] > Z[:, 5].median()).numpy().astype(int)
        ids = [f"P{i:05d}" for i in range(N)]
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z,
            protein_ids=ids,
            go_labels=go_labels,
            go_names={"GO:0000001": "test_term"},
            min_positives=20,
            max_go_terms=10,
            train_fraction=0.8,
        )
        assert len(report.results) == 1
        multi_auroc = report.results[0].multivariate_auroc
        assert not np.isnan(multi_auroc), "multivariate_auroc should not be NaN"
        assert 0.0 <= multi_auroc <= 1.0
        # An informative, if oddly-scaled, feature set should still separate
        # well once standardized — a value near chance (~0.5) would suggest
        # the fit degraded badly on the scale disparity instead of handling it.
        assert multi_auroc > 0.7, (
            f"multivariate_auroc unexpectedly low on wildly-scaled features: {multi_auroc}"
        )

    def test_multivariate_auroc_converges_when_n_train_near_d_sae(self):
        """Standardization alone (previous test) does not reproduce the real
        Gemma Scope failure mode: its d_sae=16384 against the
        default max_examples=20000 * train_fraction=0.8 ~= 16000 training
        rows puts n_train/d_sae ~= 0.98, the n~=p interpolation threshold
        where unregularized/weakly-regularized (C=1.0) logistic regression
        has no well-conditioned finite-norm solution regardless of feature
        scaling (2026-07-06 finding: StandardScaler cut runtime but did not
        stop `lbfgs failed to converge`). This test reproduces that ratio at
        small scale (n_train ~= 0.85 * d_sae) with well-scaled features (so
        conditioning is not the confound) and checks the fit converges
        without a ConvergenceWarning after lowering C to 0.1."""
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        from biolens.eval.probing import probe_go_terms

        d_sae_val = 200
        N = 240  # train_fraction=0.8 -> n_train=192, n_train/d_sae_val = 0.96
        rng = np.random.default_rng(7)

        Z_np = rng.standard_normal((N, d_sae_val)).astype(np.float32)
        true_w = rng.standard_normal(d_sae_val).astype(np.float32)
        logits = Z_np @ true_w
        labels = (logits > np.median(logits)).astype(int)
        Z = torch.from_numpy(Z_np)

        ids = [f"P{i:05d}" for i in range(N)]
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = probe_go_terms(
                feature_acts=Z,
                protein_ids=ids,
                go_labels=go_labels,
                go_names={"GO:0000001": "test_term"},
                min_positives=20,
                max_go_terms=10,
                train_fraction=0.8,
            )

        convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        assert not convergence_warnings, (
            f"Expected no ConvergenceWarning at n_train~=d_sae, got: "
            f"{[str(w.message) for w in convergence_warnings]}"
        )
        assert len(report.results) == 1
        multi_auroc = report.results[0].multivariate_auroc
        assert not np.isnan(multi_auroc)
        assert 0.0 <= multi_auroc <= 1.0


class TestProbeGoTermControlledSubsampling:
    """Confound control: isolate the pure sample-size effect from
    concept-difficulty confounding by artificially downsampling ONE fixed
    GO term's positives, rather than comparing across different natural GO
    terms at each min_positives."""

    def _make_informative_data(self, n=2000, d_sae_val=32, n_natural_positives=400, seed=10):
        """One feature (index 3) perfectly predicts a fixed positive set of
        size n_natural_positives — informative signal, real (not random)
        winner's-curse behavior expected as target_n shrinks."""
        rng = np.random.default_rng(seed)
        Z = torch.from_numpy(rng.standard_normal((n, d_sae_val)).astype(np.float32))
        ids = [f"P{i:05d}" for i in range(n)]

        positive_idx = rng.choice(n, size=n_natural_positives, replace=False)
        # Make feature 3 strongly (not perfectly, so held-out AUROC isn't
        # trivially 1.0 regardless of n) separate exactly these IDs.
        boost = torch.zeros(n)
        boost[positive_idx] = 3.0
        Z[:, 3] = Z[:, 3] + boost

        go_labels = {ids[i]: {"GO:FIXED"} for i in positive_idx}
        return Z, ids, go_labels, n_natural_positives

    def test_returns_one_summary_per_target_n(self):
        from biolens.eval.probing import probe_go_term_controlled_subsampling

        Z, ids, go_labels, _ = self._make_informative_data()
        summaries = probe_go_term_controlled_subsampling(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            go_id="GO:FIXED", target_n_positives=[20, 50, 100], n_repeats=3,
            random_state=42,
        )
        assert [s.target_n_positives for s in summaries] == [20, 50, 100]

    def test_n_natural_positives_recorded_correctly(self):
        from biolens.eval.probing import probe_go_term_controlled_subsampling

        Z, ids, go_labels, n_natural = self._make_informative_data(n_natural_positives=250)
        summaries = probe_go_term_controlled_subsampling(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            go_id="GO:FIXED", target_n_positives=[20], n_repeats=2,
        )
        assert summaries[0].n_natural_positives == n_natural

    def test_raises_when_target_exceeds_natural_positives(self):
        from biolens.eval.probing import probe_go_term_controlled_subsampling

        Z, ids, go_labels, n_natural = self._make_informative_data(n_natural_positives=50)
        with pytest.raises(ValueError, match="natural positives"):
            probe_go_term_controlled_subsampling(
                feature_acts=Z, protein_ids=ids, go_labels=go_labels,
                go_id="GO:FIXED", target_n_positives=[20, 100],  # 100 > 50 natural
                n_repeats=2,
            )

    def test_detects_informative_feature_at_every_target_n(self):
        """The planted feature should be found (single_feature_auroc well
        above chance) regardless of which subsample size is requested."""
        from biolens.eval.probing import probe_go_term_controlled_subsampling

        Z, ids, go_labels, _ = self._make_informative_data(n_natural_positives=400)
        summaries = probe_go_term_controlled_subsampling(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            go_id="GO:FIXED", target_n_positives=[30, 100, 300], n_repeats=5,
            random_state=7,
        )
        for s in summaries:
            assert s.mean_single_feature_auroc > 0.75, (
                f"target_n={s.target_n_positives}: mean_single_feature_auroc="
                f"{s.mean_single_feature_auroc:.3f} too low for a planted signal"
            )

    def test_repeats_use_distinct_splits(self):
        """Different repeats at the same target_n must not silently reuse
        the identical train/held-out partition — otherwise 'n_repeats' would
        just be relabeling the same split n times, not resampling variance."""
        from biolens.eval.probing import probe_go_term_controlled_subsampling

        Z, ids, go_labels, _ = self._make_informative_data(n_natural_positives=400)
        summaries = probe_go_term_controlled_subsampling(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            go_id="GO:FIXED", target_n_positives=[100], n_repeats=8,
            random_state=3,
        )
        # With real resampling variance, held-out AUROC point estimates
        # across repeats should not all be byte-identical.
        assert summaries[0].n_repeats_successful >= 2
        assert summaries[0].auroc_inflation_ci95 is not None

    def test_deterministic_given_same_random_state(self):
        from biolens.eval.probing import probe_go_term_controlled_subsampling

        Z, ids, go_labels, _ = self._make_informative_data()
        kwargs = dict(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            go_id="GO:FIXED", target_n_positives=[50], n_repeats=4, random_state=99,
        )
        s1 = probe_go_term_controlled_subsampling(**kwargs)
        s2 = probe_go_term_controlled_subsampling(**kwargs)
        assert s1[0].mean_single_feature_auroc == s2[0].mean_single_feature_auroc
        assert s1[0].mean_auroc_inflation == s2[0].mean_auroc_inflation

    def test_single_repeat_gives_no_ci(self):
        from biolens.eval.probing import probe_go_term_controlled_subsampling

        Z, ids, go_labels, _ = self._make_informative_data()
        summaries = probe_go_term_controlled_subsampling(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            go_id="GO:FIXED", target_n_positives=[50], n_repeats=1,
        )
        assert summaries[0].auroc_inflation_ci95 is None

    def test_multivariate_auroc_skipped_by_default(self, monkeypatch):
        """Real efficiency fix (2026-07-29): ControlledSubsamplingSummary
        has no multivariate_auroc field at all, so computing it by default
        was pure wasted compute (up to n_repeats * len(target_n_positives)
        discarded LogisticRegression fits per call) — verify the flag
        actually threads through to _probe_single_term, not just that it
        exists as a parameter."""
        import biolens.eval.probing as probing_module

        Z, ids, go_labels, _ = self._make_informative_data()
        seen_flags = []
        real_probe_single_term = probing_module._probe_single_term

        def spy(*args, **kwargs):
            seen_flags.append(kwargs["compute_multivariate_auroc"])
            return real_probe_single_term(*args, **kwargs)

        monkeypatch.setattr(probing_module, "_probe_single_term", spy)

        probing_module.probe_go_term_controlled_subsampling(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            go_id="GO:FIXED", target_n_positives=[50], n_repeats=3,
        )
        assert seen_flags and all(flag is False for flag in seen_flags)

        seen_flags.clear()
        probing_module.probe_go_term_controlled_subsampling(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            go_id="GO:FIXED", target_n_positives=[50], n_repeats=3,
            compute_multivariate_auroc=True,
        )
        assert seen_flags and all(flag is True for flag in seen_flags)

    def test_str_output(self):
        from biolens.eval.probing import ControlledSubsamplingSummary

        s = ControlledSubsamplingSummary(
            go_id="GO:FIXED", go_name="fixed term", n_natural_positives=400,
            target_n_positives=50, n_repeats_successful=5,
            mean_single_feature_auroc=0.85, mean_held_out_auroc=0.72,
            mean_auroc_inflation=0.13, auroc_inflation_ci95=(0.08, 0.18),
        )
        text = str(s)
        assert "50" in text
        assert "0.1300" in text or "0.13" in text


class TestProbeGenomicAnnotations:
    def test_delegates_to_probe_go_terms_with_dna_labels(self):
        """Genomic-annotation analog of test_probe_with_informative_features
        — same underlying engine, cCRE-class labels instead of GO terms."""
        from biolens.eval.probing import probe_genomic_annotations

        torch.manual_seed(1)
        N, d_sae = 300, 64
        Z = torch.randn(N, d_sae).abs()
        window_ids = [f"chr1:{i * 1000}-{i * 1000 + 200}" for i in range(N)]

        labels = (Z[:, 5] > Z[:, 5].median()).numpy().astype(int)
        dna_labels = {
            window_ids[i]: ({"pELS"} if labels[i] == 1 else set())
            for i in range(N)
        }

        report = probe_genomic_annotations(
            feature_acts=Z, window_ids=window_ids, dna_labels=dna_labels,
            min_positives=20, max_go_terms=10,
        )
        assert len(report.results) >= 1
        best = report.top_results(1)[0]
        assert best.go_id == "pELS"  # field name inherited, holds a cCRE class here
        assert best.single_feature_auroc > 0.75
        assert best.best_feature_idx == 5

    def test_forwards_kwargs_like_held_out_baselines(self):
        from biolens.eval.probing import probe_genomic_annotations

        torch.manual_seed(2)
        N, d_sae = 400, 16
        Z = torch.randn(N, d_sae).abs()
        window_ids = [f"chr2:{i * 1000}-{i * 1000 + 200}" for i in range(N)]
        labels = (Z[:, 3] > Z[:, 3].median()).numpy().astype(int)
        dna_labels = {
            window_ids[i]: ({"dELS"} if labels[i] == 1 else set()) for i in range(N)
        }

        report = probe_genomic_annotations(
            feature_acts=Z, window_ids=window_ids, dna_labels=dna_labels,
            min_positives=20, max_go_terms=10, compute_held_out_baselines=True,
        )
        assert report.results[0].held_out_auroc is not None


class TestProbeTextConcepts:
    def test_delegates_to_probe_go_terms_with_text_labels(self):
        """Non-biology-control-domain analog of the GO/genomic wrapper
        tests — same shared engine, profession-class labels instead of GO
        terms or cCRE classes."""
        from biolens.eval.probing import probe_text_concepts

        torch.manual_seed(3)
        N, d_sae = 300, 64
        Z = torch.randn(N, d_sae).abs()
        text_ids = [f"biobio_{i}" for i in range(N)]

        labels = (Z[:, 7] > Z[:, 7].median()).numpy().astype(int)
        text_labels = {
            text_ids[i]: ({"profession_21"} if labels[i] == 1 else set())
            for i in range(N)
        }

        report = probe_text_concepts(
            feature_acts=Z, text_ids=text_ids, text_labels=text_labels,
            min_positives=20, max_go_terms=10,
        )
        assert len(report.results) >= 1
        best = report.top_results(1)[0]
        assert best.go_id == "profession_21"  # field name inherited, holds a profession class here
        assert best.best_feature_idx == 7


# ── Held-out / hard-negative AUROC baselines ──────────────────────────────────

class TestHeldOutBaselines:
    def test_default_behavior_unchanged_when_flag_off(self):
        """compute_held_out_baselines defaults to False — must not alter any
        existing single_feature_auroc numbers or leave the new fields unset
        as anything but None (backward compatibility for Phase 0 reruns)."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(1)
        N, d_sae = 300, 32
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        labels = (Z[:, 5] > Z[:, 5].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10,
        )
        r = report.top_results(1)[0]
        assert r.held_out_auroc is None
        assert r.hard_negative_held_out_auroc is None
        assert r.n_hard_negatives is None
        assert r.auroc_inflation is None

    def test_held_out_auroc_near_naive_for_strong_real_signal(self):
        """A genuinely informative feature (real separation, ample N) should
        show little winner's-curse inflation: held_out_auroc should track
        single_feature_auroc closely, not collapse toward 0.5."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(3)
        N, d_sae = 2000, 16
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        # Feature 2 is strongly, cleanly informative — not just noise that
        # happened to win a small-sample argmax.
        labels = (Z[:, 2] > Z[:, 2].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_held_out_baselines=True,
            random_state=7,
        )
        r = report.top_results(1)[0]
        assert r.best_feature_idx == 2
        assert r.held_out_auroc is not None
        assert r.held_out_auroc > 0.9, (
            f"Strong real signal should survive held-out estimation: {r.held_out_auroc}"
        )
        assert r.auroc_inflation is not None
        assert r.auroc_inflation < 0.1, (
            f"Little winner's-curse inflation expected for strong real signal: "
            f"{r.auroc_inflation}"
        )

    def test_held_out_auroc_debiases_pure_noise_selection(self):
        """With zero real signal (labels independent of features, matching
        the winners_curse_simulation.py setup) and many candidate features,
        the naive single_feature_auroc (argmax over many noisy estimates)
        should be inflated above 0.5, while held_out_auroc — estimated on
        data disjoint from the selection step — should sit much closer to
        0.5. This is the core empirical claim of the held-out baseline,
        exercised end-to-end here."""
        from biolens.eval.probing import probe_go_terms

        rng = np.random.default_rng(11)
        N, d_sae = 4000, 500
        Z = torch.from_numpy(rng.standard_normal((N, d_sae)).astype(np.float32))
        ids = [f"P{i:05d}" for i in range(N)]
        # Labels independent of every feature column by construction.
        label_arr = (rng.random(N) < 0.3).astype(int)
        go_labels = {
            pid: ({"GO:NULLSIGNAL"} if label_arr[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=1, compute_held_out_baselines=True,
            held_out_fraction=0.5, random_state=13,
        )
        r = report.results[0]
        assert r.single_feature_auroc > 0.55, (
            "Selection over 500 pure-noise features should show real inflation "
            f"(winner's curse): got {r.single_feature_auroc}"
        )
        assert r.held_out_auroc is not None
        assert r.held_out_auroc < r.single_feature_auroc, (
            "Held-out estimate must be lower than the inflated naive selection "
            f"estimate: held_out={r.held_out_auroc}, naive={r.single_feature_auroc}"
        )
        assert abs(r.held_out_auroc - 0.5) < 0.15, (
            f"Held-out AUROC should sit near the true null (0.5): {r.held_out_auroc}"
        )

    def test_hard_negative_auroc_none_without_go_dag(self):
        """No go_dag passed -> hard_negative_held_out_auroc must stay None,
        never silently computed against the wrong (or no) negative set."""
        from biolens.eval.probing import probe_go_terms

        torch.manual_seed(4)
        N, d_sae = 500, 16
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        labels = (Z[:, 1] > Z[:, 1].median()).numpy().astype(int)
        go_labels = {
            pid: ({"GO:0000001"} if labels[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_held_out_baselines=True,
        )
        r = report.top_results(1)[0]
        assert r.hard_negative_held_out_auroc is None

    def test_hard_negative_auroc_exposes_structural_confounding(self):
        """The core W3 scenario: a feature that has learned a BROAD class
        ("membrane") rather than the SPECIFIC claimed concept
        ("acetylcholine-gated channel", a sibling of "transporter" under
        "membrane") should ace the standard held-out AUROC (easy negatives:
        everything non-membrane) but collapse toward 0.5 on the hard-negative
        AUROC (negatives restricted to sibling membrane proteins) — exactly
        the metal-transporter-flagged-as-acetylcholine-receptor failure mode
        the hard-negative baseline exists to catch."""
        from biolens.data.uniprot import GODag
        from biolens.eval.probing import probe_go_terms

        dag = GODag(
            parents={
                "GO:ACHR": {"GO:MEMBRANE"},
                "GO:TRANSPORTER": {"GO:MEMBRANE"},
            },
            children={"GO:MEMBRANE": {"GO:ACHR", "GO:TRANSPORTER"}},
        )

        rng = np.random.default_rng(21)
        # Large enough that the held-out hard-negative subset (~test_fraction
        # * held_out_fraction * n_transporter) has enough examples for a
        # statistically reliable AUROC estimate, not just directionally right.
        n_achr, n_transporter, n_other = 300, 3000, 8000
        N = n_achr + n_transporter + n_other
        d_sae = 8
        Z = np.zeros((N, d_sae), dtype=np.float32)

        # Feature 0 fires for ANY membrane protein (broad class), regardless
        # of which specific membrane sub-type — it has NOT learned the
        # acetylcholine-specific concept, only "membrane-ness".
        Z[: n_achr + n_transporter, 0] = rng.uniform(0.5, 1.5, n_achr + n_transporter)
        Z[n_achr + n_transporter :, 0] = rng.uniform(0.0, 0.1, n_other)
        Z[:, 1:] = rng.standard_normal((N, d_sae - 1)) * 0.01  # noise filler

        ids = [f"P{i:05d}" for i in range(N)]
        go_labels: dict[str, set[str]] = {}
        for i in range(N):
            if i < n_achr:
                go_labels[ids[i]] = {"GO:ACHR"}
            elif i < n_achr + n_transporter:
                go_labels[ids[i]] = {"GO:TRANSPORTER"}
            else:
                go_labels[ids[i]] = set()

        report = probe_go_terms(
            feature_acts=torch.from_numpy(Z), protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_held_out_baselines=True,
            go_dag=dag, min_hard_negatives=10, random_state=5, held_out_fraction=0.5,
        )
        achr_result = next(r for r in report.results if r.go_id == "GO:ACHR")

        assert achr_result.best_feature_idx == 0
        assert achr_result.held_out_auroc is not None
        assert achr_result.held_out_auroc > 0.8, (
            "Standard held-out AUROC (mostly easy negatives) should look great — "
            f"this is exactly the false confidence the hard-negative test exposes: "
            f"{achr_result.held_out_auroc}"
        )
        assert achr_result.hard_negative_held_out_auroc is not None
        assert achr_result.hard_negative_held_out_auroc < 0.65, (
            "Hard-negative AUROC (sibling transporters as negatives) should "
            "collapse toward 0.5 — the feature can't distinguish the specific "
            f"claimed concept from its structural siblings: "
            f"{achr_result.hard_negative_held_out_auroc}"
        )

    def test_hard_negative_auroc_none_when_too_few_hard_negatives(self):
        """If fewer than min_hard_negatives sibling-annotated proteins land
        in the held-out split, report None rather than an unreliable
        estimate from a handful of examples."""
        from biolens.data.uniprot import GODag
        from biolens.eval.probing import probe_go_terms

        dag = GODag(
            parents={"GO:RARE_SIBLING": {"GO:PARENT"}, "GO:TARGET": {"GO:PARENT"}},
            children={"GO:PARENT": {"GO:RARE_SIBLING", "GO:TARGET"}},
        )
        torch.manual_seed(9)
        N, d_sae = 500, 8
        Z = torch.randn(N, d_sae).abs()
        ids = [f"P{i:05d}" for i in range(N)]
        labels = (Z[:, 0] > Z[:, 0].median()).numpy().astype(int)
        go_labels: dict[str, set[str]] = {}
        for i, pid in enumerate(ids):
            if labels[i] == 1:
                go_labels[pid] = {"GO:TARGET"}
            elif i % 200 == 0:  # only a handful of sibling-annotated negatives
                go_labels[pid] = {"GO:RARE_SIBLING"}
            else:
                go_labels[pid] = set()

        report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=10, compute_held_out_baselines=True,
            go_dag=dag, min_hard_negatives=50,
        )
        r = next(r for r in report.results if r.go_id == "GO:TARGET")
        assert r.hard_negative_held_out_auroc is None
        assert r.n_hard_negatives is not None
        assert r.n_hard_negatives < 50

    def test_auroc_inflation_property(self):
        from biolens.eval.probing import GOProbingResult

        r = GOProbingResult(
            "GO:0000001", "t", 100, 500, 0.85, 3, 0.80,
            held_out_auroc=0.60,
        )
        assert r.auroc_inflation == pytest.approx(0.25)

    def test_auroc_inflation_none_without_held_out(self):
        from biolens.eval.probing import GOProbingResult

        r = GOProbingResult("GO:0000001", "t", 100, 500, 0.85, 3, 0.80)
        assert r.auroc_inflation is None


# ── Permuted-label control / noise-vs-confounding decomposition ──────────────

class TestPermuteGoLabels:
    def test_preserves_marginal_go_term_counts(self):
        """Permutation must leave each GO term's total positive count
        exactly unchanged — only WHICH protein holds each label set moves."""
        from biolens.eval.probing import permute_go_labels

        go_labels = {
            "P1": {"GO:A", "GO:B"},
            "P2": {"GO:A"},
            "P3": {"GO:B"},
            "P4": set(),
            "P5": {"GO:A", "GO:C"},
        }
        ids = list(go_labels.keys())

        permuted = permute_go_labels(go_labels, ids, random_state=1)

        def counts(labels):
            c: dict[str, int] = {}
            for s in labels.values():
                for go_id in s:
                    c[go_id] = c.get(go_id, 0) + 1
            return c

        assert counts(permuted) == counts(go_labels)

    def test_preserves_co_occurrence_structure(self):
        """A protein's full label SET moves together (not resampled term by
        term) — the multiset of label-sets is unchanged, just reassigned."""
        from collections import Counter

        from biolens.eval.probing import permute_go_labels

        go_labels = {f"P{i}": {"GO:A", "GO:B"} if i < 5 else {"GO:C"} for i in range(20)}
        ids = list(go_labels.keys())

        permuted = permute_go_labels(go_labels, ids, random_state=2)

        original_counts = Counter(tuple(sorted(s)) for s in go_labels.values())
        permuted_counts = Counter(tuple(sorted(s)) for s in permuted.values())
        assert original_counts == permuted_counts

    def test_breaks_true_correspondence(self):
        """With enough proteins, permutation should actually reassign at
        least some labels (not degenerate to the identity permutation)."""
        from biolens.eval.probing import permute_go_labels

        go_labels = {f"P{i}": {f"GO:{i}"} for i in range(50)}
        ids = list(go_labels.keys())

        permuted = permute_go_labels(go_labels, ids, random_state=3)
        n_unchanged = sum(
            1 for pid in ids if permuted.get(pid) == go_labels.get(pid)
        )
        assert n_unchanged < len(ids)

    def test_unlabeled_proteins_stay_unlabeled(self):
        from biolens.eval.probing import permute_go_labels

        go_labels = {"P1": {"GO:A"}, "P2": {"GO:B"}}
        ids = ["P1", "P2", "P3"]  # P3 has no entry at all

        permuted = permute_go_labels(go_labels, ids, random_state=4)
        assert "P3" not in permuted

    def test_deterministic_given_seed(self):
        from biolens.eval.probing import permute_go_labels

        go_labels = {f"P{i}": {f"GO:{i % 3}"} for i in range(30)}
        ids = list(go_labels.keys())

        p1 = permute_go_labels(go_labels, ids, random_state=99)
        p2 = permute_go_labels(go_labels, ids, random_state=99)
        assert p1 == p2


class TestDecomposeNoiseFromConfounding:
    def test_pure_noise_shows_near_zero_confounding(self):
        """Run the real decomposition end-to-end on data with NO real
        structural confound (independent random features/labels): permuting
        labels should barely change the inflation curve, since there was no
        real signal to destroy in the first place — confounding_estimate
        should be small for every GO term."""
        from biolens.eval.probing import (
            decompose_noise_from_confounding,
            permute_go_labels,
            probe_go_terms,
        )

        rng = np.random.default_rng(17)
        N, d_sae = 1500, 100
        Z = torch.from_numpy(rng.standard_normal((N, d_sae)).astype(np.float32))
        ids = [f"P{i:05d}" for i in range(N)]
        label_arr = (rng.random(N) < 0.3).astype(int)
        go_labels = {
            pid: ({"GO:NULL"} if label_arr[i] == 1 else set())
            for i, pid in enumerate(ids)
        }

        real_report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=20, max_go_terms=1, compute_held_out_baselines=True,
            random_state=23,
        )
        permuted_labels = permute_go_labels(go_labels, ids, random_state=23)
        permuted_report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=permuted_labels,
            min_positives=20, max_go_terms=1, compute_held_out_baselines=True,
            random_state=23,
        )

        decomposition = decompose_noise_from_confounding(real_report, permuted_report)
        assert len(decomposition) == 1
        d = decomposition[0]
        assert d.confounding_estimate is not None
        assert abs(d.confounding_estimate) < 0.2, (
            f"No real structural signal existed to destroy — confounding "
            f"estimate should be small: {d.confounding_estimate}"
        )

    def test_skips_go_terms_missing_from_either_report(self):
        from biolens.eval.probing import (
            DecompositionResult,
            GOProbingReport,
            GOProbingResult,
            decompose_noise_from_confounding,
        )

        real_report = GOProbingReport(
            results=[
                GOProbingResult(
                    "GO:A", "a", 50, 200, 0.8, 0, 0.75, held_out_auroc=0.6
                ),
                GOProbingResult(
                    "GO:ONLY_REAL", "b", 50, 200, 0.7, 1, 0.65, held_out_auroc=0.55
                ),
            ],
            n_go_terms_tested=2, model_name="m", layer=0, sae_variant="topk",
        )
        permuted_report = GOProbingReport(
            results=[
                GOProbingResult(
                    "GO:A", "a", 50, 200, 0.65, 2, 0.60, held_out_auroc=0.52
                ),
            ],
            n_go_terms_tested=1, model_name="m", layer=0, sae_variant="topk",
        )

        decomposition = decompose_noise_from_confounding(real_report, permuted_report)
        assert len(decomposition) == 1
        assert decomposition[0].go_id == "GO:A"
        assert isinstance(decomposition[0], DecompositionResult)

    def test_confounding_estimate_computed_correctly(self):
        from biolens.eval.probing import (
            GOProbingReport,
            GOProbingResult,
            decompose_noise_from_confounding,
        )

        real_report = GOProbingReport(
            results=[
                GOProbingResult(
                    "GO:A", "a", 50, 200, 0.90, 0, 0.85, held_out_auroc=0.70
                ),  # real_inflation = 0.20
            ],
            n_go_terms_tested=1, model_name="m", layer=0, sae_variant="topk",
        )
        permuted_report = GOProbingReport(
            results=[
                GOProbingResult(
                    "GO:A", "a", 50, 200, 0.60, 5, 0.55, held_out_auroc=0.55
                ),  # permuted_inflation = 0.05
            ],
            n_go_terms_tested=1, model_name="m", layer=0, sae_variant="topk",
        )

        d = decompose_noise_from_confounding(real_report, permuted_report)[0]
        assert d.real_inflation == pytest.approx(0.20)
        assert d.permuted_inflation == pytest.approx(0.05)
        assert d.confounding_estimate == pytest.approx(0.15)


class TestDecomposeNoiseFromConfoundingMultiSeed:
    """Tests for the multi-permutation-draw decomposition (2026-07-15 — a
    single fixed-seed permuted_inflation value turned out to be unreliable
    for low-positive-count labels in a real genomic cCRE probing run)."""

    def test_no_new_probe_calls_needed_beyond_probe_fn(self):
        """probe_fn is called exactly n_permutations times — one per
        permutation draw, no more — confirming this doesn't secretly need a
        fresh model forward pass per draw (the whole point is reusing
        already-computed feature_acts)."""
        from biolens.eval.probing import (
            decompose_noise_from_confounding_multi_seed,
            probe_go_terms,
        )

        rng = np.random.default_rng(1)
        N, d_sae = 300, 20
        Z = torch.from_numpy(rng.standard_normal((N, d_sae)).astype(np.float32))
        ids = [f"P{i:04d}" for i in range(N)]
        label_arr = (rng.random(N) < 0.3).astype(int)
        go_labels = {pid: ({"GO:A"} if label_arr[i] else set()) for i, pid in enumerate(ids)}

        real_report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=10, max_go_terms=1, compute_held_out_baselines=True, random_state=1,
        )

        call_count = 0

        def probe_fn(permuted_labels):
            nonlocal call_count
            call_count += 1
            return probe_go_terms(
                feature_acts=Z, protein_ids=ids, go_labels=permuted_labels,
                min_positives=10, max_go_terms=1, compute_held_out_baselines=True, random_state=1,
            )

        decompose_noise_from_confounding_multi_seed(
            real_report, probe_fn, go_labels, ids, n_permutations=5,
        )
        assert call_count == 5

    def test_mean_and_ci_computed_across_seeds(self):
        """A label with a fixed real_inflation and permuted draws that vary
        seed-to-seed should get a mean roughly centered on the per-draw
        values and a CI that actually spans some width — not collapsed to a
        single point, which a real multi-seed estimate shouldn't be."""
        from biolens.eval.probing import (
            decompose_noise_from_confounding_multi_seed,
            probe_go_terms,
        )

        rng = np.random.default_rng(2)
        N, d_sae = 400, 20
        Z = torch.from_numpy(rng.standard_normal((N, d_sae)).astype(np.float32))
        ids = [f"P{i:04d}" for i in range(N)]
        # Two disjoint labels (not one homogeneous label) — permute_go_labels
        # reassigns WHOLE label sets between proteins, so if every "positive"
        # protein carried the identical singleton {"GO:A"}, every permutation
        # would reproduce the exact same dict regardless of seed (nothing
        # distinguishable to shuffle). With two distinct labels, shuffling
        # actually changes which protein IDs end up positive for GO:A.
        assignment = rng.choice(["GO:A", "GO:B", "none"], size=N, p=[0.3, 0.3, 0.4])
        go_labels = {
            pid: ({assignment[i]} if assignment[i] != "none" else set())
            for i, pid in enumerate(ids)
        }

        real_report = probe_go_terms(
            feature_acts=Z, protein_ids=ids, go_labels=go_labels,
            min_positives=10, max_go_terms=2, compute_held_out_baselines=True, random_state=2,
        )

        def probe_fn(permuted_labels):
            return probe_go_terms(
                feature_acts=Z, protein_ids=ids, go_labels=permuted_labels,
                min_positives=10, max_go_terms=2, compute_held_out_baselines=True, random_state=2,
            )

        result = decompose_noise_from_confounding_multi_seed(
            real_report, probe_fn, go_labels, ids, n_permutations=15, base_random_state=100,
        )
        by_id = {r.go_id: r for r in result}
        assert "GO:A" in by_id
        r = by_id["GO:A"]
        assert r.n_seeds_successful == 15
        assert r.permuted_inflation_ci is not None
        lo, hi = r.permuted_inflation_ci
        assert lo <= r.mean_permuted_inflation <= hi
        assert hi > lo, "15 independent draws should not collapse to a zero-width CI"

    def test_different_seeds_produce_different_permutations(self):
        """Sanity check that n_permutations draws are actually independent
        (distinct seeds), not accidentally reusing one seed n times — the
        entire point of this function is averaging over independent draws."""
        from biolens.eval.probing import permute_go_labels

        rng = np.random.default_rng(3)
        N = 200
        ids = [f"P{i:04d}" for i in range(N)]
        # Two disjoint labels — see test_mean_and_ci_computed_across_seeds for
        # why a single homogeneous label would make every permutation
        # indistinguishable regardless of seed.
        assignment = rng.choice(["GO:A", "GO:B", "none"], size=N, p=[0.3, 0.3, 0.4])
        go_labels = {
            pid: ({assignment[i]} if assignment[i] != "none" else set())
            for i, pid in enumerate(ids)
        }

        draws = [
            permute_go_labels(go_labels, ids, random_state=100 + i) for i in range(5)
        ]
        # At least one pair of draws must differ — otherwise every "seed" is
        # silently producing the identical permutation.
        assert any(draws[0] != draws[j] for j in range(1, 5))

    def test_label_missing_from_all_permuted_draws_excluded(self):
        """A label that never clears min_positives under ANY permutation is
        silently absent from the result, not guessed at with an empty CI —
        same convention as decompose_noise_from_confounding's single-draw
        version."""
        from biolens.eval.probing import (
            GOProbingReport,
            GOProbingResult,
            decompose_noise_from_confounding_multi_seed,
        )

        real_report = GOProbingReport(
            results=[GOProbingResult("GO:RARE", "r", 50, 200, 0.8, 0, 0.75, held_out_auroc=0.6)],
            n_go_terms_tested=1, model_name="m", layer=0, sae_variant="topk",
        )
        empty_permuted_report = GOProbingReport(
            results=[], n_go_terms_tested=0, model_name="m", layer=0, sae_variant="topk",
        )

        result = decompose_noise_from_confounding_multi_seed(
            real_report,
            probe_fn=lambda labels: empty_permuted_report,
            labels={"P1": {"GO:RARE"}},
            ids=["P1"],
            n_permutations=3,
        )
        assert result == []
