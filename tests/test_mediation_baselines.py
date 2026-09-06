"""Tests for the Geuvadis case study's mediation baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestRawEmbeddingBaseline:
    def test_recovers_known_linear_relationship(self):
        from biolens.eval.mediation_baselines import raw_embedding_baseline

        rng = np.random.default_rng(0)
        n = 1000
        embedding_diff = rng.normal(0, 1, size=n)
        outcome = 2.0 * embedding_diff + rng.normal(0, 0.5, size=n)

        result = raw_embedding_baseline(embedding_diff, outcome)
        assert result.coefficient == pytest.approx(2.0, abs=0.2)
        assert result.r_squared > 0.8
        assert result.p_value < 0.001

    def test_no_relationship_gives_low_r_squared(self):
        from biolens.eval.mediation_baselines import raw_embedding_baseline

        rng = np.random.default_rng(1)
        n = 500
        embedding_diff = rng.normal(0, 1, size=n)
        outcome = rng.normal(0, 1, size=n)  # independent

        result = raw_embedding_baseline(embedding_diff, outcome)
        assert result.r_squared < 0.05

    def test_with_covariates(self):
        from biolens.eval.mediation_baselines import raw_embedding_baseline

        rng = np.random.default_rng(2)
        n = 800
        embedding_diff = rng.normal(0, 1, size=n)
        covariate = rng.normal(0, 1, size=n)
        outcome = 1.5 * embedding_diff + 3.0 * covariate + rng.normal(0, 0.3, size=n)

        result = raw_embedding_baseline(
            embedding_diff, outcome, covariates=pd.DataFrame({"pc1": covariate})
        )
        assert result.coefficient == pytest.approx(1.5, abs=0.2)

    def test_mismatched_lengths_raise(self):
        from biolens.eval.mediation_baselines import raw_embedding_baseline

        with pytest.raises(ValueError):
            raw_embedding_baseline(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_str_output(self):
        from biolens.eval.mediation_baselines import BaselineResult

        r = BaselineResult(method="raw_embedding", r_squared=0.5, coefficient=1.2, p_value=0.01, n_obs=100)
        s = str(r)
        assert "raw_embedding" in s
        assert "n=100" in s


class TestTwasStyleBaseline:
    def test_recovers_known_genotype_effect(self):
        from biolens.eval.mediation_baselines import twas_style_baseline

        rng = np.random.default_rng(3)
        n = 1000
        genotype = rng.integers(0, 3, size=n).astype(float)
        outcome = 0.8 * genotype + rng.normal(0, 0.5, size=n)

        result = twas_style_baseline(genotype, outcome)
        assert result.coefficient == pytest.approx(0.8, abs=0.15)
        assert result.p_value < 0.001

    def test_with_covariates(self):
        from biolens.eval.mediation_baselines import twas_style_baseline

        rng = np.random.default_rng(4)
        n = 800
        genotype = rng.integers(0, 3, size=n).astype(float)
        pc = rng.normal(0, 1, size=n)
        outcome = 0.5 * genotype + 2.0 * pc + rng.normal(0, 0.3, size=n)

        result = twas_style_baseline(genotype, outcome, covariates=pd.DataFrame({"pc1": pc}))
        assert result.coefficient == pytest.approx(0.5, abs=0.15)

    def test_mismatched_lengths_raise(self):
        from biolens.eval.mediation_baselines import twas_style_baseline

        with pytest.raises(ValueError):
            twas_style_baseline(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0]))


class TestDnaOneHot:
    def test_base_order_is_acgt(self):
        """A=0, C=1, G=2, T=3 — verified against calico/baskerville's own
        dna_1hot() (the reference implementation Borzoi was trained with),
        see _dna_one_hot's docstring."""
        from biolens.eval.mediation_baselines import _dna_one_hot

        arr = _dna_one_hot("ACGT")
        assert arr.shape == (4, 4)
        assert arr[:, 0].tolist() == [1, 0, 0, 0]  # A
        assert arr[:, 1].tolist() == [0, 1, 0, 0]  # C
        assert arr[:, 2].tolist() == [0, 0, 1, 0]  # G
        assert arr[:, 3].tolist() == [0, 0, 0, 1]  # T

    def test_lowercase_normalized(self):
        from biolens.eval.mediation_baselines import _dna_one_hot

        arr = _dna_one_hot("acgt")
        assert arr[:, 0].tolist() == [1, 0, 0, 0]

    def test_non_acgt_base_is_all_zero(self):
        from biolens.eval.mediation_baselines import _dna_one_hot

        arr = _dna_one_hot("ANG")
        assert arr[:, 1].tolist() == [0, 0, 0, 0]


class TestBorzoiBaseline:
    """Real, complete implementation (not a stub) — see borzoi_baseline's
    docstring for the primary-source verification of every I/O constant.
    ONE locus at a time (matches the real call site's per-locus, dosage-
    scaled convention — see raw_embedding_baseline's `embedding_diff_norm *
    treatment` usage in scripts/run_geuvadis_case_study.py). Tests mock the
    `borzoi_pytorch` package (same pattern test_models.py's TestEvo2Adapter
    uses for `evo2`) — this tests the GLUE CODE's correctness (shape
    handling, gene-bin aggregation, ref/alt subtraction, dosage scaling,
    OLS assembly), not real Borzoi model behavior, which is out of scope
    for a unit test (needs a real multi-hundred-MB checkpoint download)."""

    SEQ_LEN = 524288

    @pytest.fixture
    def fake_borzoi_module(self, monkeypatch):
        """Inject fake `borzoi_pytorch` + `borzoi_pytorch.gene_utils` modules
        so borzoi_baseline's deferred imports succeed without the real
        package. The fake model returns, for every track/bin, a constant
        equal to `count_of('G', input_sequence)` — lets tests compute the
        exact expected predicted_shift analytically from real string
        counting, rather than needing to know real Borzoi internals."""
        import sys
        import types

        import numpy as np
        import torch

        from biolens.eval.mediation_baselines import (
            _BORZOI_GM12878_RNA_TRACK_INDICES,
            _BORZOI_OUTPUT_BINS,
        )

        n_tracks = max(_BORZOI_GM12878_RNA_TRACK_INDICES) + 1
        calls: list[dict] = []

        class FakeBorzoiModel:
            def __init__(self):
                self._param = torch.zeros(1)

            def to(self, device):
                self._param = self._param.to(device)
                return self

            def eval(self):
                return self

            def parameters(self):
                return iter([self._param])

            def __call__(self, one_hot: "torch.Tensor"):
                # one_hot: (1, 4, seq_len) -- A=0,C=1,G=2,T=3 channel-first.
                n_g = int(one_hot[0, 2, :].sum().item())
                calls.append({"shape": tuple(one_hot.shape), "n_g": n_g})
                return torch.full((1, n_tracks, _BORZOI_OUTPUT_BINS), float(n_g))

        class FakeBorzoi:
            @staticmethod
            def from_pretrained(checkpoint):
                return FakeBorzoiModel()

        class FakeGene:
            """Minimal span=True reimplementation of gene_utils.Gene,
            sufficient to test the calling code's bin-index usage — not a
            re-test of the upstream reference implementation's own
            correctness (that's the real package's responsibility)."""

            def __init__(self, chrom, strand, kv):
                self._exons: list[tuple[int, int]] = []

            def add_exon(self, start, end):
                self._exons.append((start, end))

            def output_slice(self, seq_start, seq_len, model_stride, span=False, **kwargs):
                gene_start = min(e[0] for e in self._exons)
                gene_end = max(e[1] for e in self._exons)
                gene_seq_start = max(0, gene_start - seq_start)
                gene_seq_end = max(0, gene_end - seq_start)
                slice_start = int(round(gene_seq_start / model_stride))
                slice_end = int(round(gene_seq_end / model_stride))
                slice_max = int(seq_len / model_stride)
                slice_start = min(max(slice_start, 0), slice_max)
                slice_end = min(max(slice_end, 0), slice_max)
                if slice_start >= slice_end:
                    return np.array([], dtype=int)
                return np.arange(slice_start, slice_end)

        fake_module = types.ModuleType("borzoi_pytorch")
        fake_module.Borzoi = FakeBorzoi
        fake_gene_utils = types.ModuleType("borzoi_pytorch.gene_utils")
        fake_gene_utils.Gene = FakeGene

        monkeypatch.setitem(sys.modules, "borzoi_pytorch", fake_module)
        monkeypatch.setitem(sys.modules, "borzoi_pytorch.gene_utils", fake_gene_utils)
        return calls

    def _make_seq(self, n_g: int) -> str:
        """A SEQ_LEN-length sequence with exactly n_g 'G' bases, rest 'A' —
        the fake model in fake_borzoi_module responds deterministically to
        the total G count, so this is all these tests need to construct a
        sequence with a known, exact predicted output."""
        return "G" * n_g + "A" * (self.SEQ_LEN - n_g)

    def test_import_error_without_package(self, monkeypatch):
        """Forces the ImportError path via sys.modules rather than relying
        on borzoi-pytorch actually being absent from whatever environment
        runs this test — real regression, 2026-07-30: once borzoi-pytorch
        was genuinely installed on the GPU cluster, this test silently
        started hitting a completely different error (the real gene_span-
        overlap ValueError further down borzoi_baseline, since the import
        now succeeds) instead of exercising the path it's meant to test."""
        import sys

        from biolens.eval.mediation_baselines import borzoi_baseline

        monkeypatch.setitem(sys.modules, "borzoi_pytorch", None)
        monkeypatch.setitem(sys.modules, "borzoi_pytorch.gene_utils", None)

        with pytest.raises(ImportError, match="borzoi-pytorch"):
            borzoi_baseline(
                "A" * self.SEQ_LEN, "A" * self.SEQ_LEN, 0, (0, 100),
                genotype_dosage=np.array([0.0, 1.0, 2.0]),
                outcome=np.array([1.0, 2.0, 3.0]),
            )

    def test_wrong_sequence_length_raises(self, fake_borzoi_module):
        from biolens.eval.mediation_baselines import borzoi_baseline

        with pytest.raises(ValueError, match="524288"):
            borzoi_baseline(
                "A" * 100, "A" * 100, 0, (0, 100),
                genotype_dosage=np.array([0.0, 1.0]), outcome=np.array([1.0, 2.0]),
            )

    def test_mismatched_dosage_outcome_lengths_raise(self, fake_borzoi_module):
        from biolens.eval.mediation_baselines import borzoi_baseline

        with pytest.raises(ValueError, match="same length"):
            borzoi_baseline(
                "A" * self.SEQ_LEN, "A" * self.SEQ_LEN, 0, (0, 100),
                genotype_dosage=np.array([0.0, 1.0, 2.0]), outcome=np.array([1.0, 2.0]),
            )

    def test_model_called_once_each_for_ref_and_alt(self, fake_borzoi_module):
        """Exactly twice total for one locus — NOT once per sample, since
        the DNA sequence is identical across samples at a given allele
        (only genotype dosage varies per sample)."""
        from biolens.eval.mediation_baselines import (
            _BORZOI_CROP_BP,
            borzoi_baseline,
        )

        gene_span = (_BORZOI_CROP_BP + 1000, _BORZOI_CROP_BP + 2000)
        borzoi_baseline(
            self._make_seq(100), self._make_seq(200), 0, gene_span,
            genotype_dosage=np.zeros(50), outcome=np.zeros(50),  # 50 "samples"
        )
        assert len(fake_borzoi_module) == 2  # ref + alt, regardless of n_samples
        assert all(c["shape"] == (1, 4, self.SEQ_LEN) for c in fake_borzoi_module)

    def test_reuses_preloaded_model_when_given(self, fake_borzoi_module):
        """Passing `model=` should skip Borzoi.from_pretrained entirely —
        the real call site loads once and reuses it across many loci."""
        from biolens.eval.mediation_baselines import (
            _BORZOI_CROP_BP,
            borzoi_baseline,
            load_borzoi_model,
        )

        model = load_borzoi_model()
        gene_span = (_BORZOI_CROP_BP + 1000, _BORZOI_CROP_BP + 2000)
        for _ in range(3):
            borzoi_baseline(
                self._make_seq(100), self._make_seq(200), 0, gene_span,
                genotype_dosage=np.array([0.0, 1.0]), outcome=np.array([1.0, 2.0]),
                model=model,
            )
        # 3 loci * 2 calls each -- same single model instance throughout,
        # not re-fetched via from_pretrained each time.
        assert len(fake_borzoi_module) == 6

    def test_device_param_ignored_when_model_given(self, fake_borzoi_module):
        """Regression test, job 9727443 (2026-07-31): the docstring already
        said device= is "ignored if model is given (assumed already on the
        right device)", but the code didn't honor that — device silently
        defaulted to "cpu" even when a model pre-loaded on cuda was passed
        in via model=, crashing on a real CPU/CUDA tensor mismatch the
        first time this path ever ran against real data. Deliberately
        passes a garbage device= string: under the pre-fix code, `.to()`
        would be called with that literal string and raise a real
        RuntimeError (invalid device); the fix derives the device from
        model.parameters() instead, so this must succeed regardless of
        what device= says."""
        from biolens.eval.mediation_baselines import (
            _BORZOI_CROP_BP,
            borzoi_baseline,
            load_borzoi_model,
        )

        model = load_borzoi_model(device="cpu")
        gene_span = (_BORZOI_CROP_BP + 1000, _BORZOI_CROP_BP + 2000)
        result = borzoi_baseline(
            self._make_seq(100), self._make_seq(200), 0, gene_span,
            genotype_dosage=np.array([0.0, 1.0]), outcome=np.array([1.0, 2.0]),
            model=model, device="not_a_real_device_xyz",
        )
        assert result is not None

    def test_recovers_known_linear_relationship(self, fake_borzoi_module):
        """Construct a locus where predicted_shift (via the fake model's
        deterministic G-count response) times genotype dosage is an EXACT,
        hand-computable linear function of the outcome, and confirm OLS
        recovers it — mirroring TestRawEmbeddingBaseline's own test."""
        from biolens.eval.mediation_baselines import (
            _BORZOI_CROP_BP,
            _BORZOI_GM12878_RNA_TRACK_INDICES,
            borzoi_baseline,
        )

        # Gene span chosen on exact bin-size multiples so both the real
        # output_slice and the fake's round() land on the same bin count.
        gene_start_rel = 32 * 30   # 960
        gene_end_rel = 32 * 60     # 1920
        n_bins = 30
        gene_span = (_BORZOI_CROP_BP + gene_start_rel, _BORZOI_CROP_BP + gene_end_rel)
        n_tracks = len(_BORZOI_GM12878_RNA_TRACK_INDICES)

        ref_n_g, alt_n_g = 150, 170
        expected_shift = (alt_n_g - ref_n_g) * n_tracks * n_bins  # exact, by construction

        rng = np.random.default_rng(1)
        n_samples = 300
        dosage = rng.integers(0, 3, size=n_samples).astype(float)
        outcome = 3.0 * expected_shift * dosage + rng.normal(0, 1e-6, size=n_samples)

        result = borzoi_baseline(
            self._make_seq(ref_n_g), self._make_seq(alt_n_g), 0, gene_span,
            genotype_dosage=dosage, outcome=outcome,
        )
        assert result.coefficient == pytest.approx(3.0, rel=1e-3)
        assert result.r_squared > 0.999
        assert result.n_obs == n_samples

    def test_no_gene_overlap_raises(self, fake_borzoi_module):
        """A gene span entirely outside the output-covered range should
        raise, not silently produce a meaningless zero shift."""
        from biolens.eval.mediation_baselines import (
            _BORZOI_CROP_BP,
            _BORZOI_OUTPUT_COVERED_BP,
            borzoi_baseline,
        )

        non_overlapping_span = (
            _BORZOI_CROP_BP + _BORZOI_OUTPUT_COVERED_BP + 10_000,
            _BORZOI_CROP_BP + _BORZOI_OUTPUT_COVERED_BP + 10_100,
        )
        with pytest.raises(ValueError, match="does not overlap"):
            borzoi_baseline(
                self._make_seq(100), self._make_seq(150), 0, non_overlapping_span,
                genotype_dosage=np.array([0.0, 1.0]), outcome=np.array([1.0, 2.0]),
            )
