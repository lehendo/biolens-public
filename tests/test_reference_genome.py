"""
Tests for reference-genome sequence access (biolens.data.reference_genome).

Uses small synthetic FASTA files (real files, really indexed by pyfaidx —
not mocked) to exercise the real pyfaidx API and this module's coordinate
logic, without needing a multi-GB real genome. See the module's docstring
for why the real-genome path itself is necessarily unverified here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _write_fasta(tmp_path: Path, records: dict[str, str]) -> Path:
    fasta_path = tmp_path / "genome.fa"
    lines = []
    for name, seq in records.items():
        lines.append(f">{name}")
        for i in range(0, len(seq), 60):
            lines.append(seq[i : i + 60])
    fasta_path.write_text("\n".join(lines) + "\n")
    return fasta_path


# ── load_reference_genome ─────────────────────────────────────────────────────

class TestLoadReferenceGenome:
    def test_loads_valid_fasta(self, tmp_path):
        from biolens.data.reference_genome import load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "ACGT" * 50})
        fasta = load_reference_genome(fasta_path)
        assert "chr1" in fasta.keys()
        assert len(fasta["chr1"]) == 200

    def test_missing_file_raises_actionable_error(self, tmp_path):
        from biolens.data.reference_genome import load_reference_genome

        with pytest.raises(FileNotFoundError, match="hg38.fa.gz"):
            load_reference_genome(tmp_path / "does_not_exist.fa")


# ── generate_tiling_windows ────────────────────────────────────────────────────

class TestGenerateTilingWindows:
    def test_basic_tiling_no_overlap(self, tmp_path):
        from biolens.data.reference_genome import generate_tiling_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 1000})
        fasta = load_reference_genome(fasta_path)

        windows = generate_tiling_windows(fasta, window_size=100, stride=100, chroms=["chr1"])
        assert len(windows) == 10  # range(0, 1000-100+1, 100) -> 0,100,...,900 (10 windows)
        assert windows[0] == ("chr1:0-100", "chr1", 0, 100)
        assert windows[1] == ("chr1:100-200", "chr1", 100, 200)

    def test_overlapping_windows_with_smaller_stride(self, tmp_path):
        from biolens.data.reference_genome import generate_tiling_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 300})
        fasta = load_reference_genome(fasta_path)

        windows = generate_tiling_windows(fasta, window_size=100, stride=50, chroms=["chr1"])
        starts = [w[2] for w in windows]
        assert starts == [0, 50, 100, 150, 200]

    def test_n_heavy_windows_skipped(self, tmp_path):
        from biolens.data.reference_genome import generate_tiling_windows, load_reference_genome

        seq = "N" * 100 + "A" * 100  # first window all-N, second window clean
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        windows = generate_tiling_windows(
            fasta, window_size=100, stride=100, chroms=["chr1"], skip_n_threshold=0.1
        )
        assert len(windows) == 1
        assert windows[0][2] == 100  # only the clean window survives

    def test_restricts_to_given_chroms(self, tmp_path):
        from biolens.data.reference_genome import generate_tiling_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 200, "chr2": "C" * 200})
        fasta = load_reference_genome(fasta_path)

        windows = generate_tiling_windows(fasta, window_size=100, stride=100, chroms=["chr1"])
        assert all(w[1] == "chr1" for w in windows)

    def test_missing_chrom_skipped_not_crashed(self, tmp_path):
        from biolens.data.reference_genome import generate_tiling_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 200})
        fasta = load_reference_genome(fasta_path)

        windows = generate_tiling_windows(
            fasta, window_size=100, stride=100, chroms=["chr1", "chrDOESNOTEXIST"]
        )
        assert all(w[1] == "chr1" for w in windows)

    def test_none_chroms_uses_all(self, tmp_path):
        from biolens.data.reference_genome import generate_tiling_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 200, "chr2": "C" * 200})
        fasta = load_reference_genome(fasta_path)

        windows = generate_tiling_windows(fasta, window_size=100, stride=100, chroms=None)
        seen_chroms = {w[1] for w in windows}
        assert seen_chroms == {"chr1", "chr2"}

    def test_offset_shifts_the_tiling_grid(self, tmp_path):
        """A non-zero offset produces a genuinely different, non-overlapping-
        with-offset=0 set of window starts from the same stride — this is
        what lets a replication check (e.g. a PLS-anomaly follow-up) draw
        an independent eval window set without adding RNG plumbing."""
        from biolens.data.reference_genome import generate_tiling_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 1000})
        fasta = load_reference_genome(fasta_path)

        default = generate_tiling_windows(fasta, window_size=100, stride=200, chroms=["chr1"])
        shifted = generate_tiling_windows(
            fasta, window_size=100, stride=200, chroms=["chr1"], offset=100
        )
        default_starts = {w[2] for w in default}
        shifted_starts = {w[2] for w in shifted}
        assert default_starts == {0, 200, 400, 600, 800}
        assert shifted_starts == {100, 300, 500, 700, 900}
        assert default_starts.isdisjoint(shifted_starts)

    def test_offset_zero_matches_default_behavior(self, tmp_path):
        from biolens.data.reference_genome import generate_tiling_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 500})
        fasta = load_reference_genome(fasta_path)

        default = generate_tiling_windows(fasta, window_size=100, stride=100, chroms=["chr1"])
        explicit_zero = generate_tiling_windows(
            fasta, window_size=100, stride=100, chroms=["chr1"], offset=0
        )
        assert default == explicit_zero

    def test_offset_out_of_range_raises(self, tmp_path):
        from biolens.data.reference_genome import generate_tiling_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 500})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError):
            generate_tiling_windows(fasta, window_size=100, stride=200, chroms=["chr1"], offset=200)
        with pytest.raises(ValueError):
            generate_tiling_windows(
                fasta, window_size=100, stride=200, chroms=["chr1"], offset=-1
            )


# ── generate_gene_windows ──────────────────────────────────────────────────────

class TestGenerateGeneWindows:
    def test_tiles_within_gene_body_plus_flank(self, tmp_path):
        from biolens.data.reference_genome import generate_gene_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 1000})
        fasta = load_reference_genome(fasta_path)

        # Gene body [400, 600), flank_bp=100 -> region [300, 700).
        genes = [("ENSG1.1", "chr1", 400, 600)]
        windows = generate_gene_windows(
            fasta, genes, window_size=100, stride=100, flank_bp=100
        )
        starts = sorted(w[2] for w in windows)
        assert starts == [300, 400, 500, 600]
        assert all(w[2] + 100 <= 700 for w in windows)

    def test_every_window_tagged_with_its_gene_id(self, tmp_path):
        from biolens.data.reference_genome import generate_gene_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 1000})
        fasta = load_reference_genome(fasta_path)

        genes = [("ENSG1.1", "chr1", 400, 600)]
        windows = generate_gene_windows(fasta, genes, window_size=100, stride=100, flank_bp=0)
        assert all(w[4] == "ENSG1.1" for w in windows)

    def test_window_id_stays_unique_across_overlapping_genes(self, tmp_path):
        """Two genes whose flanked regions physically overlap must still
        produce distinct window_ids (each is a separate probing instance,
        not deduplicated by genomic position)."""
        from biolens.data.reference_genome import generate_gene_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 1000})
        fasta = load_reference_genome(fasta_path)

        genes = [("ENSG1.1", "chr1", 400, 500), ("ENSG2.1", "chr1", 450, 550)]
        windows = generate_gene_windows(fasta, genes, window_size=50, stride=50, flank_bp=0)
        window_ids = [w[0] for w in windows]
        assert len(window_ids) == len(set(window_ids))
        assert any(w[4] == "ENSG1.1" for w in windows)
        assert any(w[4] == "ENSG2.1" for w in windows)

    def test_clamps_region_to_chromosome_bounds(self, tmp_path):
        """A gene near the start/end of a chromosome shouldn't have its
        flank region run off the sequence."""
        from biolens.data.reference_genome import generate_gene_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 500})
        fasta = load_reference_genome(fasta_path)

        genes = [("ENSG1.1", "chr1", 0, 100)]  # flank would go negative without clamping
        windows = generate_gene_windows(
            fasta, genes, window_size=100, stride=100, flank_bp=1000
        )
        assert all(w[2] >= 0 for w in windows)
        assert all(w[3] <= 500 for w in windows)

    def test_n_heavy_windows_skipped(self, tmp_path):
        from biolens.data.reference_genome import generate_gene_windows, load_reference_genome

        seq = "N" * 100 + "A" * 100
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        genes = [("ENSG1.1", "chr1", 0, 200)]
        windows = generate_gene_windows(
            fasta, genes, window_size=100, stride=100, flank_bp=0, skip_n_threshold=0.1
        )
        assert len(windows) == 1
        assert windows[0][2] == 100

    def test_missing_chrom_skipped_not_crashed(self, tmp_path):
        from biolens.data.reference_genome import generate_gene_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 500})
        fasta = load_reference_genome(fasta_path)

        genes = [("ENSG1.1", "chrDOESNOTEXIST", 0, 100), ("ENSG2.1", "chr1", 0, 100)]
        windows = generate_gene_windows(fasta, genes, window_size=50, stride=50, flank_bp=0)
        assert all(w[4] == "ENSG2.1" for w in windows)

    def test_empty_gene_list_gives_empty_windows(self, tmp_path):
        from biolens.data.reference_genome import generate_gene_windows, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 500})
        fasta = load_reference_genome(fasta_path)

        assert generate_gene_windows(fasta, [], window_size=100, stride=100) == []


# ── extract_window_sequences ───────────────────────────────────────────────────

class TestExtractWindowSequences:
    def test_extracts_correct_sequence(self, tmp_path):
        from biolens.data.reference_genome import extract_window_sequences, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "ACGTACGTAC"})
        fasta = load_reference_genome(fasta_path)

        windows = [("win1", "chr1", 2, 6)]
        seqs = extract_window_sequences(fasta, windows)
        assert seqs["win1"] == "GTAC"

    def test_uppercases_soft_masked_sequence(self, tmp_path):
        from biolens.data.reference_genome import extract_window_sequences, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "ACGTacgtAC"})  # lowercase = soft-masked
        fasta = load_reference_genome(fasta_path)

        windows = [("win1", "chr1", 0, 10)]
        seqs = extract_window_sequences(fasta, windows)
        assert seqs["win1"] == "ACGTACGTAC"


# ── extract_ref_alt_context ────────────────────────────────────────────────────

class TestExtractRefAltContext:
    def test_ref_and_alt_differ_only_at_variant_position(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_context, load_reference_genome

        # Position 11 (1-based) is index 10 -> 'A' in this sequence.
        seq = "GGGGGGGGGG" + "A" + "TTTTTTTTTT"
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        ref_seq, alt_seq = extract_ref_alt_context(
            fasta, "chr1", pos=11, ref_allele="A", alt_allele="G", context_bp=5
        )
        assert len(ref_seq) == len(alt_seq) == 11
        assert ref_seq[5] == "A"
        assert alt_seq[5] == "G"
        assert ref_seq[:5] == alt_seq[:5]
        assert ref_seq[6:] == alt_seq[6:]

    def test_ref_allele_mismatch_raises(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_context, load_reference_genome

        seq = "GGGGGGGGGG" + "A" + "TTTTTTTTTT"
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError, match="mismatch"):
            extract_ref_alt_context(
                fasta, "chr1", pos=11, ref_allele="C", alt_allele="G", context_bp=5
            )  # real base is 'A', not 'C'

    def test_indel_alleles_raise(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_context, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 50})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError, match="single-nucleotide"):
            extract_ref_alt_context(
                fasta, "chr1", pos=20, ref_allele="AT", alt_allele="A", context_bp=5
            )

    def test_invalid_base_raises(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_context, load_reference_genome

        seq = "GGGGGGGGGG" + "A" + "TTTTTTTTTT"
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError, match="Non-ACGT"):
            extract_ref_alt_context(
                fasta, "chr1", pos=11, ref_allele="A", alt_allele="N", context_bp=5
            )

    def test_variant_too_close_to_start_raises(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_context, load_reference_genome

        seq = "A" * 50
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError, match="too close"):
            extract_ref_alt_context(
                fasta, "chr1", pos=2, ref_allele="A", alt_allele="G", context_bp=10
            )

    def test_asymmetric_context_produces_expected_length(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_context, load_reference_genome

        seq = "G" * 100 + "A" + "T" * 100
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        ref_seq, alt_seq = extract_ref_alt_context(
            fasta, "chr1", pos=101, ref_allele="A", alt_allele="C", context_bp=50
        )
        assert len(ref_seq) == 101  # 2*50 + 1


class TestExtractRefAltWindow:
    """extract_ref_alt_window — like extract_ref_alt_context but for an
    EXACT total window_size (odd or even), needed for fixed-input-length
    models like Borzoi (524288bp, an even number extract_ref_alt_context's
    2*context_bp+1 formula can never produce)."""

    def test_even_window_size_supported(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_window, load_reference_genome

        seq = "G" * 100 + "A" + "T" * 100
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        ref_seq, alt_seq, window_start = extract_ref_alt_window(
            fasta, "chr1", pos=101, ref_allele="A", alt_allele="C", window_size=40
        )
        assert len(ref_seq) == len(alt_seq) == 40

    def test_ref_and_alt_differ_only_at_variant_position(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_window, load_reference_genome

        seq = "GGGGGGGGGG" + "A" + "TTTTTTTTTT"
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        ref_seq, alt_seq, _ = extract_ref_alt_window(
            fasta, "chr1", pos=11, ref_allele="A", alt_allele="G", window_size=10
        )
        diffs = [i for i in range(len(ref_seq)) if ref_seq[i] != alt_seq[i]]
        assert diffs == [ref_seq.index("A")] or diffs == [alt_seq.index("G")]
        assert alt_seq[diffs[0]] == "G"
        assert ref_seq[diffs[0]] == "A"

    def test_variant_placed_at_window_size_floor_div_2(self, tmp_path):
        """The variant should sit at index window_size // 2 within the
        window, and window_start should make that arithmetic checkable
        against the real genomic coordinate."""
        from biolens.data.reference_genome import extract_ref_alt_window, load_reference_genome

        seq = "G" * 100 + "A" + "T" * 100
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        ref_seq, alt_seq, window_start = extract_ref_alt_window(
            fasta, "chr1", pos=101, ref_allele="A", alt_allele="C", window_size=40
        )
        variant_idx = 101 - 1  # 0-indexed
        relative_pos = variant_idx - window_start
        assert relative_pos == 40 // 2
        assert ref_seq[relative_pos] == "A"
        assert alt_seq[relative_pos] == "C"

    def test_window_start_is_correct_genomic_coordinate(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_window, load_reference_genome

        seq = "G" * 100 + "A" + "T" * 100
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        ref_seq, _, window_start = extract_ref_alt_window(
            fasta, "chr1", pos=101, ref_allele="A", alt_allele="C", window_size=40
        )
        # Reconstruct the window directly from the raw sequence and compare.
        assert seq[window_start : window_start + 40] == ref_seq

    def test_ref_allele_mismatch_raises(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_window, load_reference_genome

        seq = "GGGGGGGGGG" + "A" + "TTTTTTTTTT"
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError, match="mismatch"):
            extract_ref_alt_window(
                fasta, "chr1", pos=11, ref_allele="C", alt_allele="G", window_size=10
            )

    def test_indel_alleles_raise(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_window, load_reference_genome

        fasta_path = _write_fasta(tmp_path, {"chr1": "A" * 50})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError, match="single-nucleotide"):
            extract_ref_alt_window(
                fasta, "chr1", pos=20, ref_allele="AT", alt_allele="A", window_size=10
            )

    def test_invalid_base_raises(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_window, load_reference_genome

        seq = "GGGGGGGGGG" + "A" + "TTTTTTTTTT"
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError, match="Non-ACGT"):
            extract_ref_alt_window(
                fasta, "chr1", pos=11, ref_allele="A", alt_allele="N", window_size=10
            )

    def test_window_too_close_to_start_raises(self, tmp_path):
        from biolens.data.reference_genome import extract_ref_alt_window, load_reference_genome

        seq = "A" * 50
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError, match="too close"):
            extract_ref_alt_window(
                fasta, "chr1", pos=2, ref_allele="A", alt_allele="G", window_size=40
            )

    def test_window_too_close_to_end_raises(self, tmp_path):
        """Regression test: a variant near the chromosome's END used to
        silently produce a truncated ref_seq (pyfaidx slicing past the end
        doesn't raise or pad) and crash on a raw IndexError deep in the
        relative_pos indexing below, instead of the clean ValueError every
        other invalid-window case already raises — a real production
        failure (job 9727048, 2026-07-31) that an uncaught IndexError
        crashed the whole case-study run instead of being skipped for just
        that one locus, the same way the mismatched-ref-allele ValueError
        above is."""
        from biolens.data.reference_genome import extract_ref_alt_window, load_reference_genome

        seq = "A" * 50
        fasta_path = _write_fasta(tmp_path, {"chr1": seq})
        fasta = load_reference_genome(fasta_path)

        with pytest.raises(ValueError, match="too close"):
            extract_ref_alt_window(
                fasta, "chr1", pos=49, ref_allele="A", alt_allele="G", window_size=40
            )


class _FakeLiftOver:
    """Minimal duck-typed stand-in for pyliftover.LiftOver — real network/
    chain-file behavior is exercised at the pyliftover-integration level
    (a real chain file was independently validated 2026-07-31 against a
    known ground-truth case, rs6002851, producing the exact real GRCh38
    position confirmed via Ensembl), this class only needs to match
    convert_coordinate's return-shape contract for liftover_position's own
    logic (mapping-count handling, chromosome-change handling) to be
    tested without needing the real package or a real chain file."""

    def __init__(self, mapping: dict[tuple[str, int], list[tuple]]):
        self._mapping = mapping

    def convert_coordinate(self, chrom, pos):
        return self._mapping.get((chrom, pos), [])


class TestLiftoverPosition:
    def test_unambiguous_mapping_converts_correctly(self):
        """Real ground-truth case, 2026-07-31: rs6002851's VCF (GRCh37)
        position chr22:43,041,837 lifts to the real GRCh38 position
        chr22:42,645,831, confirmed independently via Ensembl."""
        from biolens.data.reference_genome import liftover_position

        lo = _FakeLiftOver({("chr22", 43041836): [("chr22", 42645830, "+", 3231095979)]})
        result = liftover_position(lo, "chr22", 43041837)
        assert result == 42645831

    def test_no_mapping_returns_none(self):
        from biolens.data.reference_genome import liftover_position

        lo = _FakeLiftOver({})
        assert liftover_position(lo, "chr22", 1000) is None

    def test_ambiguous_multi_mapping_returns_none(self):
        """A position that lifts to more than one target location is a
        real, if uncommon, outcome (reorganized/duplicated region between
        builds) — must not silently guess which one is right."""
        from biolens.data.reference_genome import liftover_position

        lo = _FakeLiftOver({
            ("chr22", 999): [
                ("chr22", 1000, "+", 1),
                ("chr22", 5000, "+", 1),
            ]
        })
        assert liftover_position(lo, "chr22", 1000) is None

    def test_mapping_to_different_chromosome_returns_none(self):
        """A liftOver that jumps to a different chromosome entirely is a
        real, if rare, outcome — the caller needs same-chromosome
        coordinates for a valid reference-genome window."""
        from biolens.data.reference_genome import liftover_position

        lo = _FakeLiftOver({("chr22", 999): [("chr21", 1000, "+", 1)]})
        assert liftover_position(lo, "chr22", 1000) is None

    def test_zero_based_conversion_at_the_boundary(self):
        """pyliftover's convert_coordinate is 0-based; liftover_position's
        own interface is 1-based (VCF convention) on both sides — position
        1 (1-based) must query offset 0 (0-based), and a 0-based lifted
        result of 0 must return back as 1-based position 1."""
        from biolens.data.reference_genome import liftover_position

        lo = _FakeLiftOver({("chr1", 0): [("chr1", 0, "+", 1)]})
        assert liftover_position(lo, "chr1", 1) == 1


class TestLoadLiftoverChain:
    def test_import_error_without_package(self, monkeypatch):
        """Forces the ImportError path via sys.modules rather than relying
        on pyliftover actually being absent from whatever environment runs
        this test — same real regression class as TestBorzoiBaseline's
        equivalent fix in test_mediation_baselines.py (2026-07-30): once a
        package is genuinely installed somewhere (e.g. the GPU cluster),
        a test relying on ambient absence silently stops testing what it
        claims to."""
        import sys

        from biolens.data.reference_genome import load_liftover_chain

        monkeypatch.setitem(sys.modules, "pyliftover", None)

        with pytest.raises(ImportError, match="pyliftover"):
            load_liftover_chain("/fake/path/hg19ToHg38.over.chain.gz")
