"""
Tests for the ENCODE cCRE genomic-annotation loader (the GENCODE/ENCODE
analog to GO-term probing).

label_dna_windows (the overlap/labeling algorithm) is pure Python — tested
directly with synthetic CCRE objects, no I/O. load_encode_ccres itself needs
a real bigBed file (pyBigWig only supports READING bigBed, not writing —
there's no way to synthesize a fixture file), so its one test is a real
network smoke test against the actual UCSC-hosted file, marked `integration`
like the existing ESM2 download tests — confirms this module's understanding
of the real BED-9+6 schema (verified once already, manually, against the
downloaded file; see the module's docstring) still holds.
"""

from __future__ import annotations

import pytest


def _ccre(chrom, start, end, accession, classes):
    from biolens.data.genomic_annotations import CCRE

    return CCRE(chrom=chrom, start=start, end=end, accession=accession, ccre_classes=frozenset(classes))


def _gene(gene_id, chrom, start, end, strand="+"):
    from biolens.data.genomic_annotations import GencodeGene

    return GencodeGene(gene_id=gene_id, chrom=chrom, start=start, end=end, strand=strand)


# ── label_dna_windows (pure Python — no I/O) ──────────────────────────────────

class TestLabelDnaWindows:
    def test_window_fully_containing_ccre_gets_its_labels(self):
        from biolens.data.genomic_annotations import label_dna_windows

        ccres = [_ccre("chr1", 1000, 1200, "EH1", {"PLS", "CTCF-bound"})]
        windows = [("win1", "chr1", 0, 5000)]

        labels = label_dna_windows(windows, ccres)
        assert labels["win1"] == {"PLS", "CTCF-bound"}

    def test_non_overlapping_window_gets_empty_set(self):
        from biolens.data.genomic_annotations import label_dna_windows

        ccres = [_ccre("chr1", 1000, 1200, "EH1", {"PLS"})]
        windows = [("win1", "chr1", 5000, 6000)]

        labels = label_dna_windows(windows, ccres)
        assert labels["win1"] == set()

    def test_partial_overlap_still_counts(self):
        """A cCRE that starts before the window and ends inside it (or vice
        versa) is a real overlap, not just full containment."""
        from biolens.data.genomic_annotations import label_dna_windows

        ccres = [_ccre("chr1", 900, 1100, "EH1", {"dELS"})]  # spans window's start
        windows = [("win1", "chr1", 1000, 2000)]

        labels = label_dna_windows(windows, ccres)
        assert labels["win1"] == {"dELS"}

    def test_adjacent_non_overlapping_intervals_excluded(self):
        """BED half-open convention: [start, end) — a cCRE ending exactly at
        the window's start does NOT overlap it."""
        from biolens.data.genomic_annotations import label_dna_windows

        ccres = [_ccre("chr1", 500, 1000, "EH1", {"PLS"})]  # ends exactly at window start
        windows = [("win1", "chr1", 1000, 2000)]

        labels = label_dna_windows(windows, ccres)
        assert labels["win1"] == set()

    def test_multiple_overlapping_ccres_union_labels(self):
        from biolens.data.genomic_annotations import label_dna_windows

        ccres = [
            _ccre("chr1", 1000, 1200, "EH1", {"PLS"}),
            _ccre("chr1", 1500, 1700, "EH2", {"dELS", "CTCF-bound"}),
        ]
        windows = [("win1", "chr1", 900, 2000)]

        labels = label_dna_windows(windows, ccres)
        assert labels["win1"] == {"PLS", "dELS", "CTCF-bound"}

    def test_different_chromosomes_dont_cross_contaminate(self):
        from biolens.data.genomic_annotations import label_dna_windows

        ccres = [_ccre("chr2", 1000, 1200, "EH1", {"PLS"})]
        windows = [("win1", "chr1", 1000, 1200)]  # same coordinates, different chrom

        labels = label_dna_windows(windows, ccres)
        assert labels["win1"] == set()

    def test_ccre_far_before_window_excluded_by_backward_scan_bound(self):
        """Regression guard for the backward-scan cutoff logic: a cCRE FAR
        before the window (beyond the max-plausible-length bound) must not
        be found, and one just within the bound must be — exercising the
        actual boundary the correctness argument in the docstring relies on."""
        from biolens.data.genomic_annotations import (
            _MAX_PLAUSIBLE_CCRE_LENGTH_BP,
            label_dna_windows,
        )

        window_start = 100_000
        far_ccre = _ccre(
            "chr1", window_start - _MAX_PLAUSIBLE_CCRE_LENGTH_BP - 10_000,
            window_start - _MAX_PLAUSIBLE_CCRE_LENGTH_BP - 9_999, "FAR", {"PLS"},
        )
        windows = [("win1", "chr1", window_start, window_start + 1000)]

        labels = label_dna_windows(windows, [far_ccre])
        assert labels["win1"] == set()

    def test_empty_ccre_list_gives_empty_labels_for_all_windows(self):
        from biolens.data.genomic_annotations import label_dna_windows

        windows = [("win1", "chr1", 0, 100), ("win2", "chr2", 0, 100)]
        labels = label_dna_windows(windows, [])
        assert labels == {"win1": set(), "win2": set()}

    def test_every_window_id_present_in_output(self):
        """Windows with no overlap still appear (as empty sets), not silently
        omitted — matches the probing engine's expectation that every
        sequence ID has a (possibly empty) label entry."""
        from biolens.data.genomic_annotations import label_dna_windows

        windows = [(f"win{i}", "chr1", i * 10000, i * 10000 + 100) for i in range(20)]
        labels = label_dna_windows(windows, [])
        assert set(labels.keys()) == {w[0] for w in windows}


# ── CCRE dataclass ─────────────────────────────────────────────────────────────

class TestCCRE:
    def test_length_property(self):
        c = _ccre("chr1", 1000, 1350, "EH1", {"PLS"})
        assert c.length == 350

    def test_frozen_and_hashable(self):
        c = _ccre("chr1", 1000, 1350, "EH1", {"PLS"})
        {c}  # must not raise — frozen dataclass with a frozenset field


# ── load_encode_ccres (real network — see module docstring) ──────────────────

@pytest.mark.integration
class TestLoadEncodeCcresReal:
    def test_real_bigbed_schema_matches_expectations(self, tmp_path):
        """Downloads (and caches in tmp_path) the real ENCODE cCRE bigBed
        file restricted to chr21 (one of the smallest human chromosomes, to
        keep this test's data transfer bounded) and checks the parsed
        schema matches what this module assumes — a real regression check
        against upstream data drift, not just our own parsing logic."""
        from biolens.data.genomic_annotations import load_encode_ccres

        ccres = load_encode_ccres(chroms={"chr21"}, data_dir=tmp_path)
        assert len(ccres) > 100  # chr21 alone has thousands of real cCREs
        assert all(c.chrom == "chr21" for c in ccres)
        assert all(c.end > c.start for c in ccres)
        assert all(c.accession.startswith("EH38E") for c in ccres)
        # Every real cCRE has at least one classification label.
        assert all(len(c.ccre_classes) > 0 for c in ccres)
        known_classes = {"PLS", "pELS", "dELS", "CTCF-bound", "DNase-H3K4me3", "CA-CTCF", "CA-H3K4me3", "CA-TF", "CA", "TF"}
        all_seen_classes = {cls for c in ccres for cls in c.ccre_classes}
        assert all_seen_classes & known_classes, (
            f"None of the expected cCRE classes found — schema may have drifted: {all_seen_classes}"
        )


# ── _parse_gtf_attribute (pure Python — no I/O) ────────────────────────────

class TestParseGtfAttribute:
    def test_extracts_gene_id(self):
        from biolens.data.genomic_annotations import _parse_gtf_attribute

        attrs = 'gene_id "ENSG00000131044.11"; gene_type "protein_coding"; gene_name "TAS2R4";'
        assert _parse_gtf_attribute(attrs, "gene_id") == "ENSG00000131044.11"

    def test_extracts_from_middle_of_field_list(self):
        from biolens.data.genomic_annotations import _parse_gtf_attribute

        attrs = 'gene_id "ENSG1.1"; transcript_id "ENST1.1"; gene_name "FOO";'
        assert _parse_gtf_attribute(attrs, "gene_name") == "FOO"

    def test_missing_key_returns_none(self):
        from biolens.data.genomic_annotations import _parse_gtf_attribute

        attrs = 'gene_id "ENSG1.1";'
        assert _parse_gtf_attribute(attrs, "nonexistent_key") is None

    def test_does_not_match_key_substring(self):
        """'gene_id' must not accidentally match inside e.g. 'havana_gene_id'."""
        from biolens.data.genomic_annotations import _parse_gtf_attribute

        attrs = 'havana_gene_id "OTTHUMG00000001.1"; gene_id "ENSG1.1";'
        assert _parse_gtf_attribute(attrs, "gene_id") == "ENSG1.1"


# ── load_gencode_genes (pure parsing logic, exercised via a synthetic GTF) ──

class TestLoadGencodeGenes:
    def _write_gtf(self, tmp_path, lines):
        import gzip

        path = tmp_path / "gencode.v50.basic.annotation.gtf.gz"
        with gzip.open(path, "wt") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def test_parses_gene_rows_only(self, tmp_path):
        from biolens.data.genomic_annotations import load_gencode_genes

        gtf_dir = tmp_path
        self._write_gtf(gtf_dir, [
            "##description: test",
            'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG1.1"; gene_name "A";',
            'chr1\tHAVANA\ttranscript\t1001\t1900\t.\t+\t.\tgene_id "ENSG1.1"; transcript_id "ENST1.1";',
            'chr1\tHAVANA\texon\t1001\t1200\t.\t+\t.\tgene_id "ENSG1.1"; transcript_id "ENST1.1";',
        ])
        genes = load_gencode_genes(data_dir=gtf_dir)
        assert len(genes) == 1
        assert genes[0].gene_id == "ENSG1.1"

    def test_converts_1based_closed_to_0based_half_open(self, tmp_path):
        from biolens.data.genomic_annotations import load_gencode_genes

        self._write_gtf(tmp_path, [
            'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG1.1";',
        ])
        genes = load_gencode_genes(data_dir=tmp_path)
        assert genes[0].start == 1000  # 1001 (1-based) -> 1000 (0-based)
        assert genes[0].end == 2000  # closed 2000 -> half-open end 2000

    def test_gene_ids_filter_restricts_output(self, tmp_path):
        from biolens.data.genomic_annotations import load_gencode_genes

        self._write_gtf(tmp_path, [
            'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG1.1";',
            'chr1\tHAVANA\tgene\t3001\t4000\t.\t-\t.\tgene_id "ENSG2.1";',
        ])
        genes = load_gencode_genes(gene_ids={"ENSG2.1"}, data_dir=tmp_path)
        assert len(genes) == 1
        assert genes[0].gene_id == "ENSG2.1"

    def test_strand_is_preserved(self, tmp_path):
        from biolens.data.genomic_annotations import load_gencode_genes

        self._write_gtf(tmp_path, [
            'chr1\tHAVANA\tgene\t1001\t2000\t.\t-\t.\tgene_id "ENSG1.1";',
        ])
        genes = load_gencode_genes(data_dir=tmp_path)
        assert genes[0].strand == "-"

    def test_rows_without_gene_id_are_skipped(self, tmp_path):
        from biolens.data.genomic_annotations import load_gencode_genes

        self._write_gtf(tmp_path, [
            'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_type "protein_coding";',
        ])
        genes = load_gencode_genes(data_dir=tmp_path)
        assert genes == []

    def test_gene_ids_filter_matches_across_version_suffix_mismatch(self, tmp_path):
        """Regression test for a real production failure (job 9701625,
        2026-07-25): the Geuvadis eQTL file's gene_id versions (e.g.
        ENSG00000000457.8) come from an older GENCODE/Ensembl build than
        whatever release this loader downloads (e.g. ENSG00000000457.16 in
        Release 50) — exact-string filtering silently excluded every single
        requested gene. Must match on the stable (unversioned) ID."""
        from biolens.data.genomic_annotations import load_gencode_genes

        self._write_gtf(tmp_path, [
            'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG00000000457.16";',
        ])
        genes = load_gencode_genes(gene_ids={"ENSG00000000457.8"}, data_dir=tmp_path)
        assert len(genes) == 1
        # Returned record keeps GENCODE's OWN version, not the requested one
        # — callers needing the original external ID back must remap via
        # stable_gene_id() themselves (see run_geuvadis_naive_probing.py).
        assert genes[0].gene_id == "ENSG00000000457.16"


class TestLoadGencodeGenesByExternalId:
    """Shared helper factored out (2026-07-29) from the identical remap
    logic previously duplicated in scripts/run_geuvadis_naive_probing.py
    and scripts/run_geuvadis_case_study.py."""

    def _write_gtf(self, tmp_path, lines):
        import gzip

        path = tmp_path / "gencode.v50.basic.annotation.gtf.gz"
        with gzip.open(path, "wt") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def test_keyed_by_external_id_despite_version_mismatch(self, tmp_path):
        from biolens.data.genomic_annotations import load_gencode_genes_by_external_id

        self._write_gtf(tmp_path, [
            'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG00000000457.16";',
        ])
        result = load_gencode_genes_by_external_id({"ENSG00000000457.8"}, data_dir=tmp_path)
        assert set(result.keys()) == {"ENSG00000000457.8"}
        # The value is the real GencodeGene record, still carrying GENCODE's
        # own version in its own gene_id field.
        assert result["ENSG00000000457.8"].gene_id == "ENSG00000000457.16"
        assert result["ENSG00000000457.8"].start == 1000
        assert result["ENSG00000000457.8"].end == 2000

    def test_unmatched_ids_omitted_not_raised(self, tmp_path):
        from biolens.data.genomic_annotations import load_gencode_genes_by_external_id

        self._write_gtf(tmp_path, [
            'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG1.1";',
        ])
        result = load_gencode_genes_by_external_id({"ENSG1.1", "ENSG_NOT_IN_GTF.1"}, data_dir=tmp_path)
        assert set(result.keys()) == {"ENSG1.1"}

    def test_empty_request_gives_empty_result(self, tmp_path):
        from biolens.data.genomic_annotations import load_gencode_genes_by_external_id

        self._write_gtf(tmp_path, [
            'chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG1.1";',
        ])
        assert load_gencode_genes_by_external_id(set(), data_dir=tmp_path) == {}


# ── stable_gene_id (pure Python — no I/O) ──────────────────────────────────

class TestStableGeneId:
    def test_strips_version_suffix(self):
        from biolens.data.genomic_annotations import stable_gene_id

        assert stable_gene_id("ENSG00000131044.11") == "ENSG00000131044"

    def test_no_version_suffix_is_a_no_op(self):
        from biolens.data.genomic_annotations import stable_gene_id

        assert stable_gene_id("ENSG00000131044") == "ENSG00000131044"


# ── label_dna_windows_by_gene (pure Python — no I/O) ───────────────────────

class TestLabelDnaWindowsByGene:
    def test_reshapes_gene_windows_into_label_dict(self):
        from biolens.data.genomic_annotations import label_dna_windows_by_gene

        windows = [
            ("ENSG1.1:chr1:0-100", "chr1", 0, 100, "ENSG1.1"),
            ("ENSG1.1:chr1:100-200", "chr1", 100, 200, "ENSG1.1"),
            ("ENSG2.1:chr2:0-100", "chr2", 0, 100, "ENSG2.1"),
        ]
        labels = label_dna_windows_by_gene(windows)
        assert labels == {
            "ENSG1.1:chr1:0-100": {"ENSG1.1"},
            "ENSG1.1:chr1:100-200": {"ENSG1.1"},
            "ENSG2.1:chr2:0-100": {"ENSG2.1"},
        }

    def test_empty_input_gives_empty_output(self):
        from biolens.data.genomic_annotations import label_dna_windows_by_gene

        assert label_dna_windows_by_gene([]) == {}


# ── find_overlapping_genes (pure Python — no I/O) ──────────────────────────

class TestFindOverlappingGenes:
    def test_window_inside_gene_body_matches(self):
        from biolens.data.genomic_annotations import find_overlapping_genes

        genes = [_gene("ENSG1.1", "chr1", 1000, 5000)]
        windows = [("win1", "chr1", 2000, 2100)]

        result = find_overlapping_genes(windows, genes)
        assert [g.gene_id for g in result["win1"]] == ["ENSG1.1"]

    def test_window_outside_any_gene_gives_empty_list(self):
        from biolens.data.genomic_annotations import find_overlapping_genes

        genes = [_gene("ENSG1.1", "chr1", 1000, 5000)]
        windows = [("win1", "chr1", 10000, 10100)]

        result = find_overlapping_genes(windows, genes)
        assert result["win1"] == []

    def test_window_spanning_two_overlapping_genes_returns_both(self):
        from biolens.data.genomic_annotations import find_overlapping_genes

        genes = [
            _gene("ENSG1.1", "chr1", 1000, 3000),
            _gene("ENSG2.1", "chr1", 2500, 4500, strand="-"),  # overlapping gene bodies
        ]
        windows = [("win1", "chr1", 2600, 2700)]

        result = find_overlapping_genes(windows, genes)
        assert {g.gene_id for g in result["win1"]} == {"ENSG1.1", "ENSG2.1"}

    def test_different_chromosomes_dont_cross_contaminate(self):
        from biolens.data.genomic_annotations import find_overlapping_genes

        genes = [_gene("ENSG1.1", "chr2", 1000, 5000)]
        windows = [("win1", "chr1", 1000, 5000)]  # same coords, different chrom

        result = find_overlapping_genes(windows, genes)
        assert result["win1"] == []

    def test_adjacent_non_overlapping_excluded(self):
        """Half-open convention: a gene ending exactly at the window's start
        does NOT overlap it."""
        from biolens.data.genomic_annotations import find_overlapping_genes

        genes = [_gene("ENSG1.1", "chr1", 0, 1000)]
        windows = [("win1", "chr1", 1000, 2000)]

        result = find_overlapping_genes(windows, genes)
        assert result["win1"] == []

    def test_large_gene_beyond_ccre_bound_still_found(self):
        """A real gene body up to _MAX_PLAUSIBLE_GENE_LENGTH_BP long, whose
        start is far before the window (further than the cCRE bound would
        allow), must still be found — this is the whole reason
        find_overlapping_genes needs its own, much larger, bound rather than
        reusing _MAX_PLAUSIBLE_CCRE_LENGTH_BP."""
        from biolens.data.genomic_annotations import (
            _MAX_PLAUSIBLE_CCRE_LENGTH_BP,
            find_overlapping_genes,
        )

        gene_start = 0
        window_start = _MAX_PLAUSIBLE_CCRE_LENGTH_BP + 100_000  # far beyond cCRE bound
        genes = [_gene("BIGGENE.1", "chr1", gene_start, window_start + 1000)]
        windows = [("win1", "chr1", window_start, window_start + 100)]

        result = find_overlapping_genes(windows, genes)
        assert [g.gene_id for g in result["win1"]] == ["BIGGENE.1"]

    def test_gene_far_beyond_max_plausible_length_excluded(self):
        """Regression guard for the backward-scan cutoff: a "gene" starting
        further back than _MAX_PLAUSIBLE_GENE_LENGTH_BP before the window
        (larger than any real human gene) must not be found."""
        from biolens.data.genomic_annotations import (
            _MAX_PLAUSIBLE_GENE_LENGTH_BP,
            find_overlapping_genes,
        )

        window_start = 10_000_000
        far_gene = _gene(
            "FAR.1", "chr1",
            window_start - _MAX_PLAUSIBLE_GENE_LENGTH_BP - 10_000,
            window_start - _MAX_PLAUSIBLE_GENE_LENGTH_BP - 9_000,
        )
        windows = [("win1", "chr1", window_start, window_start + 100)]

        result = find_overlapping_genes(windows, [far_gene])
        assert result["win1"] == []

    def test_empty_gene_list_gives_empty_lists_for_all_windows(self):
        from biolens.data.genomic_annotations import find_overlapping_genes

        windows = [("win1", "chr1", 0, 100), ("win2", "chr2", 0, 100)]
        result = find_overlapping_genes(windows, [])
        assert result == {"win1": [], "win2": []}

    def test_every_window_id_present_in_output(self):
        from biolens.data.genomic_annotations import find_overlapping_genes

        windows = [(f"win{i}", "chr1", i * 10000, i * 10000 + 100) for i in range(10)]
        result = find_overlapping_genes(windows, [])
        assert set(result.keys()) == {w[0] for w in windows}


@pytest.mark.integration
class TestLoadGencodeGenesReal:
    def test_real_gtf_schema_matches_expectations(self, tmp_path):
        """Downloads (and caches in tmp_path) the real GENCODE basic
        annotation GTF, restricted via gene_ids to one well-known real gene
        (TP53), and checks the parsed schema matches what this module
        assumes — a real regression check against upstream data drift."""
        from biolens.data.genomic_annotations import load_gencode_genes

        # TP53's stable (unversioned) Ensembl ID is ENSG00000141510 — pass it
        # through gene_ids so this test exercises the actual filtered code
        # path every real caller uses (run_geuvadis_naive_probing.py,
        # run_geuvadis_case_study.py), rather than loading and materializing
        # every gene in GENCODE just to filter by prefix afterward. The
        # request has no version suffix; load_gencode_genes' own
        # stable_gene_id() matching handles the version drift across
        # releases, so this doesn't need to know GENCODE's current version.
        genes = load_gencode_genes(gene_ids={"ENSG00000141510"}, data_dir=tmp_path)
        assert len(genes) == 1
        assert genes[0].gene_id.startswith("ENSG00000141510")
        assert genes[0].chrom == "chr17"
        assert genes[0].end > genes[0].start
        assert genes[0].strand in {"+", "-"}
