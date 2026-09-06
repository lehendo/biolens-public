"""
Tests for Geuvadis case study data loading (biolens.data.geuvadis).

Uses a small, real, valid VCF file (written to a temp path, parsed by the
real cyvcf2/htslib stack — not mocked) to pin the gt_types/gts012 genotype-
dosage encoding this module's correctness depends on (see module docstring
for why getting this wrong would silently produce inverted dosages).

The VCF fixture and the eQTL/expression fixtures deliberately use the REAL
public files' actual conventions, confirmed by directly fetching and
inspecting them on 2026-07-03 (not assumed): 1000 Genomes VCFs and Geuvadis'
own eQTL files use BARE chromosome names ("22"), while this project's
reference-genome FASTAs use "chr"-prefixed names ("chr22") — these tests
pin the normalization that reconciles the two conventions.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest


def _write_vcf(tmp_path: Path) -> Path:
    """
    3 samples x 3 variants, using BARE chromosome names ("1"), matching the
    real 1000 Genomes VCF convention confirmed directly against the public
    release (`ALL.chr22...vcf.gz`'s actual #CHROM column contains "22", not
    "chr22").
      variant1 (1:1000 A>G): S1=0/0 (HOM_REF), S2=0/1 (HET), S3=1/1 (HOM_ALT)
      variant2 (1:2000 C>T): S1=./. (missing), S2=1/1, S3=0/0
      variant3 (1:3000 G>A,T): multi-allelic — must be skipped entirely
    """
    vcf_content = dedent(
        """\
        ##fileformat=VCFv4.2
        ##contig=<ID=1,length=248956422>
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3
        1\t1000\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1
        1\t2000\t.\tC\tT\t.\tPASS\t.\tGT\t./.\t1/1\t0/0
        1\t3000\t.\tG\tA,T\t.\tPASS\t.\tGT\t0/1\t0/2\t1/1
        """
    )
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(vcf_content)
    return vcf_path


class TestChromNormalization:
    def test_normalize_adds_prefix_when_missing(self):
        from biolens.data.geuvadis import _normalize_chrom

        assert _normalize_chrom("22") == "chr22"
        assert _normalize_chrom("X") == "chrX"

    def test_normalize_is_idempotent(self):
        from biolens.data.geuvadis import _normalize_chrom

        assert _normalize_chrom("chr22") == "chr22"

    def test_strip_removes_prefix_when_present(self):
        from biolens.data.geuvadis import _strip_chrom_prefix

        assert _strip_chrom_prefix("chr22") == "22"

    def test_strip_is_idempotent(self):
        from biolens.data.geuvadis import _strip_chrom_prefix

        assert _strip_chrom_prefix("22") == "22"


class TestLoadVariantGenotypes:
    def test_output_chrom_is_normalized_to_chr_prefixed(self, tmp_path):
        """The real VCF has bare '1' in its CHROM field; this module's
        output must be 'chr1' to match the reference-genome convention used
        elsewhere in the pipeline."""
        from biolens.data.geuvadis import load_variant_genotypes

        vcf_path = _write_vcf(tmp_path)
        results = load_variant_genotypes(vcf_path, variant_positions=[("chr1", 1000)])

        assert "chr1:1000:A:G" in results
        assert results["chr1:1000:A:G"].chrom == "chr1"

    def test_accepts_bare_chrom_in_query_too(self, tmp_path):
        """Callers may pass either 'chr1' or '1' as the query chrom —
        both must match the same real (bare-named) VCF records."""
        from biolens.data.geuvadis import load_variant_genotypes

        vcf_path = _write_vcf(tmp_path)
        results = load_variant_genotypes(vcf_path, variant_positions=[("1", 1000)])

        assert "chr1:1000:A:G" in results

    def test_dosage_encoding_matches_gts012(self, tmp_path):
        """Regression pin for the gts012=True encoding this whole module
        depends on: HOM_REF=0, HET=1, HOM_ALT=2 — getting this wrong (e.g.
        forgetting gts012=True) would silently invert or corrupt dosages."""
        from biolens.data.geuvadis import load_variant_genotypes

        vcf_path = _write_vcf(tmp_path)
        results = load_variant_genotypes(vcf_path, variant_positions=[("chr1", 1000)])

        variant = results["chr1:1000:A:G"]
        assert variant.dosage_by_sample["S1"] == 0  # 0/0 -> HOM_REF
        assert variant.dosage_by_sample["S2"] == 1  # 0/1 -> HET
        assert variant.dosage_by_sample["S3"] == 2  # 1/1 -> HOM_ALT

    def test_missing_genotype_excluded(self, tmp_path):
        from biolens.data.geuvadis import load_variant_genotypes

        vcf_path = _write_vcf(tmp_path)
        results = load_variant_genotypes(vcf_path, variant_positions=[("chr1", 2000)])

        variant = results["chr1:2000:C:T"]
        assert "S1" not in variant.dosage_by_sample  # ./. excluded
        assert variant.dosage_by_sample["S2"] == 2
        assert variant.dosage_by_sample["S3"] == 0

    def test_multiallelic_site_skipped(self, tmp_path):
        from biolens.data.geuvadis import load_variant_genotypes

        vcf_path = _write_vcf(tmp_path)
        results = load_variant_genotypes(vcf_path, variant_positions=[("chr1", 3000)])
        assert len(results) == 0

    def test_restricts_to_requested_positions_only(self, tmp_path):
        from biolens.data.geuvadis import load_variant_genotypes

        vcf_path = _write_vcf(tmp_path)
        results = load_variant_genotypes(vcf_path, variant_positions=[("chr1", 1000)])
        assert set(results.keys()) == {"chr1:1000:A:G"}

    def test_variant_not_in_vcf_simply_absent(self, tmp_path):
        from biolens.data.geuvadis import load_variant_genotypes

        vcf_path = _write_vcf(tmp_path)
        results = load_variant_genotypes(
            vcf_path, variant_positions=[("chr1", 1000), ("chr1", 99999)]
        )
        assert "chr1:1000:A:G" in results
        assert len(results) == 1  # the nonexistent one is just absent, not an error


class TestLoadExpressionMatrix:
    def test_loads_and_indexes_by_target_id(self, tmp_path):
        """Real Geuvadis quantification files use 'TargetID' as the unit-ID
        column (confirmed against the real GD462.GeneQuantRPKM... file and
        its README), not 'gene_id' — the earlier, incorrect assumption."""
        from biolens.data.geuvadis import load_expression_matrix

        path = tmp_path / "expr.tsv"
        path.write_text(
            "TargetID\tGene_Symbol\tChr\tCoord\tS1\tS2\tS3\n"
            "ENSG001.1\tENSG001.1\t1\t1000\t5.2\t3.1\t8.7\n"
            "ENSG002.3\tENSG002.3\t2\t5000\t1.0\t2.0\t3.0\n"
        )

        df = load_expression_matrix(path)
        assert list(df.index) == ["ENSG001.1", "ENSG002.3"]
        assert df.loc["ENSG001.1", "S2"] == pytest.approx(3.1)

    def test_custom_gene_id_col_still_supported(self, tmp_path):
        from biolens.data.geuvadis import load_expression_matrix

        path = tmp_path / "expr.tsv"
        path.write_text("gene_id\tS1\nENSG001\t5.2\n")

        df = load_expression_matrix(path, gene_id_col="gene_id")
        assert list(df.index) == ["ENSG001"]


class TestLoadPublishedEqtlCandidates:
    def _write_real_format_eqtl_file(self, tmp_path: Path) -> Path:
        """Real format: no header, 12 tab-separated columns, bare chrom
        names — exactly matching EUR373.gene.cis.FDR5.best.rs137.txt.gz as
        documented in GeuvadisRNASeqAnalysisFiles_README.txt and confirmed
        against the live file."""
        path = tmp_path / "eqtls.txt"
        path.write_text(
            "snp_20_30533949\t-\tENSG00000131044.11\tENSG00000131044.11\t20\t20\t"
            "30533949\t30458505\t75444\t-0.416416099857943\t4.62871458260775e-17\t16.3345395980074\n"
            "rs1985842\t-\tENSG00000198951.6\tENSG00000198951.6\t22\t22\t"
            "42523409\t42466846\t56563\t0.299312531489249\t3.73521220716277e-09\t8.42768471974368\n"
        )
        return path

    def test_parses_real_format_correctly(self, tmp_path):
        from biolens.data.geuvadis import load_published_eqtl_candidates

        path = self._write_real_format_eqtl_file(tmp_path)
        candidates = load_published_eqtl_candidates(path)

        assert len(candidates) == 2
        assert candidates[0].variant_id == "snp_20_30533949"
        assert candidates[0].chrom == "chr20"  # normalized from bare "20"
        assert candidates[0].pos == 30533949
        assert candidates[0].gene_id == "ENSG00000131044.11"

    def test_second_row_parsed_correctly(self, tmp_path):
        from biolens.data.geuvadis import load_published_eqtl_candidates

        path = self._write_real_format_eqtl_file(tmp_path)
        candidates = load_published_eqtl_candidates(path)

        assert candidates[1].variant_id == "rs1985842"
        assert candidates[1].chrom == "chr22"
        assert candidates[1].pos == 42523409
        assert candidates[1].gene_id == "ENSG00000198951.6"

    def test_no_header_row_consumed_as_data(self, tmp_path):
        """Regression guard: the real file has NO header — if this function
        ever assumed one again, the first real eQTL row would be silently
        dropped as a header."""
        from biolens.data.geuvadis import load_published_eqtl_candidates

        path = self._write_real_format_eqtl_file(tmp_path)
        candidates = load_published_eqtl_candidates(path)
        assert len(candidates) == 2  # both real data rows present, none consumed as header


class TestBuildMediationArrays:
    def test_aligns_matched_samples(self, tmp_path):
        from biolens.data.geuvadis import VariantGenotypes, build_mediation_arrays

        genotypes = VariantGenotypes(
            variant_id="chr1:1000:A:G", chrom="chr1", pos=1000, ref="A", alt="G",
            dosage_by_sample={"S1": 0, "S2": 1, "S3": 2},
        )
        expression = pd.DataFrame(
            {"S1": [5.0], "S2": [3.0], "S3": [8.0]}, index=["ENSG001"]
        )

        treatment, outcome, samples = build_mediation_arrays(genotypes, expression, "ENSG001")
        assert list(treatment) == [0.0, 1.0, 2.0]
        assert list(outcome) == [5.0, 3.0, 8.0]
        assert samples == ["S1", "S2", "S3"]

    def test_restricts_to_samples_present_in_both(self, tmp_path):
        """A real, common data-quality case: genotyped and RNA-seq'd sample
        sets don't match exactly."""
        from biolens.data.geuvadis import VariantGenotypes, build_mediation_arrays

        genotypes = VariantGenotypes(
            variant_id="chr1:1000:A:G", chrom="chr1", pos=1000, ref="A", alt="G",
            dosage_by_sample={"S1": 0, "S2": 1, "S4": 2},  # S4 has no expression data
        )
        expression = pd.DataFrame(
            {"S1": [5.0], "S2": [3.0], "S3": [8.0]}, index=["ENSG001"]  # S3 has no genotype
        )

        treatment, outcome, samples = build_mediation_arrays(genotypes, expression, "ENSG001")
        assert samples == ["S1", "S2"]  # only the intersection

    def test_missing_gene_raises_keyerror(self):
        from biolens.data.geuvadis import VariantGenotypes, build_mediation_arrays

        genotypes = VariantGenotypes(
            variant_id="v1", chrom="chr1", pos=1, ref="A", alt="G",
            dosage_by_sample={"S1": 0},
        )
        expression = pd.DataFrame({"S1": [5.0]}, index=["ENSG001"])

        with pytest.raises(KeyError):
            build_mediation_arrays(genotypes, expression, "NONEXISTENT_GENE")

    def test_no_common_samples_raises_value_error(self):
        from biolens.data.geuvadis import VariantGenotypes, build_mediation_arrays

        genotypes = VariantGenotypes(
            variant_id="v1", chrom="chr1", pos=1, ref="A", alt="G",
            dosage_by_sample={"S1": 0},
        )
        expression = pd.DataFrame({"S99": [5.0]}, index=["ENSG001"])

        with pytest.raises(ValueError, match="No samples"):
            build_mediation_arrays(genotypes, expression, "ENSG001")


class TestPopulationLabels:
    def _write_panel(self, tmp_path: Path) -> Path:
        """Real column format confirmed against 1000 Genomes' own published
        integrated_call_samples_v3.20130502.ALL.panel: tab-separated, header
        row 'sample  pop  super_pop  gender'."""
        path = tmp_path / "panel.tsv"
        path.write_text(
            "sample\tpop\tsuper_pop\tgender\n"
            "S1\tGBR\tEUR\tfemale\n"
            "S2\tFIN\tEUR\tmale\n"
            "S3\tYRI\tAFR\tfemale\n"
            "S4\tCHB\tEAS\tmale\n"
        )
        return path

    def test_loads_population_and_superpopulation(self, tmp_path):
        from biolens.data.geuvadis import load_population_labels

        labels = load_population_labels(self._write_panel(tmp_path))
        assert labels["S1"].population == "GBR"
        assert labels["S1"].superpopulation == "EUR"
        assert labels["S3"].superpopulation == "AFR"

    def test_every_panel_sample_present(self, tmp_path):
        from biolens.data.geuvadis import load_population_labels

        labels = load_population_labels(self._write_panel(tmp_path))
        assert set(labels.keys()) == {"S1", "S2", "S3", "S4"}


class TestBuildAncestryCovariates:
    def _labels(self):
        from biolens.data.geuvadis import PopulationLabel

        return {
            "S1": PopulationLabel(population="GBR", superpopulation="EUR"),
            "S2": PopulationLabel(population="FIN", superpopulation="EUR"),
            "S3": PopulationLabel(population="YRI", superpopulation="AFR"),
            "S4": PopulationLabel(population="CHB", superpopulation="EAS"),
        }

    def test_one_hot_encodes_superpopulation_dropping_one_level(self):
        from biolens.data.geuvadis import build_ancestry_covariates

        samples = ["S1", "S2", "S3", "S4"]
        covariates = build_ancestry_covariates(samples, self._labels())

        # 3 distinct superpopulations present (EUR, AFR, EAS) -> 2 dummy
        # columns (one dropped to avoid collinearity with the intercept).
        assert covariates.shape == (4, 2)

    def test_row_order_matches_input_samples(self):
        from biolens.data.geuvadis import build_ancestry_covariates

        samples = ["S3", "S1"]  # AFR, EUR — deliberately not panel order
        covariates = build_ancestry_covariates(samples, self._labels())
        # The two rows must differ (different superpopulations) and be in
        # samples' order, not the label dict's insertion order.
        assert not covariates.iloc[0].equals(covariates.iloc[1])

    def test_missing_sample_raises_keyerror(self):
        from biolens.data.geuvadis import build_ancestry_covariates

        with pytest.raises(KeyError, match="no population label"):
            build_ancestry_covariates(["S1", "S_UNKNOWN"], self._labels())

    def test_output_is_directly_usable_as_ols_covariates(self):
        """The whole point: this DataFrame must work as-is with
        run_ikt_mediation(covariates=...) / sensitivity_analysis(covariates=...),
        which both do `data[col] = covariates[col].to_numpy()` per column."""
        from biolens.data.geuvadis import build_ancestry_covariates

        samples = ["S1", "S2", "S3", "S4"]
        covariates = build_ancestry_covariates(samples, self._labels())
        for col in covariates.columns:
            arr = covariates[col].to_numpy()
            assert arr.dtype.kind == "f"
            assert len(arr) == 4


class TestSummarizePopulationComposition:
    def test_counts_by_superpopulation(self):
        from biolens.data.geuvadis import PopulationLabel, summarize_population_composition

        labels = {
            "S1": PopulationLabel(population="GBR", superpopulation="EUR"),
            "S2": PopulationLabel(population="FIN", superpopulation="EUR"),
            "S3": PopulationLabel(population="YRI", superpopulation="AFR"),
        }
        counts = summarize_population_composition(["S1", "S2", "S3", "S1"], labels)
        assert counts == {"EUR": 3, "AFR": 1}


def _write_phased_vcf(tmp_path: Path) -> Path:
    """
    3 samples x 4 variants, PHASED genotypes ('|' separator, matching the
    real 1000 Genomes Phase 3 VCF — SHAPEIT2-phased, confirmed directly on
    the server 2026-08-XX via cyvcf2's own variant.genotypes: real entries
    return [allele1, allele2, True]). Bare chromosome name ("1"), matching
    the real-file convention already pinned by _write_vcf above.
      v1 (1:1000 A>G): S1=0|0, S2=0|1 (het, alt on haplotype 1 only), S3=1|1
      v2 (1:1010 C>T): S1=./. (missing), S2=1|0 (het, alt on haplotype 0), S3=0|0
      v3 (1:1020 G>A,T): multi-allelic -- must be skipped entirely
      v4 (1:2000 T>C): far outside any test window -- must never appear
      v5 (1:1005 CT>C): a biallelic INDEL (single ALT allele, but REF is
          2 bases) -- must be skipped entirely, same as v3. Real, not
          hypothetical: this exact shape (biallelic but multi-base REF)
          passed the old single-ALT-allele-only filter and crashed
          build_haplotype_sequence on real chr22 data (job 9755424,
          2026-08-03) with a spurious "reference genome mismatch" (the
          multi-base REF can never equal the single window base being
          compared against, regardless of whether the genome build is
          actually correct there).
    """
    vcf_content = dedent(
        """\
        ##fileformat=VCFv4.2
        ##contig=<ID=1,length=248956422>
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3
        1\t1000\t.\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1
        1\t1005\t.\tCT\tC\t.\tPASS\t.\tGT\t0|1\t1|1\t0|0
        1\t1010\t.\tC\tT\t.\tPASS\t.\tGT\t.|.\t1|0\t0|0
        1\t1020\t.\tG\tA,T\t.\tPASS\t.\tGT\t0|1\t0|2\t1|1
        1\t2000\t.\tT\tC\t.\tPASS\t.\tGT\t1|1\t0|0\t0|1
        """
    )
    vcf_path = tmp_path / "test_phased.vcf"
    vcf_path.write_text(vcf_content)
    return vcf_path


class TestLoadWindowGenotypes:
    def test_collects_variants_within_window(self, tmp_path):
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(vcf_path, [("locusA", "chr1", 990, 1015)])

        positions = [g.pos for g in result["locusA"]]
        # v5 (indel, 1005) and v3 (multiallelic) and v4 (out of range) all excluded
        assert positions == [1000, 1010]

    def test_multiallelic_site_skipped(self, tmp_path):
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(vcf_path, [("locusA", "chr1", 990, 1025)])

        assert 1020 not in [g.pos for g in result["locusA"]]

    def test_indel_skipped(self, tmp_path):
        """Regression test for a real production crash (job 9755424,
        2026-08-03): a biallelic indel (single ALT allele, but REF/ALT not
        single bases) passed the old filter here (which only checked
        len(variant.ALT) != 1, i.e. "biallelic," not "SNV") and reached
        build_haplotype_sequence, which raised a spurious "reference genome
        mismatch" comparing the indel's multi-base REF against exactly one
        base of the window sequence. Must be skipped here, same as a
        genuinely multi-allelic site."""
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(vcf_path, [("locusA", "chr1", 990, 1015)])

        assert 1005 not in [g.pos for g in result["locusA"]]

    def test_variant_outside_any_window_excluded(self, tmp_path):
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(vcf_path, [("locusA", "chr1", 990, 1015)])

        assert 2000 not in [g.pos for g in result["locusA"]]

    def test_phased_alleles_parsed_correctly(self, tmp_path):
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(vcf_path, [("locusA", "chr1", 990, 1015)])

        v1 = next(g for g in result["locusA"] if g.pos == 1000)
        assert v1.alleles_by_sample["S1"] == (0, 0)
        assert v1.alleles_by_sample["S2"] == (0, 1)  # het, alt on haplotype 1
        assert v1.alleles_by_sample["S3"] == (1, 1)

    def test_missing_genotype_excluded_from_alleles(self, tmp_path):
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(vcf_path, [("locusA", "chr1", 990, 1015)])

        v2 = next(g for g in result["locusA"] if g.pos == 1010)
        assert "S1" not in v2.alleles_by_sample  # missing call
        assert v2.alleles_by_sample["S2"] == (1, 0)

    def test_different_chromosomes_dont_cross_contaminate(self, tmp_path):
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(vcf_path, [("locusA", "chr2", 990, 1015)])

        assert result["locusA"] == []

    def test_every_requested_locus_key_present(self, tmp_path):
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(
            vcf_path, [("locusA", "chr1", 990, 1015), ("locusB", "chr9", 0, 100)]
        )
        assert set(result.keys()) == {"locusA", "locusB"}
        assert result["locusB"] == []

    def test_overlapping_windows_both_get_the_variant(self, tmp_path):
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(
            vcf_path,
            [("locusA", "chr1", 990, 1015), ("locusB", "chr1", 995, 1020)],
        )
        assert 1000 in [g.pos for g in result["locusA"]]
        assert 1000 in [g.pos for g in result["locusB"]]

    def test_results_sorted_by_position(self, tmp_path):
        from biolens.data.geuvadis import load_window_genotypes

        vcf_path = _write_phased_vcf(tmp_path)
        result = load_window_genotypes(vcf_path, [("locusA", "chr1", 990, 1015)])
        positions = [g.pos for g in result["locusA"]]
        assert positions == sorted(positions)


class TestBuildHaplotypeSequence:
    def _variant(self, pos, ref, alt, alleles_by_sample):
        from biolens.data.geuvadis import PhasedGenotype

        return PhasedGenotype(pos=pos, ref=ref, alt=alt, alleles_by_sample=alleles_by_sample)

    def test_homozygous_ref_no_change(self):
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "AAAAAAAAAA"
        variants = [self._variant(1005, "A", "G", {"S1": (0, 0)})]
        result = build_haplotype_sequence(ref_seq, window_start=1000, variants=variants,
                                           sample_id="S1", haplotype_index=0)
        assert result == ref_seq

    def test_heterozygous_differs_by_haplotype(self):
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "AAAAAAAAAA"
        variants = [self._variant(1005, "A", "G", {"S1": (0, 1)})]  # alt on haplotype 1 only
        hap0 = build_haplotype_sequence(ref_seq, 1000, variants, "S1", haplotype_index=0)
        hap1 = build_haplotype_sequence(ref_seq, 1000, variants, "S1", haplotype_index=1)
        assert hap0 == "AAAAAAAAAA"
        assert hap1 == "AAAAGAAAAA"  # offset 4 == (1005-1)-1000, 0-indexed

    def test_homozygous_alt_substitutes_both_haplotypes(self):
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "AAAAAAAAAA"
        variants = [self._variant(1005, "A", "G", {"S1": (1, 1)})]
        hap0 = build_haplotype_sequence(ref_seq, 1000, variants, "S1", haplotype_index=0)
        hap1 = build_haplotype_sequence(ref_seq, 1000, variants, "S1", haplotype_index=1)
        assert hap0 == hap1 == "AAAAGAAAAA"

    def test_missing_genotype_leaves_reference_base(self):
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "AAAAAAAAAA"
        variants = [self._variant(1005, "A", "G", {"S2": (1, 1)})]  # no entry for S1
        result = build_haplotype_sequence(ref_seq, 1000, variants, "S1", haplotype_index=0)
        assert result == ref_seq

    def test_variant_outside_range_skipped(self):
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "AAAAAAAAAA"  # covers [1000, 1010)
        variants = [self._variant(2000, "A", "G", {"S1": (1, 1)})]
        result = build_haplotype_sequence(ref_seq, 1000, variants, "S1", haplotype_index=0)
        assert result == ref_seq

    def test_multiple_variants_substituted_together(self):
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "AAAAAAAAAA"
        variants = [
            self._variant(1002, "A", "C", {"S1": (1, 1)}),
            self._variant(1007, "A", "T", {"S1": (1, 1)}),
        ]
        result = build_haplotype_sequence(ref_seq, 1000, variants, "S1", haplotype_index=0)
        assert result == "ACAAAATAAA"  # offsets 1 and 6, 0-indexed

    def test_reference_mismatch_skipped_not_raised(self):
        """Regression test for a real production crash (job 9901091,
        2026-08-12): a ref-genome mismatch on one of many NEARBY variants
        (not the lead SNP) used to raise ValueError, which crashed an
        entire multi-hour job the first time any single nearby variant --
        out of hundreds per locus -- had an isolated liftOver imprecision.
        Must be skipped (left as the reference base), not raised."""
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "AAAAAAAAAA"
        variants = [self._variant(1005, "C", "G", {"S1": (1, 1)})]  # real base is 'A', not 'C'
        result = build_haplotype_sequence(ref_seq, 1000, variants, "S1", haplotype_index=0)
        assert result == "AAAAAAAAAA"  # mismatched variant skipped -- reference base kept

    def test_reference_mismatch_does_not_affect_other_variants(self):
        """The whole point of skipping rather than raising: one bad variant
        among many must not prevent the others from being applied."""
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "AAAAAAAAAA"
        variants = [
            self._variant(1002, "A", "C", {"S1": (1, 1)}),  # valid, offset 1
            self._variant(1005, "C", "G", {"S1": (1, 1)}),  # mismatch (real base 'A'), offset 4
            self._variant(1007, "A", "T", {"S1": (1, 1)}),  # valid, offset 6
        ]
        result = build_haplotype_sequence(ref_seq, 1000, variants, "S1", haplotype_index=0)
        assert result == "ACAAAATAAA"  # offsets 1 and 6 substituted; offset 4 skipped, stays 'A'

    def test_output_length_matches_input(self):
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "ACGTACGTAC"
        result = build_haplotype_sequence(ref_seq, 1000, [], "S1", haplotype_index=0)
        assert len(result) == len(ref_seq)

    def test_output_is_uppercase(self):
        from biolens.data.geuvadis import build_haplotype_sequence

        ref_seq = "acgtacgtac"
        result = build_haplotype_sequence(ref_seq, 1000, [], "S1", haplotype_index=0)
        assert result == "ACGTACGTAC"
