"""
Tests for biolens.eval.genomic_verification — the genomic-domain analog of
the ESM2/UniProt verification pattern, checking a naive-probing feature's
claimed gene against its real top-activating windows genome-wide.

verify_genomic_feature is pure Python (no I/O, no cache/SAE needed) —
tested directly with synthetic GencodeGene/window fixtures.
"""

from __future__ import annotations


def _gene(gene_id, chrom, start, end, strand="+"):
    from biolens.data.genomic_annotations import GencodeGene

    return GencodeGene(gene_id=gene_id, chrom=chrom, start=start, end=end, strand=strand)


class TestVerifyGenomicFeature:
    def test_all_top_windows_inside_claimed_gene_is_confirmed(self):
        from biolens.eval.genomic_verification import verify_genomic_feature

        genes = [_gene("ENSG1.1", "chr1", 0, 10000)]
        windows = [
            ("chr1:1000-2000", "chr1", 1000, 2000),
            ("chr1:2000-3000", "chr1", 2000, 3000),
            ("chr1:3000-4000", "chr1", 3000, 4000),
        ]

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=["ENSG1.1"], ranked_windows=windows, genes=genes,
        )
        assert result.status == "confirmed"
        assert result.match_rate == 1.0
        assert result.matched_windows == [w[0] for w in windows]

    def test_no_top_windows_touching_claimed_gene_is_spurious(self):
        from biolens.eval.genomic_verification import verify_genomic_feature

        genes = [_gene("ENSG1.1", "chr1", 0, 10000)]
        windows = [
            ("chr5:1000-2000", "chr5", 1000, 2000),
            ("chr7:5000-6000", "chr7", 5000, 6000),
        ]

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=["ENSG1.1"], ranked_windows=windows, genes=genes,
        )
        assert result.status == "spurious"
        assert result.match_rate == 0.0
        assert result.matched_windows == []

    def test_minority_overlap_is_plausible_not_confirmed(self):
        from biolens.eval.genomic_verification import verify_genomic_feature

        genes = [_gene("ENSG1.1", "chr1", 0, 10000)]
        windows = [
            ("chr1:1000-2000", "chr1", 1000, 2000),  # inside claimed gene
            ("chr9:1000-2000", "chr9", 1000, 2000),  # elsewhere
            ("chr9:5000-6000", "chr9", 5000, 6000),  # elsewhere
        ]

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=["ENSG1.1"], ranked_windows=windows, genes=genes,
        )
        assert result.status == "plausible"
        assert abs(result.match_rate - 1 / 3) < 1e-9

    def test_exactly_at_confirmed_threshold_is_confirmed(self):
        from biolens.eval.genomic_verification import verify_genomic_feature

        genes = [_gene("ENSG1.1", "chr1", 0, 10000)]
        windows = [
            ("chr1:1000-2000", "chr1", 1000, 2000),
            ("chr9:1000-2000", "chr9", 1000, 2000),
        ]

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=["ENSG1.1"], ranked_windows=windows, genes=genes,
            confirmed_match_rate=0.5,
        )
        assert result.match_rate == 0.5
        assert result.status == "confirmed"

    def test_version_suffix_mismatch_still_matches(self):
        """Claimed gene_id and GENCODE's own gene_id can carry different
        version suffixes for the same stable gene (the same real mismatch
        load_gencode_genes_by_external_id guards against) — comparison must
        be on the stable ID."""
        from biolens.eval.genomic_verification import verify_genomic_feature

        genes = [_gene("ENSG00000000457.16", "chr1", 0, 10000)]
        windows = [("chr1:1000-2000", "chr1", 1000, 2000)]

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=["ENSG00000000457.8"],
            ranked_windows=windows, genes=genes,
        )
        assert result.status == "confirmed"

    def test_feature_claimed_by_many_genes_is_spurious_regardless_of_overlap(self):
        """A feature that's the single-feature-AUROC-best discriminator for
        many different genes can't specifically be about any one of them —
        classified spurious even if its top windows happen to sit inside
        one of the claimed genes."""
        from biolens.eval.genomic_verification import verify_genomic_feature

        claimed = [f"ENSG{i}.1" for i in range(10)]
        genes = [_gene("ENSG0.1", "chr1", 0, 10000)]
        windows = [("chr1:1000-2000", "chr1", 1000, 2000)]  # inside ENSG0.1, a claimed gene

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=claimed, ranked_windows=windows, genes=genes,
            max_claims_before_generic=5,
        )
        assert result.status == "spurious"
        assert "promiscuous" in result.reason
        assert result.n_claims == 10

    def test_claims_at_generic_threshold_not_flagged(self):
        from biolens.eval.genomic_verification import verify_genomic_feature

        claimed = [f"ENSG{i}.1" for i in range(5)]
        genes = [_gene("ENSG0.1", "chr1", 0, 10000)]
        windows = [("chr1:1000-2000", "chr1", 1000, 2000)]

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=claimed, ranked_windows=windows, genes=genes,
            max_claims_before_generic=5,
        )
        assert result.status == "confirmed"

    def test_no_claimed_genes_is_unverifiable(self):
        from biolens.eval.genomic_verification import verify_genomic_feature

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=[], ranked_windows=[], genes=[],
        )
        assert result.status == "unverifiable"

    def test_empty_ranked_windows_is_spurious_not_divide_by_zero(self):
        from biolens.eval.genomic_verification import verify_genomic_feature

        genes = [_gene("ENSG1.1", "chr1", 0, 10000)]
        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=["ENSG1.1"], ranked_windows=[], genes=genes,
        )
        assert result.status == "spurious"
        assert result.match_rate == 0.0

    def test_evidence_records_overlapping_genes_per_window(self):
        from biolens.eval.genomic_verification import verify_genomic_feature

        genes = [_gene("ENSG1.1", "chr1", 0, 10000)]
        windows = [
            ("chr1:1000-2000", "chr1", 1000, 2000),
            ("chr9:1000-2000", "chr9", 1000, 2000),
        ]

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=["ENSG1.1"], ranked_windows=windows, genes=genes,
        )
        evidence_by_window = dict(result.evidence)
        assert evidence_by_window["chr1:1000-2000"] == ["ENSG1.1"]
        assert evidence_by_window["chr9:1000-2000"] == []

    def test_multiple_claimed_genes_each_count_as_a_match(self):
        from biolens.eval.genomic_verification import verify_genomic_feature

        genes = [
            _gene("ENSG1.1", "chr1", 0, 2000),
            _gene("ENSG2.1", "chr2", 0, 2000),
        ]
        windows = [
            ("chr1:500-1000", "chr1", 500, 1000),
            ("chr2:500-1000", "chr2", 500, 1000),
        ]

        result = verify_genomic_feature(
            feature_idx=42, claimed_gene_ids=["ENSG1.1", "ENSG2.1"],
            ranked_windows=windows, genes=genes,
        )
        assert result.status == "confirmed"
        assert result.match_rate == 1.0


class TestLoadGenomicVerifiedFeatures:
    def test_round_trips_a_written_registry(self, tmp_path):
        import yaml

        from biolens.eval.genomic_verification import load_genomic_verified_features

        path = tmp_path / "registry.yaml"
        path.write_text(yaml.dump({
            "features": {
                42: {
                    "status": "confirmed",
                    "claimed_gene_ids": ["ENSG1.1"],
                    "n_claims": 1,
                    "match_rate": 0.75,
                    "reason": "15/20 top-activating windows fall within a claimed gene",
                    "matched_windows": ["chr1:1000-2000"],
                },
                99: {
                    "status": "spurious",
                    "claimed_gene_ids": ["ENSG2.1"],
                    "n_claims": 1,
                    "match_rate": 0.0,
                    "reason": "none of the top-activating windows genome-wide fall within any claimed gene",
                    "matched_windows": [],
                },
            }
        }))

        result = load_genomic_verified_features(path)
        assert set(result.keys()) == {42, 99}
        assert result[42].status == "confirmed"
        assert result[42].claimed_gene_ids == ["ENSG1.1"]
        assert result[42].match_rate == 0.75
        assert result[99].status == "spurious"

    def test_missing_optional_fields_default_gracefully(self, tmp_path):
        import yaml

        from biolens.eval.genomic_verification import load_genomic_verified_features

        path = tmp_path / "registry.yaml"
        path.write_text(yaml.dump({"features": {7: {"status": "confirmed"}}}))

        result = load_genomic_verified_features(path)
        assert result[7].status == "confirmed"
        assert result[7].claimed_gene_ids == []
        assert result[7].n_claims == 0
        assert result[7].match_rate == 0.0
        assert result[7].matched_windows == []

    def test_empty_features_gives_empty_result(self, tmp_path):
        import yaml

        from biolens.eval.genomic_verification import load_genomic_verified_features

        path = tmp_path / "registry.yaml"
        path.write_text(yaml.dump({"features": {}}))

        assert load_genomic_verified_features(path) == {}
