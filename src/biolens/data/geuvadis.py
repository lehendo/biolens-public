"""
Geuvadis case study data loading: genotype dosage from 1000 Genomes VCF,
RNA-seq expression, and candidate-locus restriction to previously-published
Geuvadis eQTLs.

Fully open data — no dbGaP gate, unlike GTEx. Sources, format-verified
directly against the real public files (2026-07-03), not assumed:
  - Genotypes: 1000 Genomes Project phase 3 VCF, e.g.
    `http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
    ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz`
  - Expression: `GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz` at
    `https://ftp.ebi.ac.uk/pub/databases/microarray/data/experiment/GEUV/
    E-GEUV-1/analysis_results/` — despite the "RPKM" filename, per Geuvadis'
    own README this specific file has PEER normalization applied on top of
    library-depth scaling ("Library depth & expressed & PEER"), which is why
    real values are NOT non-negative RPKM (confirmed directly: real values
    include negatives, e.g. -0.00090...) — do not describe this as raw RPKM.
  - eQTLs: `EUR373.gene.cis.FDR5.best.rs137.txt.gz` from the same directory
    (Lappalainen et al. 2013's own published cis-eQTL results) — format
    confirmed against `GeuvadisRNASeqAnalysisFiles_README.txt` in the same
    directory, not guessed at.

CHROMOSOME NAMING MISMATCH, confirmed real and handled here: the 1000
Genomes VCF and the Geuvadis eQTL file both use bare chromosome names
("22"), while UCSC reference genome FASTAs (e.g. hg38.fa, used by
biolens.data.reference_genome) use "chr"-prefixed names ("chr22"). Every
chrom value entering or leaving this module is normalized to the
"chr"-prefixed convention, matching reference_genome.py and
genomic_annotations.py, with the bare-name form used only internally where
required to match the VCF's own raw CHROM field.

Uses cyvcf2 (htslib-backed, the standard fast Python VCF parser — not a
hand-rolled VCF-format parser, which has many correctness-critical edge
cases: multi-allelic sites, phased/unphased genotypes, missing calls).

IMPORTANT genotype-encoding note: cyvcf2 requires `gts012=True` at VCF-open
time to get HOM_REF=0/HET=1/HOM_ALT=2/UNKNOWN=3 dosage-aligned encoding —
the default encoding is HOM_REF=0/HET=1/UNKNOWN=2/HOM_ALT=3, a DIFFERENT,
non-monotonic order that would silently produce wrong dosages if used
directly as a numeric treatment variable. This module always opens with
gts012=True; see test_geuvadis.py for a regression test pinning this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MISSING_GT = 3  # gts012=True encoding — see module docstring

# Real column order in Geuvadis' published cis-eQTL "best" files (e.g.
# EUR373.gene.cis.FDR5.best.rs137.txt.gz), confirmed against
# GeuvadisRNASeqAnalysisFiles_README.txt — no header row in the real file.
_EQTL_COLUMNS = [
    "snp_id", "id_unused", "gene_id", "probe_id", "chr_snp", "chr_gene",
    "snp_pos", "tss_pos", "distance", "rvalue", "pvalue", "log10pvalue",
]


def _normalize_chrom(chrom: str) -> str:
    """Ensure 'chr'-prefixed convention (matching reference_genome.py /
    genomic_annotations.py), regardless of whether the input already has it —
    the 1000 Genomes VCF and Geuvadis eQTL files both use bare names ('22'),
    UCSC reference FASTAs use prefixed names ('chr22')."""
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def _strip_chrom_prefix(chrom: str) -> str:
    """Inverse of _normalize_chrom — needed when matching against the VCF's
    own raw (unprefixed) CHROM field."""
    return chrom[3:] if chrom.startswith("chr") else chrom


@dataclass
class VariantGenotypes:
    variant_id: str
    chrom: str  # always "chr"-prefixed — see module docstring
    pos: int
    ref: str
    alt: str
    dosage_by_sample: dict[str, int]  # sample_id -> 0/1/2 (missing calls excluded)


def load_variant_genotypes(
    vcf_path: str | Path,
    variant_positions: list[tuple[str, int]],
) -> dict[str, VariantGenotypes]:
    """
    Extract genotype dosage for a specific list of variants (chrom, pos)
    from a VCF — restricted to a candidate list, not a whole-genome scan
    (matches the plan's multiple-testing-tractability design decision).

    Args:
        vcf_path: Path to a (optionally bgzipped) VCF file. For real 1000
            Genomes data this should be bgzip-compressed + tabix-indexed for
            practical region-query performance; this function iterates the
            whole file otherwise, which is correct but slow for a
            whole-chromosome VCF — fine for a candidate-locus-restricted
            case study, not intended for whole-genome scanning.
        variant_positions: (chrom, pos) pairs to extract — 1-based VCF
            position convention. `chrom` may be given either "chr"-prefixed
            or bare ("chr22" or "22") — normalized internally either way.

    Returns:
        variant_id ("chrom:pos:ref:alt", chrom "chr"-prefixed) ->
        VariantGenotypes. Variants in variant_positions not found in the VCF
        are simply absent from the result (not an error — a candidate eQTL
        list may reference variants not present in a given VCF subset/
        chromosome file).
    """
    import cyvcf2

    # Match against the VCF's own raw (bare, unprefixed) CHROM convention.
    target_positions = {(_strip_chrom_prefix(c), p) for c, p in variant_positions}
    vcf = cyvcf2.VCF(str(vcf_path), gts012=True)
    samples = vcf.samples

    results: dict[str, VariantGenotypes] = {}
    try:
        for variant in vcf:
            if (variant.CHROM, variant.POS) not in target_positions:
                continue
            if len(variant.ALT) != 1:
                logger.debug(
                    "Skipping multi-allelic site %s:%d (%d alt alleles) — "
                    "only biallelic SNVs supported",
                    variant.CHROM, variant.POS, len(variant.ALT),
                )
                continue

            gt_types = variant.gt_types  # gts012=True: 0/1/2/3=missing
            dosage_by_sample = {
                samples[i]: int(gt_types[i])
                for i in range(len(samples))
                if gt_types[i] != _MISSING_GT
            }

            chrom = _normalize_chrom(variant.CHROM)
            variant_id = f"{chrom}:{variant.POS}:{variant.REF}:{variant.ALT[0]}"
            results[variant_id] = VariantGenotypes(
                variant_id=variant_id, chrom=chrom, pos=variant.POS,
                ref=variant.REF, alt=variant.ALT[0], dosage_by_sample=dosage_by_sample,
            )
    finally:
        vcf.close()

    logger.info(
        "Loaded genotypes for %d/%d requested variants", len(results), len(variant_positions)
    )
    return results


@dataclass
class PhasedGenotype:
    """One biallelic SNP's real phased genotype across samples — the
    multi-variant, phase-aware analog of VariantGenotypes, needed for
    building each individual's actual personal haplotype sequence rather
    than a single locus-level dosage scalar (see load_window_genotypes)."""

    pos: int  # 1-based VCF position
    ref: str
    alt: str
    # sample_id -> (allele1, allele2), each 0=ref/1=alt (haplotype-ordered
    # per the VCF's own phasing) — samples with a missing call at this
    # variant are absent, matching VariantGenotypes' convention.
    alleles_by_sample: dict[str, tuple[int, int]]


def load_window_genotypes(
    vcf_path: str | Path,
    windows: list[tuple[str, str, int, int]],
) -> dict[str, list[PhasedGenotype]]:
    """
    Extract PHASED per-sample genotypes for every variant falling within
    any of the given windows, in a SINGLE linear scan of the VCF — not one
    scan per window. (This is the same "load once, not per-locus" fix
    already applied once this session to load_variant_genotypes's caller in
    run_geuvadis_case_study.py, generalized here from an exact-position
    query to a window-membership query.)

    This exists to fix a real, referee-confirmed identification failure in
    the original causal-mediation design: a mediator built from a single
    (chrom, pos) genotype's dosage times one fixed activation-shift scalar
    is `M = Delta * D` — an exact rescaling of the treatment — which makes
    the outcome regression's design matrix rank-deficient (mediation effect
    and direct effect are not separably identified; confirmed empirically:
    two datasets with opposite true mediation fractions, 100% vs 0%,
    produce statistically indistinguishable results under that
    construction). The fix requires each individual's mediator to carry
    real information beyond their dosage at the lead variant — i.e. their
    actual local genetic background, all nearby variants included, not
    just the one lead eQTL SNP.

    Args:
        vcf_path: Path to the (bgzipped) VCF.
        windows: (locus_key, chrom, start, end) tuples — 0-indexed,
            half-open, chrom either "chr"-prefixed or bare. locus_key is a
            caller-chosen grouping label (e.g. the lead variant's
            variant_id) — windows may overlap without conflict, since
            results are grouped by locus_key, not deduplicated by position.

    Returns:
        locus_key -> list of PhasedGenotype for every biallelic SNV (single-
        nucleotide ref AND alt — indels are skipped, same "out of scope,
        shifts the coordinate frame" scoping decision already enforced for
        the lead variant elsewhere in this codebase; a real, not
        hypothetical, fix — an indel originally passed the earlier
        single-ALT-allele check here, since biallelic only means "one ALT
        allele," not "one base," and crashed build_haplotype_sequence on
        real chr22 data, job 9755424, 2026-08-03) found within that locus's
        window, sorted by position. Every requested locus_key is present in
        the result (possibly with an empty list, in the pathological case
        where even the lead SNP itself doesn't appear — e.g. it was
        filtered as multi-allelic or an indel).

    Correctness note on the backward-scan bound: mirrors genomic_annotations.
    find_overlapping_genes's bisect + backward-scan approach, but the bound
    here is derived from the actual requested windows' own maximum width
    (not a fixed genomic-feature-class constant like cCRE/gene length,
    since window width is a caller-chosen parameter, not a biological
    property) — guaranteed not to miss a true overlap for any window this
    function was actually asked to check.
    """
    import bisect

    import cyvcf2

    by_chrom: dict[str, list[tuple[int, int, str]]] = {}
    for locus_key, chrom, start, end in windows:
        by_chrom.setdefault(_strip_chrom_prefix(chrom), []).append((start, end, locus_key))
    index: dict[str, tuple[list[int], list[tuple[int, int, str]]]] = {}
    for chrom, items in by_chrom.items():
        items.sort(key=lambda t: t[0])
        index[chrom] = ([it[0] for it in items], items)

    max_width = max((end - start for _, _, start, end in windows), default=0)
    results: dict[str, list[PhasedGenotype]] = {locus_key: [] for locus_key, *_ in windows}

    vcf = cyvcf2.VCF(str(vcf_path))
    samples = vcf.samples
    try:
        for variant in vcf:
            starts, items = index.get(variant.CHROM, ([], []))
            if not items:
                continue
            if len(variant.ALT) != 1:
                continue
            # SNV-only, matching the same scoping decision already enforced
            # for the lead variant elsewhere in this codebase (biolens.data.
            # reference_genome._validate_snv_alleles: "indels shift the
            # coordinate frame and need separate handling, out of scope").
            # Real, not hypothetical: found via a real production crash on
            # real chr22 data, job 9755424, 2026-08-03 -- an indel (REF=
            # "CT") passed the single-ALT-allele check above (it's
            # biallelic, just not a SNV) and reached
            # build_haplotype_sequence, which compares a variant's ref
            # allele against exactly ONE base of the window sequence;
            # against a multi-base ref this always "mismatches" regardless
            # of whether the reference genome is actually correct there,
            # and even a correct match wouldn't be substitutable without
            # shifting every downstream position in the window (the same
            # reason indels are out of scope for the lead-variant
            # substitution too).
            if len(variant.REF) != 1 or len(variant.ALT[0]) != 1:
                continue
            variant_idx = variant.POS - 1  # VCF 1-based -> 0-based

            # Rightmost window with start <= variant_idx (candidates start
            # here, scanning backward) — a window overlaps iff additionally
            # end > variant_idx, so a window whose start is far enough back
            # that even the widest requested window couldn't reach
            # variant_idx can safely stop the scan.
            upper = bisect.bisect_right(starts, variant_idx) - 1
            cutoff = variant_idx - max_width
            j = upper
            genotypes = None  # lazily materialized only if >=1 window matches
            while j >= 0 and items[j][0] >= cutoff:
                start, end, locus_key = items[j]
                if start <= variant_idx < end:
                    if genotypes is None:
                        genotypes = variant.genotypes
                    alleles_by_sample = {
                        samples[i]: (genotypes[i][0], genotypes[i][1])
                        for i in range(len(samples))
                        if genotypes[i][0] >= 0 and genotypes[i][1] >= 0  # exclude missing (-1)
                    }
                    results[locus_key].append(PhasedGenotype(
                        pos=variant.POS, ref=variant.REF, alt=variant.ALT[0],
                        alleles_by_sample=alleles_by_sample,
                    ))
                j -= 1
    finally:
        vcf.close()

    for locus_key in results:
        results[locus_key].sort(key=lambda g: g.pos)

    return results


def build_haplotype_sequence(
    ref_sequence: str,
    window_start: int,
    variants: list[PhasedGenotype],
    sample_id: str,
    haplotype_index: int,
) -> str:
    """
    Substitute every variant in `variants` into `ref_sequence` according to
    one individual's one haplotype's allele at each position — the
    multi-variant, per-individual analog of
    biolens.data.reference_genome.extract_ref_alt_window's single-
    substitution ref/alt pair. This is what gives the causal-mediation
    design a real, non-degenerate per-individual mediator: two individuals
    with the SAME dosage at the lead eQTL variant can still have DIFFERENT
    personal sequences (and therefore different SAE feature activations)
    if they carry different OTHER nearby variants — see
    load_window_genotypes's docstring for why this is required, not
    optional, for a valid mediation estimand.

    Args:
        ref_sequence: Reference window sequence (e.g. from
            biolens.data.reference_genome.extract_window_sequences).
        window_start: 0-indexed genomic coordinate of ref_sequence[0].
        variants: PhasedGenotype records overlapping this window (from
            load_window_genotypes) — variants outside
            [window_start, window_start + len(ref_sequence)) are silently
            skipped (defensive: a caller using a narrower context than the
            original window-collection pass may pass some out-of-range
            variants; not an error, just nothing to substitute there).
        sample_id: Which individual's genotype to use.
        haplotype_index: 0 or 1 — which of the two phased haplotypes.

    Returns:
        The personal haplotype sequence, same length as ref_sequence,
        uppercase, differing from the reference at every position where
        this sample carries the alt allele on this haplotype. A missing
        genotype call for this sample at a given variant leaves that
        position as the reference base — the standard imputation-free
        convention (matches how a no-call is already handled for the lead
        variant elsewhere in this pipeline: absent from
        VariantGenotypes.dosage_by_sample, not treated as an error). A
        variant whose recorded ref allele doesn't match the window
        sequence's actual base at that position is SKIPPED (left as the
        reference base, logged at warning level, NOT raised) — see the
        note below for why this differs from the lead-variant case.

    Note on ref-genome mismatches (real production incident, job 9901091,
    2026-08-12): an earlier version of this function raised ValueError on a
    mismatch, mirroring reference_genome.extract_ref_alt_window's behavior
    for the single lead-variant case. That's the right call for the LEAD
    variant — a mismatch there means the locus's own coordinate is wrong,
    so the whole locus is unusable. It is the WRONG call here: this
    function substitutes many (often 100-300+) NEARBY variants into one
    sequence, each individually lifted from GRCh37 to GRCh38 (see
    load_window_genotypes's docstring) — a liftOver imprecision for ONE
    specific nearby variant among hundreds is a real, expected, isolated
    data-quality edge case, not evidence the whole window or locus is
    compromised. Raising here crashed an entire multi-hour production run
    on the first such variant encountered, discarding all loci processed
    before it. Skipping just the one mismatched variant (matching how a
    missing genotype call or an out-of-range variant are already handled
    above) keeps the other, unaffected variants' real information intact.
    """
    seq = list(ref_sequence.upper())
    for variant in variants:
        offset = (variant.pos - 1) - window_start
        if not (0 <= offset < len(seq)):
            continue
        actual_ref_base = seq[offset]
        if actual_ref_base != variant.ref.upper():
            logger.warning(
                "Skipping variant at position %d (window offset %d) for sample %s: "
                "reference genome mismatch — window sequence has %r, VCF says ref=%r "
                "(likely an isolated liftOver imprecision for this one nearby variant, "
                "not a problem with the locus as a whole).",
                variant.pos, offset, sample_id, actual_ref_base, variant.ref,
            )
            continue
        alleles = variant.alleles_by_sample.get(sample_id)
        if alleles is None:
            continue
        if alleles[haplotype_index] == 1:
            seq[offset] = variant.alt.upper()
    return "".join(seq)


@dataclass
class PopulationLabel:
    population: str  # e.g. "GBR" (1000 Genomes population code)
    superpopulation: str  # e.g. "EUR" (1000 Genomes super-population code)


def load_population_labels(path: str | Path) -> dict[str, PopulationLabel]:
    """
    Load 1000 Genomes' own published sample-to-population panel (e.g.
    `integrated_call_samples_v3.20130502.ALL.panel` from
    `http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/`) — real
    column format confirmed against the published file: tab-separated, with
    a header row `sample  pop  super_pop  gender`.

    This supplies the ancestry-covariate data needed for the
    sequential-ignorability assumption underlying the causal-mediation
    analysis: population stratification is the classic unmeasured
    confounder in genotype-expression analysis, and without ancestry
    adjustment it would be unaddressed. Used as categorical covariates via
    build_ancestry_covariates, not genome-wide genotype PCs: this pipeline's
    VCF is restricted to the candidate loci' chromosome(s) (matching the
    eQTL candidate list's scope), which is not enough genome-wide markers
    for a proper LD-pruned PCA — 1000 Genomes' own self-reported population/
    superpopulation labels are the real, directly available, methodologically
    standard alternative at this scope.

    Returns:
        sample_id -> PopulationLabel, for every sample in the panel file
        (the full 1000 Genomes release, a superset of any one project's
        genotyped/expression-matched sample set).
    """
    df = pd.read_csv(path, sep="\t")
    return {
        str(row.sample): PopulationLabel(
            population=str(row.pop), superpopulation=str(row.super_pop),
        )
        for row in df.itertuples()
    }


def build_ancestry_covariates(
    samples: list[str], population_labels: dict[str, PopulationLabel],
) -> pd.DataFrame:
    """
    One-hot-encode each sample's superpopulation label (dropping one level
    to avoid collinearity with the regression intercept) into a covariates
    DataFrame directly usable via
    `run_ikt_mediation(..., covariates=...)` /
    `sensitivity_analysis(..., covariates=...)`.

    Args:
        samples: Sample IDs in the SAME order as the matched
            treatment/mediator/outcome arrays (i.e. the `samples` list
            build_mediation_arrays returns) — row i of the result
            corresponds to samples[i].
        population_labels: From load_population_labels.

    Returns:
        (n, k-1) float DataFrame of 0/1 superpopulation dummy columns,
        k = number of distinct superpopulations actually present among
        `samples` (not the full panel's superpopulation count).

    Raises:
        KeyError: if any sample in `samples` has no population label —
            fails loudly rather than silently dropping a sample from the
            ancestry adjustment while it stays in treatment/mediator/
            outcome, which would desynchronize row alignment between the
            covariates DataFrame and those arrays.
    """
    missing = [s for s in samples if s not in population_labels]
    if missing:
        raise KeyError(
            f"{len(missing)} sample(s) have no population label (e.g. "
            f"{missing[:5]}) — check the population-labels panel file covers "
            f"every sample in the genotype/expression intersection."
        )
    superpop = pd.Series(
        [population_labels[s].superpopulation for s in samples], name="superpop"
    )
    return pd.get_dummies(superpop, prefix="superpop", drop_first=True).astype(float)


def summarize_population_composition(
    samples: list[str], population_labels: dict[str, PopulationLabel],
) -> dict[str, int]:
    """
    Real superpopulation composition counts (superpopulation -> sample
    count) for a matched sample set — exists specifically to document the
    analyzed cohort's real population composition rather than leave it an
    unchecked assumption (a cohort description that merely asserts
    "roughly-homogeneous EUR-labeled" without having actually counted it
    would be an unverified claim).
    """
    counts: dict[str, int] = {}
    for s in samples:
        superpop = population_labels[s].superpopulation
        counts[superpop] = counts.get(superpop, 0) + 1
    return counts


def load_expression_matrix(path: str | Path, gene_id_col: str = "TargetID") -> pd.DataFrame:
    """
    Load a Geuvadis normalized expression matrix: quantification units
    (rows) x samples (columns), tab-separated, with a header row.

    Default `gene_id_col="TargetID"` matches Geuvadis' real published
    quantification files (e.g. GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz)
    directly, per `GeuvadisRNASeqAnalysisFiles_README.txt`'s own column
    documentation — NOT "gene_id" (an earlier, unverified assumption in this
    module, corrected 2026-07-03 against the real file).

    Returns a DataFrame indexed by the quantification-unit ID, columns =
    sample IDs. Despite some Geuvadis filenames containing "RPKM", the
    specific file this project uses has PEER normalization applied on top of
    library-depth scaling (confirmed real values include negatives) — this
    function does not transform units; know what you actually downloaded
    before treating these as raw non-negative RPKM values.
    """
    df = pd.read_csv(path, sep="\t")
    return df.set_index(gene_id_col)


@dataclass
class EqtlCandidate:
    variant_id: str
    chrom: str  # always "chr"-prefixed — see module docstring
    pos: int
    gene_id: str


def load_published_eqtl_candidates(path: str | Path) -> list[EqtlCandidate]:
    """
    Load a pre-specified candidate-locus list from Geuvadis' own published
    cis-eQTL results: restricting candidate loci to previously-published
    Geuvadis eQTLs keeps multiple-testing burden tractable and
    pre-specified.

    Parses the REAL format of Geuvadis' published "best cis-eQTL per gene"
    files (e.g. EUR373.gene.cis.FDR5.best.rs137.txt.gz): tab-separated, NO
    header row, 12 columns, confirmed directly against
    `GeuvadisRNASeqAnalysisFiles_README.txt` — not the earlier, incorrect
    assumption of a 3-column `chrom`/`pos`/`gene_id` file with a header
    (corrected 2026-07-03 against the real public file).
    """
    df = pd.read_csv(path, sep="\t", header=None, names=_EQTL_COLUMNS)

    return [
        EqtlCandidate(
            variant_id=str(row.snp_id),
            chrom=_normalize_chrom(row.chr_snp),
            pos=int(row.snp_pos),
            gene_id=str(row.gene_id),
        )
        for row in df.itertuples()
    ]


def build_mediation_arrays(
    genotypes: VariantGenotypes,
    expression: pd.DataFrame,
    gene_id: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Align a variant's per-sample genotype dosage with a gene's per-sample
    expression, restricted to samples present in BOTH (a real, common data-
    quality step — genotyped and RNA-seq'd sample sets don't always match
    exactly).

    Args:
        genotypes:  From load_variant_genotypes (one variant).
        expression: From load_expression_matrix (genes x samples).
        gene_id:    Which gene's expression to align against.

    Returns:
        (treatment_dosage, outcome_expression, sample_ids) — all length n,
        n = |samples with both genotype and expression data|, in matched order.

    Raises:
        KeyError: if gene_id isn't in the expression matrix.
        ValueError: if no samples have both genotype and expression data
            (nothing to analyze — surfaced explicitly, not silently
            returned as empty arrays that would fail confusingly downstream).
    """
    if gene_id not in expression.index:
        raise KeyError(f"Gene {gene_id!r} not found in expression matrix")

    gene_expression = expression.loc[gene_id]
    common_samples = sorted(set(genotypes.dosage_by_sample.keys()) & set(gene_expression.index))
    if not common_samples:
        raise ValueError(
            f"No samples with both genotype ({genotypes.variant_id}) and expression "
            f"({gene_id}) data — check sample ID conventions match between the VCF "
            f"and expression matrix"
        )

    treatment = np.array([genotypes.dosage_by_sample[s] for s in common_samples], dtype=float)
    outcome = np.array([gene_expression[s] for s in common_samples], dtype=float)
    return treatment, outcome, common_samples
