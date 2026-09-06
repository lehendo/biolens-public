"""
ENCODE candidate cis-regulatory elements (cCREs) — the genomic-annotation
analog to GO-term probing (biolens.data.uniprot), for Evo 2 interpretability
probing.

Data source verified directly (not guessed) via UCSC's genome browser
download index: the ENCODE3 Registry of cCREs, combined track, hg38
assembly — a stable, versioned bigBed file:
  http://hgdownload.soe.ucsc.edu/gbdb/hg38/encode3/ccre/encodeCcreCombined.bb
(48MB, BED 9+6 schema; confirmed via bb.SQL() against the real downloaded
file, not from documentation alone.) Each cCRE carries one or more
classification labels (e.g. "PLS" promoter-like, "pELS"/"dELS" proximal/
distal enhancer-like, "CTCF-bound") in its `ccre` field, comma-separated —
directly analogous to a protein having multiple GO term annotations.

Parsed via pyBigWig (a real Python binding for UCSC's bigBed/bigWig binary
formats — chosen over shelling out to the `bigBedToBed` UCSC command-line
tool, which would add an external non-pip binary dependency this project's
other loaders don't need).
"""

from __future__ import annotations

import bisect
import gzip
import logging
from dataclasses import dataclass
from pathlib import Path

from biolens.data._http_cache import _get_or_download

logger = logging.getLogger(__name__)

# Verified via bb.chroms() / bb.SQL() against the real downloaded file —
# see module docstring.
_CCRE_BIGBED_URL = "http://hgdownload.soe.ucsc.edu/gbdb/hg38/encode3/ccre/encodeCcreCombined.bb"

# GENCODE Human Release 50, "basic" gene set (reference chromosomes only) —
# the main annotation file for most users per gencodegenes.org/human/;
# confirmed directly via `curl -sI` against the real file (79,970,394 bytes,
# gzip) before downloading, not assumed from documentation alone. Used for
# gene-body coordinates (Geuvadis naive-feature probing), not
# transcript-level detail — every GENCODE GTF carries
# one top-level "gene" feature-type row per gene regardless of how many
# transcripts it has, which is all this module reads.
_GENCODE_BASIC_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/"
    "gencode.v50.basic.annotation.gtf.gz"
)

# ENCODE cCREs are short regulatory elements (empirically ~150-350bp in the
# real data) — this bounds the backward interval-overlap scan in
# label_dna_windows to a value far larger than any real cCRE, guaranteeing no
# missed overlaps (see that function's docstring for the correctness argument).
_MAX_PLAUSIBLE_CCRE_LENGTH_BP = 5000

# Same backward-scan-bound trick as _MAX_PLAUSIBLE_CCRE_LENGTH_BP, but for
# genes rather than cCREs — genes can span far more than a cCRE. DMD
# (dystrophin), the longest known human protein-coding gene, is ~2.2Mb; this
# bound is set safely above that so find_overlapping_genes never misses a
# real overlap.
_MAX_PLAUSIBLE_GENE_LENGTH_BP = 3_000_000


@dataclass(frozen=True)
class CCRE:
    """One ENCODE candidate cis-regulatory element."""

    chrom: str
    start: int  # 0-indexed, half-open (BED convention)
    end: int
    accession: str
    ccre_classes: frozenset[str]  # e.g. {"pELS", "CTCF-bound"}

    @property
    def length(self) -> int:
        return self.end - self.start


def load_encode_ccres(
    chroms: set[str] | None = None,
    data_dir: Path | None = None,
) -> list[CCRE]:
    """
    Download (if needed) and parse the ENCODE cCRE registry for hg38.

    Args:
        chroms: If given, restrict to these chromosomes (e.g. {"chr1", "chr2"})
                — faster and lower-memory for a probing run scoped to a
                subset of the genome. None = all 24 chromosomes (~2.35M cCREs).
        data_dir: Override default annotation cache directory
                  (BIOLENS_DATA_DIR env var, same convention as biolens.data.uniprot).

    Returns:
        List of CCRE records.
    """
    import pyBigWig

    bb_path = _get_or_download(_CCRE_BIGBED_URL, "encodeCcreCombined.bb", data_dir)
    bb = pyBigWig.open(str(bb_path))
    try:
        available_chroms = set(bb.chroms().keys())
        target_chroms = (chroms & available_chroms) if chroms else available_chroms

        ccres: list[CCRE] = []
        for chrom in sorted(target_chroms):
            chrom_len = bb.chroms()[chrom]
            entries = bb.entries(chrom, 0, chrom_len) or []
            for start, end, rest in entries:
                fields = rest.split("\t")
                # BED 9+6 schema, confirmed via bb.SQL() (see module docstring):
                # name, score, strand, thickStart, thickEnd, reserved, ccre,
                # encodeLabel, zScore, ucscLabel, accessionLabel, description
                accession = fields[0]
                ccre_field = fields[6] if len(fields) > 6 else ""
                classes = frozenset(c for c in ccre_field.split(",") if c)
                ccres.append(
                    CCRE(chrom=chrom, start=start, end=end, accession=accession, ccre_classes=classes)
                )
    finally:
        bb.close()

    logger.info(
        "Loaded %d ENCODE cCREs across %d chromosome(s)", len(ccres), len(target_chroms)
    )
    return ccres


def label_dna_windows(
    windows: list[tuple[str, str, int, int]],
    ccres: list[CCRE],
) -> dict[str, set[str]]:
    """
    Assign cCRE classification labels to DNA windows by genomic overlap — the
    genomic-annotation analog of GO-term protein labels (biolens.eval.probing
    treats the returned dict exactly like a `go_labels` dict; the probing
    engine itself is domain-agnostic).

    Args:
        windows: (window_id, chrom, start, end) tuples — 0-indexed, half-open,
                 matching BED convention. window_id should be unique (e.g. the
                 sequence ID used elsewhere in the Evo 2 activation cache).
        ccres:   From load_encode_ccres().

    Returns:
        window_id -> set of cCRE class labels for every cCRE overlapping that
        window (union across all overlapping cCREs, since a window can span
        several regulatory elements). Windows with no overlapping cCRE map to
        an empty set (present in the dict, not silently omitted).

    Correctness note on the backward-scan bound: cCREs are indexed per
    chromosome sorted by start position. For a window [start, end), any cCRE
    that could overlap must have `cCRE.start < end`. Because real cCREs are
    short (empirically ~150-350bp; bounded here at
    _MAX_PLAUSIBLE_CCRE_LENGTH_BP), a cCRE starting more than that bound
    before `start` cannot possibly reach into the window — so scanning
    backward from the bisect point until `cCRE.start < start -
    _MAX_PLAUSIBLE_CCRE_LENGTH_BP` is guaranteed not to miss any true overlap
    with real ENCODE cCRE data, while staying far cheaper than an
    interval tree for this specific, short-feature use case.
    """
    by_chrom: dict[str, list[CCRE]] = {}
    for c in ccres:
        by_chrom.setdefault(c.chrom, []).append(c)
    index: dict[str, tuple[list[int], list[CCRE]]] = {}
    for chrom, items in by_chrom.items():
        items.sort(key=lambda c: c.start)
        index[chrom] = ([c.start for c in items], items)

    labels: dict[str, set[str]] = {}
    for window_id, chrom, start, end in windows:
        starts, items = index.get(chrom, ([], []))
        window_labels: set[str] = set()

        if items:
            # Rightmost cCRE with start < end (candidates start here, scanning backward).
            upper = bisect.bisect_left(starts, end) - 1
            cutoff = start - _MAX_PLAUSIBLE_CCRE_LENGTH_BP
            j = upper
            while j >= 0 and items[j].start >= cutoff:
                ccre = items[j]
                if ccre.start < end and ccre.end > start:  # true interval overlap
                    window_labels |= ccre.ccre_classes
                j -= 1

        labels[window_id] = window_labels

    return labels


@dataclass(frozen=True)
class GencodeGene:
    """One GENCODE gene-body record (gene-level, not transcript-level)."""

    gene_id: str  # versioned Ensembl ID, e.g. "ENSG00000131044.11", AS
                  # WRITTEN IN THIS GENCODE RELEASE — the version suffix is
                  # NOT guaranteed to match an older data source's gene_id
                  # for the same stable gene (confirmed 2026-07-25: real
                  # Geuvadis eQTL data uses ENSG00000000457.8, current
                  # GENCODE Release 50 uses ENSG00000000457.16 for the same
                  # gene — a real production failure the first time this
                  # was joined against Geuvadis data by exact string match,
                  # not a hypothetical). Callers joining against a different
                  # gene-ID source should match on stable_gene_id(), not
                  # this field directly.
    chrom: str
    start: int  # 0-indexed, half-open (BED convention, converted from GTF's
                # 1-indexed closed interval — see load_gencode_genes)
    end: int
    strand: str


def stable_gene_id(gene_id: str) -> str:
    """Strip the Ensembl/GENCODE version suffix: 'ENSG00000131044.11' ->
    'ENSG00000131044'. The stable ID is what's safe to join across data
    sources built from different GENCODE/Ensembl releases (see GencodeGene's
    gene_id docstring for the real mismatch this guards against) — the
    version suffix only tells you which release's transcript models were
    used, not a different gene identity."""
    return gene_id.split(".")[0]


def _parse_gtf_attribute(attributes_field: str, key: str) -> str | None:
    """Extract e.g. 'gene_id "ENSG00000131044.11";' -> 'ENSG00000131044.11'
    from a GTF attributes field (semicolon-separated key-value pairs)."""
    for part in attributes_field.split(";"):
        part = part.strip()
        if part.startswith(key + ' "'):
            return part[len(key):].strip().strip('"')
    return None


def load_gencode_genes(
    gene_ids: set[str] | None = None,
    data_dir: Path | None = None,
) -> list[GencodeGene]:
    """
    Download (if needed) and parse GENCODE's basic gene annotation GTF,
    extracting gene-body coordinates for gene-locus-level genomic labeling
    (Geuvadis naive-feature probing).

    Args:
        gene_ids: If given, restrict to these gene IDs (e.g. the specific
            eQTL candidate genes from biolens.data.geuvadis.
            load_published_eqtl_candidates) — the GTF has several hundred
            thousand rows across all feature types; only 'gene' rows are
            ever parsed either way, so this is a memory/time optimization
            for a small candidate list, not a correctness requirement.
            Matched via stable_gene_id() (version suffix stripped) on BOTH
            sides, not exact string equality — a real production failure
            the first time this was exact-matched against Geuvadis data
            (ENSG00000000457.8 there vs. ENSG00000000457.16 in this GENCODE
            release, for the same gene; see GencodeGene's gene_id
            docstring), not a hypothetical edge case.
        data_dir: Override default annotation cache directory
            (BIOLENS_DATA_DIR env var, same convention as load_encode_ccres).

    Returns:
        List of GencodeGene records, 0-indexed half-open coordinates
        (matching this module's CCRE convention — the raw GTF is 1-indexed
        and closed-interval; converted here). Each record's gene_id is
        exactly as GENCODE wrote it, including THIS release's version
        suffix — callers joining against an external gene-ID source (e.g.
        Geuvadis) should compare via stable_gene_id(), not this field
        directly, and should remap back to the external source's own
        gene_id string if that's what a downstream join key needs to be.
    """
    gtf_path = _get_or_download(
        _GENCODE_BASIC_GTF_URL, "gencode.v50.basic.annotation.gtf.gz", data_dir
    )

    stable_requested = {stable_gene_id(g) for g in gene_ids} if gene_ids is not None else None

    genes: list[GencodeGene] = []
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom, _source, _feature, start_1based, end_1based, _score, strand, _frame, attrs = fields
            gene_id = _parse_gtf_attribute(attrs, "gene_id")
            if gene_id is None:
                continue
            if stable_requested is not None and stable_gene_id(gene_id) not in stable_requested:
                continue
            genes.append(GencodeGene(
                gene_id=gene_id,
                chrom=chrom,
                start=int(start_1based) - 1,  # GTF 1-based closed -> 0-based half-open
                end=int(end_1based),
                strand=strand,
            ))

    logger.info(
        "Loaded %d GENCODE genes%s",
        len(genes),
        f" (filtered to {len(gene_ids)} requested IDs, matched by stable ID)" if gene_ids is not None else "",
    )
    return genes


def load_gencode_genes_by_external_id(
    external_gene_ids: set[str],
    data_dir: Path | None = None,
) -> dict[str, GencodeGene]:
    """
    Like load_gencode_genes, but keyed by the CALLER's own gene_id strings
    rather than GENCODE's own — handles the real version-suffix mismatch
    between GENCODE releases and external gene-ID sources (e.g. Geuvadis,
    whose eQTL file uses an older GENCODE/Ensembl build's version suffixes
    than this loader downloads; confirmed real, not hypothetical, 2026-07-25:
    ENSG00000000457.8 in Geuvadis vs. ENSG00000000457.16 here, same gene).

    This exact remap (build a stable-ID -> external-ID map, then walk
    load_gencode_genes' results translating each match back) was
    independently duplicated in scripts/run_geuvadis_naive_probing.py and
    scripts/run_geuvadis_case_study.py before being factored out here
    (2026-07-29) — both need "the gene matching my own gene_id string," just
    for different downstream shapes (a window-generation tuple list vs. a
    (start, end) span dict), which this function's dict return supports
    either from directly.

    Args:
        external_gene_ids: Gene IDs as spelled by the external source —
            these become the keys of the returned dict, even though
            GENCODE's own gene_id for the same stable gene may carry a
            different version suffix.
        data_dir: Override default annotation cache directory.

    Returns:
        external_gene_id -> GencodeGene, for every requested ID that has a
        matching stable gene in GENCODE. IDs with no match are omitted, not
        raised or filled in — callers that need to know which requested IDs
        were missing should diff the returned dict's keys against
        external_gene_ids themselves; this function logs the match rate.
    """
    genes = load_gencode_genes(gene_ids=external_gene_ids, data_dir=data_dir)
    external_id_by_stable = {stable_gene_id(gid): gid for gid in external_gene_ids}
    result = {
        external_id_by_stable[stable_gene_id(g.gene_id)]: g
        for g in genes
        if stable_gene_id(g.gene_id) in external_id_by_stable
    }
    logger.info(
        "Resolved %d/%d external gene IDs to GENCODE records (stable-ID matched)",
        len(result), len(external_gene_ids),
    )
    return result


def label_dna_windows_by_gene(
    windows: list[tuple[str, str, int, int, str]],
) -> dict[str, set[str]]:
    """
    Build the window_id -> {gene_id} label dict biolens.eval.probing.
    probe_genomic_annotations expects, from generate_gene_windows' output.

    Unlike label_dna_windows (overlap search against a separately-loaded
    annotation set), each window from generate_gene_windows already carries
    its source gene_id — this is a direct reshape, not a search, since
    windows are generated per-gene in the first place.

    Args:
        windows: (window_id, chrom, start, end, gene_id) tuples from
            biolens.data.reference_genome.generate_gene_windows().

    Returns:
        window_id -> {gene_id} (a singleton set, for a uniform interface
        with label_dna_windows' possibly-multi-label return type).
    """
    return {window_id: {gene_id} for window_id, _chrom, _start, _end, gene_id in windows}


def find_overlapping_genes(
    windows: list[tuple[str, str, int, int]],
    genes: list[GencodeGene],
) -> dict[str, list[GencodeGene]]:
    """
    Find which real GENCODE gene(s) each window genuinely falls within, by
    genomic coordinate overlap — the reverse direction of
    label_dna_windows_by_gene (which labels a window with the gene it was
    GENERATED for). This instead answers "what gene does an arbitrary
    genome-wide coordinate actually sit in", which is what
    biolens.eval.genomic_verification needs: checking whether a candidate
    feature's real top-activating windows (pulled genome-wide, not from the
    artificially gene-locus-restricted window set a naive probing pass used)
    actually fall within the gene that pass claims the feature is "about".

    Args:
        windows: (window_id, chrom, start, end) tuples — 0-indexed,
            half-open, e.g. from a whole-genome tiling cache's window IDs.
        genes:   From load_gencode_genes() (or load_gencode_genes_by_
            external_id().values()) — should generally be an UNFILTERED or
            broadly-filtered gene list, since a window's real overlapping
            gene may not be among any small candidate set the caller had in
            mind.

    Returns:
        window_id -> list of overlapping GencodeGene records (possibly
        empty — an intergenic window overlaps no gene; possibly more than
        one, for windows in a region of overlapping gene bodies).

    Correctness note: mirrors label_dna_windows' bisect + backward-scan-
    bound approach, but with a much larger bound (_MAX_PLAUSIBLE_GENE_
    LENGTH_BP) since genes, unlike cCREs, can span over a megabase. Still
    far cheaper than a real interval tree for the actual call volume here —
    verifying a feature's top few dozen activating windows, not labeling
    every window in the genome.
    """
    by_chrom: dict[str, list[GencodeGene]] = {}
    for g in genes:
        by_chrom.setdefault(g.chrom, []).append(g)
    index: dict[str, tuple[list[int], list[GencodeGene]]] = {}
    for chrom, items in by_chrom.items():
        items.sort(key=lambda g: g.start)
        index[chrom] = ([g.start for g in items], items)

    result: dict[str, list[GencodeGene]] = {}
    for window_id, chrom, start, end in windows:
        starts, items = index.get(chrom, ([], []))
        hits: list[GencodeGene] = []

        if items:
            upper = bisect.bisect_left(starts, end) - 1
            cutoff = start - _MAX_PLAUSIBLE_GENE_LENGTH_BP
            j = upper
            while j >= 0 and items[j].start >= cutoff:
                gene = items[j]
                if gene.start < end and gene.end > start:  # true interval overlap
                    hits.append(gene)
                j -= 1

        result[window_id] = hits

    return result
