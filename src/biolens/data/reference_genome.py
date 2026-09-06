"""
Reference-genome sequence access, shared by two pipelines that both need it
(avoiding building it twice):
  - Genomic-annotation probing: tile the genome into fixed windows, extract
    Evo 2 activations, label via ENCODE cCRE overlap
    (biolens.data.genomic_annotations).
  - Geuvadis case study: extract symmetric reference/alt-allele sequence
    context around 1000 Genomes variants.

Uses pyfaidx — the standard Python package for efficient random-access
indexed FASTA reading (samtools-faidx-equivalent), not a hand-rolled parser;
a well-established, battle-tested tool, not something worth reinventing.

UNTESTED against a real reference genome file in this environment: GRCh38
primary assembly is ~3GB, far too large to download in this sandbox. Tested
here against small synthetic FASTA fixtures that exercise the same pyfaidx
API and coordinate logic — the windowing/substitution logic itself is fully
verified; only "does a real 3GB GRCh38 file behave the same way" is
unverified, which is standard, well-established pyfaidx usage, not a novel
code path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyfaidx
    from pyliftover import LiftOver

logger = logging.getLogger(__name__)

_VALID_BASES = set("ACGT")


def load_reference_genome(fasta_path: str | Path) -> pyfaidx.Fasta:
    """
    Open an indexed reference genome FASTA (builds the .fai index on first
    use if missing — pyfaidx handles this automatically). Raises a clear,
    actionable error if the file doesn't exist, rather than pyfaidx's more
    opaque internal error.
    """
    import pyfaidx

    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"Reference genome FASTA not found: {fasta_path}. Download e.g. GRCh38 "
            f"primary assembly from https://hgdownload.soe.ucsc.edu/goldenPath/hg38/"
            f"bigZips/hg38.fa.gz (large — ~3GB uncompressed) before running this."
        )
    return pyfaidx.Fasta(str(fasta_path))


def _passes_n_threshold(seq: str, skip_n_threshold: float) -> bool:
    """True if `seq`'s fraction of 'N' bases is within skip_n_threshold —
    shared by generate_tiling_windows and generate_gene_windows, both of
    which skip N-heavy (unassembled/masked) windows the same way."""
    n_frac = seq.upper().count("N") / len(seq) if seq else 1.0
    return n_frac <= skip_n_threshold


def generate_tiling_windows(
    fasta: pyfaidx.Fasta,
    window_size: int,
    stride: int,
    chroms: list[str] | None = None,
    skip_n_threshold: float = 0.1,
    offset: int = 0,
) -> list[tuple[str, str, int, int]]:
    """
    Tile chromosome(s) into fixed-size, fixed-stride windows for
    genomic-annotation probing.

    Args:
        fasta:       Loaded reference genome (load_reference_genome()).
        window_size: Window length in bp.
        stride:      Step between consecutive window starts (< window_size
                     for overlapping windows, == window_size for tiling with
                     no overlap).
        chroms:      Restrict to these chromosomes; None = every sequence in
                     the FASTA (careful — includes scaffolds/patches for a
                     full assembly; pass an explicit chr1..chr22,chrX,chrY
                     list for a standard analysis).
        skip_n_threshold: Skip a window if more than this fraction of its
                     bases are 'N' (unassembled/masked regions) — an N-heavy
                     window has no real sequence signal for the model to
                     extract activations from.
        offset:      Shift the tiling grid's starting position by this many
                     bp (0 <= offset < stride for a genuinely different,
                     non-overlapping-with-offset=0 window set from the same
                     stride). Deterministic and reproducible, unlike a random
                     window draw — exists so a replication check (e.g. a
                     PLS-anomaly follow-up) can draw an independent eval
                     window set from the exact same tiling scheme without
                     adding RNG plumbing.

    Returns:
        List of (window_id, chrom, start, end) tuples, 0-indexed half-open,
        matching BED convention (directly compatible with
        biolens.data.genomic_annotations.label_dna_windows).
    """
    if not (0 <= offset < stride):
        raise ValueError(f"offset must satisfy 0 <= offset < stride ({stride}), got {offset}")

    target_chroms = chroms if chroms is not None else list(fasta.keys())
    windows: list[tuple[str, str, int, int]] = []

    for chrom in target_chroms:
        if chrom not in fasta:
            logger.warning("Chromosome %s not found in reference genome — skipping", chrom)
            continue
        chrom_len = len(fasta[chrom])

        for start in range(offset, max(chrom_len - window_size + 1, 0), stride):
            end = start + window_size
            seq = str(fasta[chrom][start:end])
            if not _passes_n_threshold(seq, skip_n_threshold):
                continue
            windows.append((f"{chrom}:{start}-{end}", chrom, start, end))

    logger.info(
        "Generated %d tiling windows (window_size=%d, stride=%d, offset=%d) across %d chromosome(s)",
        len(windows), window_size, stride, offset, len(target_chroms),
    )
    return windows


def generate_gene_windows(
    fasta: pyfaidx.Fasta,
    genes: list[tuple[str, str, int, int]],
    window_size: int,
    stride: int,
    flank_bp: int = 2000,
    skip_n_threshold: float = 0.1,
) -> list[tuple[str, str, int, int, str]]:
    """
    Tile windows across each given gene's body (+ symmetric flanking region)
    — the gene-scoped analog of generate_tiling_windows' whole-chromosome
    tiling, for gene-locus-level probing (Geuvadis naive-feature pass: the
    naive-arm feature for each candidate eQTL gene needs its own probing
    target, not a genome-wide cCRE-class label).

    Unlike generate_tiling_windows + biolens.data.genomic_annotations.
    label_dna_windows' separate generate-then-overlap-search pattern
    (appropriate there because cCREs are short — bounded at 5000bp — and
    numerous relative to any window set), each window here already knows
    which gene it belongs to at generation time: gene bodies can span
    >1Mb, so label_dna_windows' fixed-radius backward-scan bound doesn't
    hold for gene-length intervals, and generating per-gene sidesteps
    needing a proper interval tree for what is, in practice, a short
    (hundreds, not millions) candidate gene list.

    Args:
        fasta: Loaded reference genome.
        genes: (gene_id, chrom, start, end) tuples — 0-indexed, half-open,
            from biolens.data.genomic_annotations.load_gencode_genes().
        window_size, stride, skip_n_threshold: Same semantics as
            generate_tiling_windows.
        flank_bp: Extend each gene's region by this many bp on both sides
            before tiling — captures proximal promoter/regulatory context
            around the gene body, not just the transcribed region itself.

    Returns:
        (window_id, chrom, start, end, gene_id) tuples. window_id embeds
        the gene_id and coordinates so it stays unique even where two
        genes' flanked regions physically overlap (each is a separate
        probing instance, not a deduplicated genomic position).
    """
    windows: list[tuple[str, str, int, int, str]] = []
    for gene_id, chrom, start, end in genes:
        if chrom not in fasta:
            logger.warning("Gene %s: chrom %s not in reference genome — skipping", gene_id, chrom)
            continue
        chrom_len = len(fasta[chrom])
        region_start = max(0, start - flank_bp)
        region_end = min(chrom_len, end + flank_bp)

        for pos in range(region_start, max(region_end - window_size + 1, region_start), stride):
            win_end = pos + window_size
            seq = str(fasta[chrom][pos:win_end])
            if not _passes_n_threshold(seq, skip_n_threshold):
                continue
            windows.append((f"{gene_id}:{chrom}:{pos}-{win_end}", chrom, pos, win_end, gene_id))

    logger.info(
        "Generated %d gene-locus windows (window_size=%d, stride=%d, flank_bp=%d) across %d gene(s)",
        len(windows), window_size, stride, flank_bp, len(genes),
    )
    return windows


def extract_window_sequences(
    fasta: pyfaidx.Fasta,
    windows: list[tuple[str, str, int, int]],
) -> dict[str, str]:
    """Extract uppercase sequence strings for each (window_id, chrom, start,
    end) tuple from generate_tiling_windows()."""
    return {
        window_id: str(fasta[chrom][start:end]).upper()
        for window_id, chrom, start, end in windows
    }


def extract_ref_alt_context(
    fasta: pyfaidx.Fasta,
    chrom: str,
    pos: int,
    ref_allele: str,
    alt_allele: str,
    context_bp: int,
) -> tuple[str, str]:
    """
    Extract a symmetric reference/alt-allele sequence window around a 1000
    Genomes variant (Geuvadis case study).

    Args:
        fasta:       Loaded reference genome.
        chrom:       Chromosome (must match the FASTA's naming convention —
                     e.g. "chr1" vs "1"; check against `fasta.keys()` if
                     variant calls use a different convention than the
                     reference file, a common real-world mismatch).
        pos:         1-based variant position (VCF convention).
        ref_allele:  Reference allele from the VCF (e.g. "A").
        alt_allele:  Alternate allele from the VCF (e.g. "G"). Only
                     single-nucleotide substitutions are supported — indels
                     would shift the window's coordinate frame and need
                     separate handling, out of scope for this feasibility
                     case study (a deliberate scope choice, not a
                     definitive analysis).
        context_bp:  Number of bases on EACH side of the variant (total
                     window length = 2 * context_bp + 1).

    Returns:
        (ref_sequence, alt_sequence) — same length, differing only at the
        variant position.

    Raises:
        ValueError: if ref_allele/alt_allele aren't both single bases, or if
            the reference genome's base at `pos` doesn't match `ref_allele`
            (a real, not-uncommon data-quality check — silently proceeding
            on a ref-allele mismatch would produce a silently wrong "alt"
            sequence, exactly the kind of unverified claim this project's
            own standard exists to catch).
    """
    # A symmetric 2*context_bp+1 window is exactly what extract_ref_alt_window
    # produces when window_size//2 == context_bp — true for every odd
    # window_size, which 2*context_bp+1 always is. Delegating avoids
    # duplicating the allele-validation/mismatch-check/alt-splice logic here.
    ref_seq, alt_seq, _window_start = extract_ref_alt_window(
        fasta, chrom, pos, ref_allele, alt_allele, window_size=2 * context_bp + 1
    )
    return ref_seq, alt_seq


def _validate_snv_alleles(ref_allele: str, alt_allele: str) -> None:
    if len(ref_allele) != 1 or len(alt_allele) != 1:
        raise ValueError(
            f"Only single-nucleotide substitutions supported (got ref={ref_allele!r}, "
            f"alt={alt_allele!r}) — indels shift the coordinate frame and need separate handling"
        )
    if ref_allele.upper() not in _VALID_BASES or alt_allele.upper() not in _VALID_BASES:
        raise ValueError(f"Non-ACGT allele: ref={ref_allele!r}, alt={alt_allele!r}")


def extract_ref_alt_window(
    fasta: pyfaidx.Fasta,
    chrom: str,
    pos: int,
    ref_allele: str,
    alt_allele: str,
    window_size: int,
) -> tuple[str, str, int]:
    """
    Like extract_ref_alt_context, but for an EXACT total window_size
    (odd or even) rather than a symmetric `2 * context_bp + 1` window —
    needed for models with a fixed input length that isn't necessarily odd
    (e.g. Borzoi's 524288bp, biolens.eval.mediation_baselines.
    borzoi_baseline — extract_ref_alt_context's `2 * context_bp + 1` can
    never equal an even number, so it can't produce a Borzoi-sized window
    at all, regardless of context_bp).

    The variant is placed as close to center as the exact window_size
    allows: `window_size // 2` bases before it, the rest after (for an
    even window_size this means one fewer base after the variant than
    before it).

    Args:
        fasta, chrom, pos, ref_allele, alt_allele: Same as
            extract_ref_alt_context.
        window_size: Exact total output length in bp.

    Returns:
        (ref_sequence, alt_sequence, window_start) — ref/alt are the same
        length (window_size), differing only at the variant position.
        window_start is the 0-indexed genomic coordinate of the window's
        first base — needed by callers that must map model OUTPUT bins
        back to genomic coordinates (e.g. Borzoi's track-to-gene
        aggregation, which needs to know exactly where in the genome bin 0
        of the model's output corresponds to).

    Raises:
        ValueError: same conditions as extract_ref_alt_context (non-SNV
            alleles, window running off the start of the chromosome, or a
            reference-genome/VCF mismatch at the variant position).
    """
    _validate_snv_alleles(ref_allele, alt_allele)

    variant_idx = pos - 1  # VCF is 1-based; pyfaidx/BED-style slicing is 0-based
    start = variant_idx - window_size // 2
    end = start + window_size
    if start < 0:
        raise ValueError(
            f"Variant at {chrom}:{pos} is too close to the chromosome start for "
            f"window_size={window_size} (would read before position 0)"
        )
    chrom_len = len(fasta[chrom])
    if end > chrom_len:
        # pyfaidx slicing past the end silently truncates rather than
        # raising or padding — without this check, a variant near a
        # chromosome's tail produces a ref_seq shorter than window_size,
        # and the relative_pos indexing below crashes with a raw
        # IndexError instead of the clean ValueError callers already catch
        # (real production failure, job 9727048, 2026-07-31 — a per-locus
        # ValueError is caught and skipped by run_geuvadis_case_study.py's
        # _run_one_locus, but an uncaught IndexError crashed the whole run).
        raise ValueError(
            f"Variant at {chrom}:{pos} is too close to the chromosome end for "
            f"window_size={window_size} (would read past position {chrom_len})"
        )

    ref_seq = str(fasta[chrom][start:end]).upper()
    relative_pos = variant_idx - start
    actual_ref_base = ref_seq[relative_pos]
    if actual_ref_base != ref_allele.upper():
        raise ValueError(
            f"Reference genome mismatch at {chrom}:{pos} — FASTA has "
            f"{actual_ref_base!r}, VCF says ref={ref_allele!r}. Check chromosome "
            f"naming convention (chr1 vs 1) and genome build (must be the same "
            f"GRCh38 build the VCF was called against)."
        )

    alt_seq = ref_seq[:relative_pos] + alt_allele.upper() + ref_seq[relative_pos + 1 :]
    return ref_seq, alt_seq, start


def load_liftover_chain(chain_path: str | Path) -> LiftOver:
    """
    Load a UCSC liftOver chain file (e.g. hg19ToHg38.over.chain.gz —
    https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/) for
    build-to-build genomic coordinate conversion.

    Real, concrete need this exists for (not speculative): the Geuvadis
    case study's published eQTL candidate file and the 1000 Genomes Phase 3
    VCF (`ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502...`) are
    both GRCh37-coordinate — confirmed via a real discrepancy, not assumed
    (rs6002851: the VCF reports chr22:43,041,837; the real GRCh38 position,
    per Ensembl, is chr22:42,645,831 — a 396,006bp difference, definitively
    a build mismatch, not a rounding error) — while the reference genome
    FASTA, GENCODE gene annotations, and the Borzoi baseline this pipeline
    otherwise uses are all GRCh38. `extract_ref_alt_context`'s own ref-base
    mismatch check only catches this by COINCIDENCE (whenever a wrong-build
    lookup happens to land on a different base than the true ref allele,
    not because the position is actually validated) — job 9727212
    (2026-07-31) "completed" 6 loci that all happened to pass that check
    purely by chance at the wrong genomic position, not because their
    coordinates were correct.

    Raises:
        ImportError: if `pyliftover` isn't installed (`pip install
            pyliftover`).
    """
    try:
        from pyliftover import LiftOver
    except ImportError as exc:
        raise ImportError(
            "The 'pyliftover' package is required for genome-build coordinate "
            "conversion but is not installed. Install with `pip install pyliftover`."
        ) from exc
    return LiftOver(str(chain_path))


def liftover_position(lo: LiftOver, chrom: str, pos: int) -> int | None:
    """
    Convert a 1-based genomic position from the liftOver chain's source
    build to its target build (e.g. GRCh37 -> GRCh38), via load_liftover_
    chain()'s loaded chain file.

    Args:
        lo: A loaded LiftOver object (load_liftover_chain()).
        chrom: "chr"-prefixed chromosome (e.g. "chr22").
        pos: 1-based source-build position (VCF convention).

    Returns:
        1-based target-build position, or None if no unambiguous mapping
        exists (deleted/reorganized region between builds, or a mapping
        that splits across multiple target chromosomes/positions — a real,
        if uncommon, liftOver outcome that must not be silently guessed at;
        callers should treat None the same as any other per-locus data
        issue, i.e. skip that locus, not raise).
    """
    results = lo.convert_coordinate(chrom, pos - 1)  # pyliftover is 0-based
    if not results or len(results) != 1:
        return None
    lifted_chrom, lifted_pos_0based, _strand, _score = results[0]
    if lifted_chrom != chrom:
        return None
    return lifted_pos_0based + 1
