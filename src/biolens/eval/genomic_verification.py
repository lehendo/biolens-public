"""
Verification of naive-feature-probing results in the genomic (Evo 2) domain
— the genomic-domain analog of feature_inspection.VerifiedAnnotation (built
for ESM2/UniProt), adapted for a structurally different claim type.

For a protein feature, "this feature means GO term Y" is checked against a
real ground-truth database (UniProt) — see biolens.data.uniprot and the
configs/verified_features/*.yaml registries. There is no equivalent database
for "this feature discriminates gene Y's windows": a DNA window's identity
IS its genomic coordinate. So verification here instead:

  1. Pulls each candidate feature's REAL top-activating windows genome-wide
     (from the standard whole-genome activation cache used for cCRE probing
     — not the artificially gene-locus-restricted window set the naive
     probing pass itself used, which only ever compared "this gene's
     windows" against "other candidate genes' windows", never against the
     rest of the genome).
  2. Looks up which real gene (if any) each top-activating window actually
     falls within, via biolens.data.genomic_annotations.find_overlapping_genes.
  3. Checks whether that matches the gene the naive-probing pass claims the
     feature is "about".

Two independent, real failure modes this catches (neither reduces to the
other):
  - A feature's top-activating windows genome-wide never touch the claimed
    gene at all — the naive probing pass's within-locus AUROC was driven by
    something that happens to separate this gene's small window set from
    its comparison genes' windows, not a real feature-gene relationship.
  - A feature is the single-feature-AUROC-best discriminator for many
    different, unrelated candidate genes (real data check, job 9712776's
    naive_feature_results.json, 2026-07-30: right-skewed — 20 of 75 unique
    features are individually "best" for 6-31 different genes, while 29 are
    claimed by exactly one). A feature that promiscuous is picking up on
    something generic (GC content, a repeat family, window-composition
    artifacts) that happens to separate any one gene's windows from
    everyone else's in a small per-locus comparison — not a real
    gene-specific signal, regardless of what its top windows overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from biolens.data.genomic_annotations import GencodeGene, find_overlapping_genes, stable_gene_id

# See module docstring's second failure mode: a feature claimed by more than
# this many distinct candidate genes is classified "spurious" outright
# (too promiscuous to be specifically about any one of them), without even
# checking window overlap.
DEFAULT_MAX_CLAIMS_BEFORE_GENERIC = 5

# Fraction of a feature's top-activating windows (genome-wide) that must
# fall within a claimed gene for "confirmed" (vs. "plausible" for partial,
# non-majority overlap, vs. "spurious" for none).
DEFAULT_CONFIRMED_MATCH_RATE = 0.5


@dataclass(frozen=True)
class GenomicFeatureVerification:
    feature_idx: int
    status: str  # "confirmed" | "plausible" | "spurious" | "unverifiable"
    reason: str
    claimed_gene_ids: list[str]
    n_claims: int
    match_rate: float
    matched_windows: list[str] = field(default_factory=list)
    # (window_id, [overlapping gene_id, ...]) for every ranked window checked
    # — kept as evidence so a status can be spot-checked without re-running
    # the whole pipeline.
    evidence: list[tuple[str, list[str]]] = field(default_factory=list)


def verify_genomic_feature(
    feature_idx: int,
    claimed_gene_ids: list[str],
    ranked_windows: list[tuple[str, str, int, int]],
    genes: list[GencodeGene],
    max_claims_before_generic: int = DEFAULT_MAX_CLAIMS_BEFORE_GENERIC,
    confirmed_match_rate: float = DEFAULT_CONFIRMED_MATCH_RATE,
) -> GenomicFeatureVerification:
    """
    Classify one candidate feature's naive-probing claim(s) against its real
    top-activating windows.

    Args:
        feature_idx: The SAE feature index being verified.
        claimed_gene_ids: Every gene_id (external spelling, e.g. Geuvadis'
            own gene_id string — see genomic_annotations.
            load_gencode_genes_by_external_id) this feature was the
            single-feature-AUROC-best discriminator for, in the naive
            probing pass.
        ranked_windows: This feature's top-activating windows GENOME-WIDE
            (not gene-locus-restricted), as (window_id, chrom, start, end)
            tuples, ranked by activation descending — e.g. from
            biolens.eval.feature_inspection.inspect_feature's top_examples,
            with each protein_id/window_id parsed back into (chrom, start,
            end).
        genes: A broad (ideally unfiltered) GencodeGene list to search for
            real overlaps in — a window's true overlapping gene may not be
            among the small candidate-gene set the naive pass considered.
        max_claims_before_generic, confirmed_match_rate: See module-level
            defaults' docstrings.

    Returns:
        GenomicFeatureVerification with a reasoned status and the window-
        level evidence used to reach it.
    """
    if not claimed_gene_ids:
        return GenomicFeatureVerification(
            feature_idx=feature_idx,
            status="unverifiable",
            reason="no claimed gene IDs given",
            claimed_gene_ids=[],
            n_claims=0,
            match_rate=0.0,
        )

    n_claims = len(claimed_gene_ids)
    overlap_by_window = find_overlapping_genes(ranked_windows, genes)
    evidence = [
        (window_id, sorted({g.gene_id for g in overlap_by_window.get(window_id, [])}))
        for window_id, _chrom, _start, _end in ranked_windows
    ]

    if n_claims > max_claims_before_generic:
        return GenomicFeatureVerification(
            feature_idx=feature_idx,
            status="spurious",
            reason=(
                f"claimed by {n_claims} different genes (> {max_claims_before_generic}) "
                "— too promiscuous to be specifically about any one of them"
            ),
            claimed_gene_ids=claimed_gene_ids,
            n_claims=n_claims,
            match_rate=0.0,
            evidence=evidence,
        )

    claimed_stable = {stable_gene_id(g) for g in claimed_gene_ids}
    matched_windows = [
        window_id
        for window_id, hit_gene_ids in evidence
        if any(stable_gene_id(gid) in claimed_stable for gid in hit_gene_ids)
    ]
    match_rate = len(matched_windows) / len(ranked_windows) if ranked_windows else 0.0

    if match_rate >= confirmed_match_rate:
        status = "confirmed"
        reason = (
            f"{len(matched_windows)}/{len(ranked_windows)} top-activating windows "
            "genome-wide fall within a claimed gene"
        )
    elif match_rate > 0:
        status = "plausible"
        reason = (
            f"only {len(matched_windows)}/{len(ranked_windows)} top-activating windows "
            "fall within a claimed gene — partial, non-majority overlap"
        )
    else:
        status = "spurious"
        reason = "none of the top-activating windows genome-wide fall within any claimed gene"

    return GenomicFeatureVerification(
        feature_idx=feature_idx,
        status=status,
        reason=reason,
        claimed_gene_ids=claimed_gene_ids,
        n_claims=n_claims,
        match_rate=match_rate,
        matched_windows=matched_windows,
        evidence=evidence,
    )


def load_genomic_verified_features(path: str | Path) -> dict[int, GenomicFeatureVerification]:
    """
    Load a genomic verified_features registry (scripts/
    verify_geuvadis_naive_features.py's YAML output) -> feature_idx ->
    GenomicFeatureVerification.

    Distinct from biolens.eval.feature_inspection.load_verified_annotations:
    that loader reads the ESM2/UniProt VerifiedAnnotation schema (claimed_go,
    claimed_concept, real_identity, evidence_accessions) — a differently-
    shaped claim type this module's module docstring explains isn't
    meaningful for DNA windows. Consumers needing "is there a confirmed
    feature for this genomic SAE checkpoint" (e.g.
    scripts/run_geuvadis_case_study.py) should use this loader, not that one.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    return {
        int(feat_idx): GenomicFeatureVerification(
            feature_idx=int(feat_idx),
            status=entry["status"],
            reason=entry.get("reason", ""),
            claimed_gene_ids=entry.get("claimed_gene_ids", []),
            n_claims=entry.get("n_claims", 0),
            match_rate=entry.get("match_rate", 0.0),
            matched_windows=entry.get("matched_windows", []),
        )
        for feat_idx, entry in (data.get("features") or {}).items()
    }
