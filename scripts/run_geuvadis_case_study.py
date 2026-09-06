"""
Geuvadis case study — a feasibility-scale demonstration of why feature
verification matters practically, not just statistically.

REBUILT 2026-08 after a methodological review found the original mediator
construction (`M = Delta * D`, a single locus-level activation-shift
scalar times genotype dosage) was NOT just weakly identified but exactly
rank-deficient: M is a deterministic rescaling of the treatment D, so the
outcome regression's design matrix cannot separate the mediated (M) and
direct (D) effects — confirmed empirically (two synthetic datasets with 100%
vs 0% true mediation were statistically indistinguishable under that
construction; see biolens.eval.mediation.run_positive_control's docstring and
test_mediation.py's TestRunPositiveControl for the regression test that now
guards against this permanently). This invalidates the original 26/26
"prop_mediated=0.000" result recorded in docs/geuvadis_case_study_results.json
— that was a scale artifact of the SAE's arbitrary activation units, not a
real finding.

The fix (biolens.data.geuvadis.load_window_genotypes /
build_haplotype_sequence): instead of one ref/alt forward pass per locus,
build each individual's TRUE personal local haplotype sequence — substitute
ALL nearby phased variants (not just the lead eQTL SNP) into the reference
window, separately per haplotype — and use that individual's own measured SAE
feature activation as their mediator value. Two individuals with the SAME
dosage at the lead variant can now have DIFFERENT mediator values if they
carry different other nearby variants, giving the design genuine
Var(M | D) > 0 — the property required for ACME/ADE to be separably
identified at all.

For each candidate (variant, gene) locus from a pre-specified published
Geuvadis eQTL list:
  1. Extract the GRCh38 reference window around the (liftOver'd) lead
     variant (biolens.data.reference_genome).
  2. Build every sample's two personal haplotype sequences in that window
     (biolens.data.geuvadis.build_haplotype_sequence, using nearby variants
     from biolens.data.geuvadis.load_window_genotypes) and run them through
     Evo 2 + the SAE — deduplicating identical haplotype sequences first
     (many individuals share a local haplotype, especially within a
     population — see _compute_per_individual_activations) to keep the
     ~2*n_samples-per-locus forward-pass cost tractable.
  3. Compute the per-individual SAE feature activation for the naive
     best-AUROC feature (always) and, when the genomic verified_features
     registry (biolens.eval.genomic_verification) has at least one
     "confirmed" feature, a second "verified" arm using that feature —
     the two-arm comparison that demonstrates verification's practical
     stakes (a single run with verified features alone does not logically
     establish this).

     Real data check (2026-07-30, job 9724784): the first genomic
     verification pass found 0/75 candidate features confirmed — every
     naive-probing "best feature" failed the genome-wide window-overlap
     check against its claimed gene. When that's the case, the "verified"
     arm has nothing to run (there's no confirmed feature to use), so this
     script still runs the naive arm + all baselines for every locus and
     additionally attaches that naive feature's own verification verdict
     (status/reason/match_rate) as metadata on each result — the finding
     becomes "the naive arm's mediation results, however they look, are
     for a feature verification independently rejects," not a missing
     comparison. If a future, larger-scale Evo 2 SAE run does produce
     confirmed features, this script picks the "verified" arm back up
     automatically (see verified_feat_idx below) with no code changes.
  4. Run IKT mediation (biolens.eval.mediation) for each arm that has a
     feature to run, using the real per-individual mediator from step 2.
  5. Run the negative-control arm (random feature).
  6. Run all three baselines (biolens.eval.mediation_baselines): raw-
     embedding (now also genuinely per-individual — each sample's own dense-
     embedding distance from the pure reference, not a locus-level scalar
     times dosage; this also resolves the raw-embedding/TWAS baseline
     redundancy the same review flagged, since the two are no longer the
     same scalar-times-D shape), TWAS-style, and Borzoi (pretrained
     sequence-to-expression inference — needs its own wider, Borzoi-sized
     ref/alt window and the target gene's GENCODE span, both handled
     internally here; skip with --skip-borzoi if the `borzoi-pytorch`
     package isn't installed; Borzoi's baseline is intentionally left at
     the locus-level ref/alt comparison — it's a SOTA-sequence-model
     comparison point, not part of the identification fix).
  7. Run the sensitivity analysis for each arm.
  8. After every locus completes, apply Benjamini-Hochberg FDR correction
     across all completed loci's ACME p-values, per arm (biolens.eval.
     statistics.apply_multiplicity_correction) — real gap found by external
     methodological review, 2026-08-05: no multiplicity correction existed
     anywhere in this codebase despite testing multiple loci per run.

Before touching any real locus, main() runs a startup positive-control check
(_run_positive_control_check) against synthetic data with a KNOWN true
mediation fraction, through the exact same run_ikt_mediation code path used
below — and aborts the whole run if it fails. This is the review's own
highest-value recommendation operationalized as a permanent safeguard, not a
one-off script: it is what would have caught the original M=Delta*D bug
immediately, and it runs on every invocation so a future regression can't
silently reproduce it.

Requires a trained Evo 2 SAE checkpoint, real Geuvadis genotype (VCF) and
expression data, and a genomic verified_features registry for that
checkpoint (scripts/verify_geuvadis_naive_features.py's output) — none of
which exist locally in this dev environment; this script is the
orchestration layer tying together the individually-tested components in
biolens.data.geuvadis / biolens.data.reference_genome / biolens.eval.mediation
/ biolens.eval.mediation_baselines, meant to run on the cluster once those
inputs exist.

Usage:
  python scripts/run_geuvadis_case_study.py \\
      --sae-checkpoint /path/to/evo2_7b_sae/final.pt \\
      --model evo2_7b --layer 16 \\
      --reference-genome /path/to/hg38.fa \\
      --vcf /path/to/geuvadis_genotypes.vcf.gz \\
      --expression /path/to/geuvadis_expression.tsv \\
      --eqtl-candidates /path/to/published_geuvadis_eqtls.tsv \\
      --naive-feature-results /path/to/naive_feature_results.json \\
      --genomic-verified-features configs/verified_features/evo2_7b_layer16_topk_k32_geuvadis_naive.yaml \\
      --liftover-chain /path/to/hg19ToHg38.over.chain.gz \\
      --population-labels /path/to/integrated_call_samples_v3.20130502.ALL.panel \\
      --context-bp 4096 \\
      --output-dir docs/geuvadis_case_study_results
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.data.genomic_annotations import (  # noqa: E402
    load_gencode_genes_by_external_id,
)
from biolens.data.geuvadis import (  # noqa: E402
    build_ancestry_covariates,
    build_haplotype_sequence,
    build_mediation_arrays,
    load_expression_matrix,
    load_population_labels,
    load_published_eqtl_candidates,
    load_variant_genotypes,
    load_window_genotypes,
    summarize_population_composition,
)
from biolens.data.reference_genome import (  # noqa: E402
    extract_ref_alt_window,
    liftover_position,
    load_liftover_chain,
    load_reference_genome,
)
from biolens.eval.genomic_verification import load_genomic_verified_features  # noqa: E402
from biolens.eval.mediation import (  # noqa: E402
    run_ikt_mediation,
    run_negative_control,
    run_positive_control,
    sensitivity_analysis,
)
from biolens.eval.mediation_baselines import (  # noqa: E402
    borzoi_baseline,
    extract_borzoi_ref_alt_window,
    load_borzoi_model,
    raw_embedding_baseline,
    twas_style_baseline,
)
from biolens.eval.statistics import apply_multiplicity_correction  # noqa: E402
from biolens.models.registry import ModelRegistry  # noqa: E402
from biolens.sae.train import build_sae_from_checkpoint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# How far a recovered positive-control proportion_mediated may drift from
# its known true value before the run is aborted — matches the tolerance
# test_mediation.py's TestRunPositiveControl already validates in CI
# (abs=0.15); a real locus run is far more expensive than that test, so this
# check must use the same tolerance, not a looser one just because it's
# running standalone here.
_POSITIVE_CONTROL_TOLERANCE = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-checkpoint", required=True)
    p.add_argument("--model", default="evo2_7b")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--reference-genome", required=True)
    p.add_argument("--vcf", required=True)
    p.add_argument("--expression", required=True)
    p.add_argument("--eqtl-candidates", required=True)
    p.add_argument(
        "--naive-feature-results", required=True,
        help="go_probing_results.json (genomic-annotation probing run) — "
             "supplies the naive best-AUROC feature per locus",
    )
    p.add_argument(
        "--genomic-verified-features", required=True,
        help="Path to a genomic verified_features registry — feature_idx -> "
             "verification verdict, built by scripts/verify_geuvadis_naive_"
             "features.py checking each naive-probing candidate's real top-"
             "activating windows genome-wide against GENCODE. Distinct from "
             "the ESM2/UniProt verified_features schema (biolens.eval."
             "feature_inspection.VerifiedAnnotation) — there's no ground-"
             "truth database to check a DNA window's identity against "
             "directly, so this registry's claim type is structurally "
             "different (see biolens.eval.genomic_verification's docstring).",
    )
    p.add_argument(
        "--liftover-chain", required=True,
        help="Path to a UCSC liftOver chain file (e.g. hg19ToHg38.over.chain.gz — "
             "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/). Required, "
             "not optional: the Geuvadis eQTL candidate file and 1000 Genomes Phase "
             "3 VCF are GRCh37-coordinate, but the reference genome FASTA, GENCODE "
             "annotations, and Borzoi baseline this script otherwise uses are all "
             "GRCh38 — confirmed via a real discrepancy (rs6002851: VCF says "
             "chr22:43,041,837, real GRCh38 position per Ensembl is "
             "chr22:42,645,831, 2026-07-31), not assumed. Every nearby variant used "
             "to build a personal haplotype sequence (not just the lead SNP) is "
             "individually lifted with this same chain — see "
             "biolens.data.geuvadis.load_window_genotypes's docstring. See "
             "biolens.data.reference_genome.load_liftover_chain's docstring for "
             "the full story of how this was caught.",
    )
    p.add_argument("--context-bp", type=int, default=4096)
    p.add_argument("--n-rep", type=int, default=1000)
    p.add_argument(
        "--population-labels", required=True,
        help="Path to 1000 Genomes' own published sample-to-population panel "
             "(e.g. integrated_call_samples_v3.20130502.ALL.panel from "
             "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/). "
             "Required, not optional: population stratification is a real, "
             "unaddressed confound flagged as the single biggest open item "
             "in the original design — "
             "every mediation/sensitivity/baseline regression below adjusts "
             "for it via biolens.data.geuvadis.build_ancestry_covariates.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--activation-batch-size", type=int, default=32,
        help="Sequences per get_activations() chunk when computing per-individual "
             "haplotype activations (biolens.models.evo2.Evo2Model.get_activations' "
             "batch_size — a Python-level loop chunk size, not a padded batched "
             "forward pass). Up to ~2*n_samples unique sequences per locus before "
             "deduplication; tune to the GPU's real memory headroom.",
    )
    p.add_argument(
        "--max-loci", type=int, default=None,
        help="Process only the first N candidate loci — for a cheap validation run "
             "on real data before committing to the full candidate list (the real "
             "per-locus cost went up substantially with this rebuild: up to "
             "~2*n_samples Evo 2 forward passes per locus instead of 2).",
    )
    p.add_argument(
        "--skip-borzoi", action="store_true",
        help="Skip the Borzoi baseline (requires `pip install -e '.[borzoi]'` and "
             "downloads a real pretrained checkpoint on first use) — use if that "
             "package isn't installed rather than letting every locus fail on it.",
    )
    p.add_argument(
        "--borzoi-checkpoint", default="johahi/borzoi-replicate-0",
        help="Which Borzoi replicate to use (see biolens.eval.mediation_baselines."
             "borzoi_baseline's docstring for why a single replicate, not the "
             "full ensemble).",
    )
    p.add_argument(
        "--borzoi-device", default=None,
        help="Device for the Borzoi model specifically (defaults to --device). "
             "Borzoi is a separate, large model from the Evo 2 SAE pipeline above "
             "and may need to run on a different device/be skipped independently.",
    )
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _run_positive_control_check(output_dir: Path) -> None:
    """
    Startup smoke test, run before any real locus: construct synthetic data
    with a KNOWN true mediation fraction and confirm run_ikt_mediation (the
    exact same function every real locus below calls) recovers it. This
    check is operationalized as part of the actual pipeline rather than
    left as a standalone unit test — see
    biolens.eval.mediation.run_positive_control's docstring for why: had
    this existed before, it would have caught the original M=Delta*D
    identification bug (recovered proportion_mediated ~0 regardless of true
    mediation) immediately, on the very first run, instead of surviving
    through a full real-data pass.

    Raises:
        RuntimeError: if any check's recovered proportion_mediated is more
            than _POSITIVE_CONTROL_TOLERANCE from its known true value —
            deliberately fatal, not a warning: continuing to spend real GPU-
            hours on a mediation pipeline that fails a check with a KNOWN
            correct answer would waste the entire job on results nobody
            could trust.
    """
    checks = []
    failures = []
    for true_prop in (0.0, 0.5, 1.0):
        result = run_positive_control(true_prop, n_rep=1000)
        error = abs(result.recovered_proportion_mediated - true_prop)
        checks.append({
            "true_proportion_mediated": result.true_proportion_mediated,
            "recovered_proportion_mediated": result.recovered_proportion_mediated,
            "error": error,
        })
        if error > _POSITIVE_CONTROL_TOLERANCE:
            failures.append((true_prop, result.recovered_proportion_mediated, error))
        else:
            logger.info(
                "Positive control OK: true=%.2f, recovered=%.2f (error=%.3f)",
                true_prop, result.recovered_proportion_mediated, error,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(checks, output_dir / "positive_control_check.json")

    if failures:
        detail = "; ".join(
            f"true={t:.2f} recovered={r:.2f} (error={e:.3f} > {_POSITIVE_CONTROL_TOLERANCE})"
            for t, r, e in failures
        )
        raise RuntimeError(
            f"POSITIVE CONTROL FAILED for {len(failures)}/3 checks: {detail}. The "
            f"mediation pipeline does not correctly recover a KNOWN synthetic "
            f"mediation fraction — refusing to run real loci until this is fixed. "
            f"See biolens.eval.mediation.run_positive_control's docstring."
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_positive_control_check(output_dir)

    # ── Load model + SAE ─────────────────────────────────────────────────────
    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    # build_sae_from_checkpoint infers the real architecture (expansion_factor,
    # k, variant) from the checkpoint's own saved metadata, rather than
    # constructing a fresh SAE with SAEConfig's hardcoded default
    # expansion_factor=8 — the exact bug that silently zeroed out the
    # esm2_8m x4/x16 expansion-factor sweep variants (2026-07-03/04, fixed
    # in scripts/run_eval.py first, then found here too via the same
    # `grep -rl "SAEConfig(d_model="` check).
    sae, _ = build_sae_from_checkpoint(
        args.sae_checkpoint, device=args.device, variant="topk", d_model=model_cfg.hidden_dim,
    )
    model = registry.load_model(args.model, device=args.device)

    # ── Load candidate loci + supporting data ────────────────────────────────
    fasta = load_reference_genome(args.reference_genome)
    liftover = load_liftover_chain(args.liftover_chain)
    candidates = load_published_eqtl_candidates(args.eqtl_candidates)
    if args.max_loci is not None:
        candidates = candidates[: args.max_loci]
        logger.info("--max-loci=%d: restricting to the first %d candidate loci", args.max_loci, len(candidates))
    expression = load_expression_matrix(args.expression)
    verified = load_genomic_verified_features(args.genomic_verified_features)
    population_labels = load_population_labels(args.population_labels)
    logger.info("Loaded population labels for %d samples", len(population_labels))
    with open(args.naive_feature_results) as f:
        naive_results = {r["go_id"]: r["best_feature_idx"] for r in json.load(f)}

    # Same "verified" arm feature for every locus (a documented, real gap —
    # see the module docstring: ideally locus-relevant, not a fixed one
    # across the whole candidate set — not fixed here, since doing so
    # requires real verified-feature data to test against). Computed once,
    # not once per locus: `verified` never changes across the loop below.
    # None when no feature in the registry is "confirmed" (the actual result
    # of the first real verification pass, job 9724784, 2026-07-30: 0/75) —
    # _run_one_locus handles that by running the naive arm alone rather than
    # skipping the locus (see module docstring).
    verified_feat_idx = next(
        (idx for idx, v in verified.items() if v.status == "confirmed"), None
    )
    if verified_feat_idx is None:
        logger.warning(
            "No 'confirmed' feature in the genomic verified_features registry "
            "(%s) — running the naive arm alone for every locus, with each "
            "naive feature's own verification verdict attached as metadata.",
            args.genomic_verified_features,
        )

    logger.info("Loaded %d candidate loci", len(candidates))

    # ── Genotypes for every candidate, loaded ONCE ───────────────────────────
    # load_variant_genotypes does a full linear scan of the VCF per call (see
    # its own docstring: "correct but slow for a whole-chromosome VCF... fine
    # for a candidate-locus-restricted case study" — that assumption held
    # only if called once with the WHOLE candidate list, not once per locus.
    # Real regression, job 9725435, 2026-07-30: called per-locus inside
    # _run_one_locus, this VCF-scanned from scratch for every single
    # candidate (~95s each observed on the real chr22 VCF) — with most
    # candidates not even on chr22, it burned the entire 4h SLURM budget on
    # ~150 fruitless scans without completing a single locus. Fixed by
    # loading all genotypes once here, matching the same "load once outside
    # the loop" pattern already used for the SAE/model/Borzoi model below.
    genotypes_by_position = {
        (g.chrom, g.pos): g
        for g in load_variant_genotypes(
            args.vcf, variant_positions=[(c.chrom, c.pos) for c in candidates]
        ).values()
    }
    logger.info(
        "Loaded genotypes for %d/%d candidate variants (VCF scanned once, not per-locus)",
        len(genotypes_by_position), len(candidates),
    )

    # ── Nearby-variant windows for the per-individual mediator, loaded ONCE ──
    # GRCh37 space (matches the VCF's native coordinates directly — see
    # biolens.data.geuvadis.load_window_genotypes's docstring). Symmetric
    # context_bp on each side of each candidate's own lead-variant position,
    # matching extract_ref_alt_context's window convention so the same
    # context_bp value means the same thing in both places. One linear VCF
    # scan for every candidate locus at once — the same "load once, not
    # per-locus" fix already applied to genotypes_by_position above,
    # generalized from an exact-position query to a window query.
    grch37_windows = [
        (c.variant_id, c.chrom, c.pos - 1 - args.context_bp, c.pos - 1 + args.context_bp + 1)
        for c in candidates
    ]
    window_genotypes = load_window_genotypes(args.vcf, grch37_windows)
    logger.info(
        "Loaded nearby-variant windows for %d candidate loci (VCF scanned once, not per-locus)",
        len(window_genotypes),
    )

    # ── Gene spans for the Borzoi baseline's track-to-gene aggregation ───────
    # Keyed by the eQTL file's own gene_id strings, not GENCODE's — same real
    # version-suffix mismatch already hit and fixed in
    # scripts/run_geuvadis_naive_probing.py (2026-07-25).
    gene_spans: dict[str, tuple[int, int]] = {}
    if not args.skip_borzoi:
        candidate_gene_ids = {c.gene_id for c in candidates}
        gencode_by_gene_id = load_gencode_genes_by_external_id(candidate_gene_ids)
        gene_spans = {
            gene_id: (g.start, g.end) for gene_id, g in gencode_by_gene_id.items()
        }
        logger.info(
            "Resolved GENCODE spans for %d/%d candidate genes (Borzoi baseline)",
            len(gene_spans), len(candidate_gene_ids),
        )

    borzoi_model = None
    if not args.skip_borzoi:
        logger.info("Loading Borzoi model (%s)...", args.borzoi_checkpoint)
        borzoi_model = load_borzoi_model(
            checkpoint=args.borzoi_checkpoint, device=args.borzoi_device or args.device,
        )

    all_results = []
    for candidate in candidates:
        result = _run_one_locus(
            candidate, fasta, model, sae, args, expression, verified_feat_idx, naive_results,
            gene_spans, borzoi_model, verified, genotypes_by_position, liftover,
            window_genotypes.get(candidate.variant_id, []), population_labels,
        )
        if result is not None:
            all_results.append(result)

    _apply_multiplicity_correction(all_results, verified_feat_idx)

    _save_json(all_results, output_dir / "geuvadis_case_study_results.json")
    logger.info("Completed %d/%d loci; results in %s", len(all_results), len(candidates), output_dir)


def _compute_per_individual_activations(
    model, sae, ref_sequence, window_start, variants, samples, layer, batch_size, device,
):
    """
    Build each sample's two personal haplotype sequences
    (biolens.data.geuvadis.build_haplotype_sequence), forward-pass only the
    UNIQUE sequences among them, and return each individual's own
    (haplotype-averaged) dense embedding and SAE feature activation.

    Deduplication is not a micro-optimization here: many individuals share
    an identical local haplotype in a several-kb window — especially likely
    given 1000 Genomes' population structure — so a naive 2*n_samples
    forward passes per locus does real, avoidable extra GPU work. Grouping
    identical sequences before the forward pass and scattering results back
    out is what keeps the real per-locus cost (up to ~890 sequences for
    n=445 diploid samples, vs. 2 in the original design) tractable.

    This is what gives the mediator genuine Var(M | D) > 0 — see
    biolens.data.geuvadis.build_haplotype_sequence's docstring for why that
    property, not just "uses real per-individual data," is what actually
    fixes the identification failure.

    Returns:
        (per_sample_emb, per_sample_z, n_unique_sequences, n_total_sequences)
        per_sample_emb: (n_samples, d_model) CPU float32 tensor — each
            sample's dense embedding, averaged over its two haplotypes.
        per_sample_z:   (n_samples, d_sae) CPU float32 tensor — same, after
            the SAE encoder.
    """
    import torch

    hap_sequences: list[str] = [
        build_haplotype_sequence(ref_sequence, window_start, variants, sample_id, haplotype_index)
        for sample_id in samples
        for haplotype_index in (0, 1)
    ]

    unique_seqs = sorted(set(hap_sequences))
    seq_to_row = {seq: i for i, seq in enumerate(unique_seqs)}

    with torch.no_grad():
        unique_acts = model.get_activations(
            unique_seqs, layer=layer, pooling="mean", batch_size=batch_size
        ).to(device)
        unique_z = sae.encode(unique_acts)

    unique_acts_cpu = unique_acts.cpu()
    unique_z_cpu = unique_z.cpu()

    n_samples = len(samples)
    per_sample_emb = torch.zeros(n_samples, unique_acts_cpu.shape[1])
    per_sample_z = torch.zeros(n_samples, unique_z_cpu.shape[1])
    for i, seq in enumerate(hap_sequences):
        sample_idx = i // 2
        row = seq_to_row[seq]
        per_sample_emb[sample_idx] += unique_acts_cpu[row] / 2
        per_sample_z[sample_idx] += unique_z_cpu[row] / 2

    return per_sample_emb, per_sample_z, len(unique_seqs), len(hap_sequences)


def _run_one_locus(
    candidate, fasta, model, sae, args, expression, verified_feat_idx, naive_results,
    gene_spans, borzoi_model, verified, genotypes_by_position, liftover, nearby_variants_grch37,
    population_labels,
):
    import torch

    genotype = genotypes_by_position.get((candidate.chrom, candidate.pos))
    if genotype is None:
        logger.warning("Skipping %s: variant not found in VCF", candidate.variant_id)
        return None

    try:
        treatment, outcome, samples = build_mediation_arrays(
            genotype, expression, candidate.gene_id
        )
    except (KeyError, ValueError) as exc:
        logger.warning("Skipping %s: %s", candidate.variant_id, exc)
        return None

    # Ancestry (superpopulation) covariates — population stratification is
    # the classic unmeasured confounder in genotype-expression analysis and
    # was flagged as the single biggest open item in the original design.
    # Included in every regression below via the existing covariates=
    # parameter. A sample missing a
    # population label is a real, if unexpected, data-quality issue (the
    # panel file should cover every 1000 Genomes sample) — skip the locus
    # rather than silently dropping that sample and desynchronizing it from
    # treatment/outcome, matching how every other per-locus data-quality
    # issue here is handled.
    try:
        ancestry_covariates = build_ancestry_covariates(samples, population_labels)
    except KeyError as exc:
        logger.warning("Skipping %s: %s", candidate.variant_id, exc)
        return None
    population_composition = summarize_population_composition(samples, population_labels)

    # candidate.pos (from the eQTL file) and the VCF's own positions are
    # both GRCh37 — genotype matching above is correct as-is. But the
    # reference genome FASTA (and GENCODE, and Borzoi) are GRCh38, so any
    # FASTA-based sequence extraction below must use the LIFTED position,
    # not candidate.pos directly (see load_liftover_chain's docstring for
    # the real discrepancy that caught this).
    grch38_pos = liftover_position(liftover, candidate.chrom, candidate.pos)
    if grch38_pos is None:
        logger.warning(
            "Skipping %s: no unambiguous GRCh37->GRCh38 liftOver mapping for %s:%d",
            candidate.variant_id, candidate.chrom, candidate.pos,
        )
        return None

    window_size = 2 * args.context_bp + 1
    try:
        ref_seq, _alt_seq_unused, window_start = extract_ref_alt_window(
            fasta, candidate.chrom, grch38_pos, genotype.ref, genotype.alt,
            window_size=window_size,
        )
    except ValueError as exc:
        logger.warning("Skipping %s: %s", candidate.variant_id, exc)
        return None

    # Nearby variants were queried in GRCh37 (matching the VCF directly —
    # see load_window_genotypes's docstring), but the sequence they get
    # substituted into is GRCh38 — each one needs ITS OWN liftOver, not just
    # the lead SNP's above. A variant that fails to lift is skipped, not
    # fatal to the locus (build_haplotype_sequence already silently skips
    # any variant landing outside [window_start, window_start+window_size),
    # so no separate range filter is needed here).
    lifted_variants = []
    n_lift_failed = 0
    for v in nearby_variants_grch37:
        lifted_pos = liftover_position(liftover, candidate.chrom, v.pos)
        if lifted_pos is None:
            n_lift_failed += 1
            continue
        lifted_variants.append(dataclasses.replace(v, pos=lifted_pos))
    if n_lift_failed:
        logger.debug(
            "%s: %d/%d nearby variants failed liftOver (skipped, not fatal)",
            candidate.variant_id, n_lift_failed, len(nearby_variants_grch37),
        )

    naive_feat_idx = naive_results.get(candidate.gene_id)
    if naive_feat_idx is None:
        logger.warning("Skipping %s: no naive feature for gene %s", candidate.variant_id, candidate.gene_id)
        return None

    # get_activations always returns a CPU tensor (biolens.models.evo2.
    # Evo2Model.get_activations' documented contract) — ref_acts stays CPU
    # here since it's only used below for a CPU numpy diff against
    # per_sample_emb (also CPU), not fed to the SAE.
    ref_acts = model.get_activations(ref_seq, layer=args.layer, pooling="mean")

    per_sample_emb, per_sample_z, n_unique, n_total = _compute_per_individual_activations(
        model, sae, ref_seq, window_start, lifted_variants, samples,
        layer=args.layer, batch_size=args.activation_batch_size, device=args.device,
    )
    logger.info(
        "%s: %d nearby variants (%d liftOver-failed), %d/%d haplotype sequences unique "
        "(%.0f%% forward-pass savings from dedup)",
        candidate.variant_id, len(lifted_variants), n_lift_failed, n_unique, n_total,
        100 * (1 - n_unique / n_total) if n_total else 0.0,
    )

    locus_result: dict = {
        "variant_id": candidate.variant_id, "gene_id": candidate.gene_id,
        "grch37_pos": candidate.pos, "grch38_pos": grch38_pos,
        "n_nearby_variants": len(lifted_variants),
        "n_nearby_variants_liftover_failed": n_lift_failed,
        "n_haplotype_sequences_unique": n_unique,
        "n_haplotype_sequences_total": n_total,
        # Real superpopulation composition of THIS locus's matched sample
        # set — answers the review's explicit ask to document it rather
        # than assume it (see biolens.data.geuvadis.
        # summarize_population_composition's docstring).
        "population_composition": population_composition,
    }

    # "verified" arm only runs when the registry actually has a confirmed
    # feature (verified_feat_idx is not None) — see module docstring for why
    # a missing verified arm doesn't mean skipping the locus: the naive arm
    # + its own verification verdict below is the actual finding when
    # nothing in the registry is confirmed (the real result of job 9724784,
    # 2026-07-30: 0/75 confirmed).
    arms = [("naive", naive_feat_idx)]
    if verified_feat_idx is not None:
        arms.append(("verified", verified_feat_idx))

    for arm_name, feat_idx in arms:
        # The real per-individual mediator: each sample's OWN SAE feature
        # activation on their own personal haplotype sequence — not a single
        # locus-level scalar times dosage (the identification bug this
        # rebuild fixes; see module docstring).
        mediator = per_sample_z[:, feat_idx].numpy()
        mediation = run_ikt_mediation(
            treatment, mediator, outcome, covariates=ancestry_covariates, n_rep=args.n_rep,
        )
        sensitivity = sensitivity_analysis(treatment, mediator, outcome, covariates=ancestry_covariates)
        locus_result[f"{arm_name}_feature_idx"] = feat_idx
        locus_result[f"{arm_name}_mediation"] = str(mediation)
        # ACME is only testable when the mediator has nonzero variance in
        # this sample (see MediationResult.acme_reportable's docstring): a
        # constant mediator collapses the IKT bootstrap to a point mass at
        # exactly zero, which produces a spurious p=0 "highly significant"
        # result rather than a genuine finding (caught in job 9904287,
        # 2026-08-13: 7/26 naive-arm loci had this exact signature). Reported
        # as null, not a misleading p=0, when that's not the case — the same
        # "flag, don't silently report" convention as proportion_mediated
        # below, and downstream multiplicity correction excludes these loci
        # from the p-value family entirely rather than treating them as
        # untested nulls.
        locus_result[f"{arm_name}_acme_reportable"] = mediation.acme_reportable
        locus_result[f"{arm_name}_acme_estimate"] = (
            mediation.acme_estimate if mediation.acme_reportable else None
        )
        locus_result[f"{arm_name}_acme_p"] = (
            mediation.acme_p_value if mediation.acme_reportable else None
        )
        locus_result[f"{arm_name}_ade_estimate"] = mediation.ade_estimate
        locus_result[f"{arm_name}_ade_p"] = mediation.ade_p_value
        locus_result[f"{arm_name}_total_effect"] = mediation.total_effect
        locus_result[f"{arm_name}_total_effect_p"] = mediation.total_effect_p_value
        # ACME/ADE above are the primary, always-interpretable estimates.
        # proportion_mediated is a ratio (ACME/total_effect) that's badly
        # behaved near a zero denominator — only meaningful when the total
        # effect is itself distinguishable from zero (see
        # MediationResult.proportion_mediated_reportable's
        # docstring). Reported as null, not a misleadingly precise number,
        # when that's not the case — a downstream table/plot must check this
        # flag rather than reading proportion_mediated unconditionally.
        locus_result[f"{arm_name}_proportion_mediated_reportable"] = mediation.proportion_mediated_reportable
        locus_result[f"{arm_name}_proportion_mediated"] = (
            mediation.proportion_mediated if mediation.proportion_mediated_reportable else None
        )
        locus_result[f"{arm_name}_sensitivity_rho_star"] = sensitivity.rho_star

    # Attach the naive feature's own verification verdict as metadata —
    # this is what turns a 0-confirmed-features registry into a real finding
    # instead of a gap: whatever the naive arm's mediation result above says,
    # this records whether independent genome-wide verification would have
    # rejected the feature it's built on.
    naive_verification = verified.get(naive_feat_idx)
    if naive_verification is not None:
        locus_result["naive_feature_verification_status"] = naive_verification.status
        locus_result["naive_feature_verification_reason"] = naive_verification.reason
        locus_result["naive_feature_match_rate"] = naive_verification.match_rate

    negative_control = run_negative_control(
        treatment, outcome, covariates=ancestry_covariates, n_rep=args.n_rep,
    )
    locus_result["negative_control_acme_p"] = negative_control.acme_p_value

    # Per-individual dense-embedding baseline: each sample's own distance
    # from the pure reference embedding — genuinely per-individual now,
    # instead of the old `embedding_diff_norm * treatment` (a single
    # locus-level scalar times dosage — the same degenerate M=Delta*D shape
    # as the mediator bug this rebuild fixes, which is also why it was
    # numerically redundant with the TWAS-style baseline: both were, in
    # effect, a rescaling of D). Now genuinely distinct information from
    # TWAS-style (which uses raw genotype dosage, not any sequence-model
    # output).
    # Ancestry-adjusted here too, same as the mediation/sensitivity calls
    # above — an unadjusted baseline vs. an ancestry-adjusted main result
    # would be an unfair comparison (a baseline could look artificially
    # weaker or stronger purely from the covariate asymmetry, not from the
    # actual question being compared). Borzoi's baseline below is the one
    # exception: it doesn't accept covariates at all (its internal `Y ~ X`
    # OLS formula is hardcoded) — a known, documented gap, out of scope for
    # this rebuild's required item list.
    embedding_diff = torch.linalg.norm(per_sample_emb - ref_acts, dim=1).numpy()
    raw_baseline = raw_embedding_baseline(embedding_diff, outcome, covariates=ancestry_covariates)
    twas_baseline = twas_style_baseline(treatment, outcome, covariates=ancestry_covariates)
    locus_result["raw_embedding_baseline"] = str(raw_baseline)
    locus_result["twas_style_baseline"] = str(twas_baseline)

    if borzoi_model is not None:
        gene_span = gene_spans.get(candidate.gene_id)
        if gene_span is None:
            logger.warning(
                "%s: no GENCODE span for gene %s — skipping Borzoi baseline for "
                "this locus only (other arms above are unaffected)",
                candidate.variant_id, candidate.gene_id,
            )
        else:
            try:
                borzoi_ref_seq, borzoi_alt_seq, borzoi_window_start = extract_borzoi_ref_alt_window(
                    fasta, candidate.chrom, grch38_pos, genotype.ref, genotype.alt,
                )
                borzoi_result = borzoi_baseline(
                    borzoi_ref_seq, borzoi_alt_seq, borzoi_window_start, gene_span,
                    genotype_dosage=treatment, outcome=outcome, model=borzoi_model,
                )
                locus_result["borzoi_baseline"] = str(borzoi_result)
            except ValueError as exc:
                # One locus's window-extraction/gene-overlap failure (e.g. too
                # close to a chromosome end for the 524288bp window, or the
                # gene doesn't overlap Borzoi's output window here) shouldn't
                # abort the batch — same pattern as the VCF-loading try/except
                # above, applied to this baseline specifically.
                logger.warning(
                    "%s: Borzoi baseline failed (%s) — other arms above are unaffected",
                    candidate.variant_id, exc,
                )

    return locus_result


def _apply_multiplicity_correction(all_results: list[dict], verified_feat_idx) -> None:
    """
    Benjamini-Hochberg FDR correction across every completed locus's ACME
    p-value, per arm — real gap fixed 2026-08-05: no multiplicity correction
    existed anywhere in this codebase despite testing multiple loci in the
    same run (26 for the original Geuvadis pass, more with the rebuilt
    pipeline), which any careful review of the statistics would flag
    immediately.

    Naive and verified arms are corrected as two SEPARATE families, not
    pooled into one: they test different candidate features (different
    hypotheses), not the same claim reported twice, so mixing them would
    understate each family's own effective multiple-testing burden. This
    only runs the "verified" family when verified_feat_idx is not None,
    matching the same "verified arm only exists when the registry has a
    confirmed feature" condition used everywhere else in this script — see
    the module docstring for why a missing verified arm doesn't block the
    naive arm's own correction.

    Mutates `all_results` in place: adds `f"{arm}_acme_p_bh_adjusted"`
    (float or None) and `f"{arm}_acme_significant_after_bh"` (bool) to every
    result dict, for each arm that ran. No-op if all_results is empty
    (nothing to correct — e.g. every candidate locus was skipped).

    Loci where `f"{arm}_acme_reportable"` is False (mediator had zero
    variance — see MediationResult.acme_reportable's docstring) are
    excluded from the p-value family entirely rather than counted as
    untested nulls: their p=0 is a formula artifact of a degenerate
    bootstrap distribution, not a real test, so folding it into the BH
    family would corrupt every other locus's adjusted p-value in that
    family. These loci get `_acme_p_bh_adjusted=None` and
    `_acme_significant_after_bh=False`.
    """
    if not all_results:
        return

    arms = ["naive", *(["verified"] if verified_feat_idx is not None else [])]
    for arm in arms:
        reportable = [r for r in all_results if r[f"{arm}_acme_reportable"]]
        non_reportable = [r for r in all_results if not r[f"{arm}_acme_reportable"]]
        for result in non_reportable:
            result[f"{arm}_acme_p_bh_adjusted"] = None
            result[f"{arm}_acme_significant_after_bh"] = False

        if not reportable:
            logger.info(
                "%s arm: no loci with a reportable ACME (mediator had zero "
                "variance everywhere) — skipping BH-FDR correction",
                arm,
            )
            continue

        p_values = [r[f"{arm}_acme_p"] for r in reportable]
        correction = apply_multiplicity_correction(p_values, method="fdr_bh")
        for result, adjusted_p, rejected in zip(
            reportable, correction.adjusted_p_values, correction.rejected, strict=True
        ):
            result[f"{arm}_acme_p_bh_adjusted"] = float(adjusted_p)
            result[f"{arm}_acme_significant_after_bh"] = bool(rejected)
        logger.info(
            "%s arm: BH-FDR correction across %d loci (%d excluded: "
            "non-reportable ACME) — %d/%d ACME p-values remain significant "
            "after adjustment (alpha=0.05)",
            arm, correction.n_tests, len(non_reportable),
            correction.n_significant, correction.n_tests,
        )


def _save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
