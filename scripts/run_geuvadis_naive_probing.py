"""
Gene-locus-level naive-feature probing for the Geuvadis case study — produces
the "naive best-AUROC feature per locus" input
scripts/run_geuvadis_case_study.py's --naive-feature-results requires.

Structurally the gene-locus analog of run_experiment1b_evo2_genomic.py's
cCRE-class probing (biolens.eval.probing is domain-agnostic — same
probe_go_terms engine underneath), but the label set here is "which
candidate eQTL gene does this window belong to" instead of "which ENCODE
cCRE class does this window overlap" — the existing cCRE-level
genomic_probing_results.json is NOT usable as run_geuvadis_case_study.py's
--naive-feature-results input, since that script looks up naive_results by
gene_id (`r["go_id"]` for each candidate.gene_id), not by cCRE class.

For each candidate eQTL gene, tiles windows across its own gene body (+ a
promoter-proximal flank) via biolens.data.reference_genome.
generate_gene_windows, extracts Evo 2 activations for exactly those windows,
encodes through the trained SAE, and finds the single-feature-AUROC-best
feature for discriminating "window belongs to gene X" vs. "window belongs to
any other candidate gene." This is a feasibility-scale per-locus probing
pass (Geuvadis's own eQTL candidate list has already restricted the
candidate set to a manageable size, which keeps multiple-testing burden
tractable and pre-specified), not a high-power statistical claim in its
own right.

Usage:
  python scripts/run_geuvadis_naive_probing.py \\
      --model evo2_7b --layer 16 \\
      --sae-checkpoint /scratch/arjunc4/biolens/checkpoints/evo2_7b_layer16_topk/evo2_7b_L16_topk_k32/final.pt \\
      --reference-genome /scratch/arjunc4/biolens/data/reference/hg38.fa \\
      --eqtl-candidates /scratch/arjunc4/biolens/data/geuvadis/EUR373.gene.cis.FDR5.best.rs137.txt.gz \\
      --output /scratch/arjunc4/biolens/eval/geuvadis_naive_probing/naive_feature_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.data.genomic_annotations import (  # noqa: E402
    label_dna_windows_by_gene,
    load_gencode_genes_by_external_id,
)
from biolens.data.geuvadis import load_published_eqtl_candidates  # noqa: E402
from biolens.data.reference_genome import (  # noqa: E402
    extract_window_sequences,
    generate_gene_windows,
    load_reference_genome,
)
from biolens.eval.probing import probe_genomic_annotations  # noqa: E402
from biolens.models.registry import ModelRegistry  # noqa: E402
from biolens.sae.train import build_sae_from_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %s", path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="evo2_7b")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--sae-checkpoint", required=True)
    p.add_argument(
        "--expansion-factor", type=int, default=None,
        help="Only needed for checkpoints predating saved sae_cfg metadata.",
    )
    p.add_argument("--reference-genome", required=True)
    p.add_argument("--eqtl-candidates", required=True)
    p.add_argument(
        "--max-candidate-genes", type=int, default=500,
        help="Cap the number of distinct candidate genes probed (also forwarded as "
             "probe_go_terms' max_go_terms) — same tractability reasoning as every "
             "other probing driver in this project.",
    )
    p.add_argument("--window-size", type=int, default=4096)
    p.add_argument(
        "--window-stride", type=int, default=4096,
        help="Non-overlapping tiling by default (stride == window-size) — a "
             "denser stride (e.g. 512, 8x overlap between consecutive windows) "
             "was tried first and real-production-timed-out (job 9701717, "
             "2026-07-28: 66,516 heavily-redundant windows took 3.5 of the 4h "
             "budget just for the Evo 2 forward pass, leaving no time to "
             "probe). Non-overlapping windows are also the more defensible "
             "statistical design, not just faster — consecutive windows at a "
             "small stride are >85% identical sequence, inflating the visible "
             "positive count per gene without adding real information.",
    )
    p.add_argument(
        "--flank-bp", type=int, default=2000,
        help="Extend each gene's tiled region by this many bp on both sides — "
             "captures proximal promoter/regulatory context around the gene body.",
    )
    p.add_argument(
        "--min-positives", type=int, default=5,
        help="Lower than protein GO-probing's default 50 — a single gene's tiled "
             "window count is typically in the tens, not hundreds, so the same "
             "threshold used for GO terms (annotated across thousands of proteins) "
             "would exclude nearly every candidate gene.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--held-out-baselines", action="store_true",
        help="Compute held-out AUROC baseline alongside the naive max, matching "
             "run_experiment1b_evo2_genomic.py. Off by default here since this "
             "script's output is consumed as a naive-arm INPUT to the Geuvadis "
             "case study, not reported as a standalone winner's-curse result.",
    )
    p.add_argument("--held-out-fraction", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load SAE ──────────────────────────────────────────────────────────────
    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    sae, step = build_sae_from_checkpoint(
        args.sae_checkpoint, device=args.device, variant="topk",
        d_model=model_cfg.hidden_dim, expansion_factor=args.expansion_factor,
    )
    logger.info("Loaded SAE from step %d (d_sae=%d)", step, sae.cfg.d_sae)

    # ── Candidate genes + gene-body windows ──────────────────────────────────
    candidates = load_published_eqtl_candidates(args.eqtl_candidates)
    candidate_gene_ids = {c.gene_id for c in candidates}
    logger.info(
        "Loaded %d eQTL candidate loci across %d distinct genes",
        len(candidates), len(candidate_gene_ids),
    )
    if len(candidate_gene_ids) > args.max_candidate_genes:
        # Deterministic, not random — first N by sorted gene_id, same
        # "cap, don't silently subsample randomly" spirit as probe_go_terms'
        # own max_go_terms handling (sorted by positive count there; sorted
        # by ID here since there's no equivalent count to rank by yet).
        candidate_gene_ids = set(sorted(candidate_gene_ids)[: args.max_candidate_genes])
        logger.info("Capped to %d candidate genes", len(candidate_gene_ids))

    logger.info("Loading reference genome %s...", args.reference_genome)
    fasta = load_reference_genome(args.reference_genome)

    logger.info("Loading GENCODE gene annotations...")
    # Keyed by the eQTL file's own gene_id strings, not GENCODE's — the eQTL
    # file's version suffixes are from an older GENCODE/Ensembl build than
    # this script downloads (confirmed 2026-07-25: real data has
    # ENSG00000000457.8 in Geuvadis vs. ENSG00000000457.16 in current
    # GENCODE, same gene); exact-string matching silently excluded every
    # single candidate gene the first time this ran (job 9701625).
    # run_geuvadis_case_study.py looks up naive_results by candidate.gene_id
    # verbatim, so this script's output must be keyed the same way the eQTL
    # file itself spells each gene_id, not however GENCODE spells it.
    gencode_by_gene_id = load_gencode_genes_by_external_id(candidate_gene_ids)
    missing = candidate_gene_ids - set(gencode_by_gene_id)
    if missing:
        logger.warning(
            "%d/%d candidate genes not found in GENCODE (no matching stable "
            "gene ID, or a non-'gene'-type record) — excluded: %s",
            len(missing), len(candidate_gene_ids), sorted(missing)[:10],
        )

    gene_tuples = [
        (gene_id, g.chrom, g.start, g.end) for gene_id, g in gencode_by_gene_id.items()
    ]
    windows = generate_gene_windows(
        fasta, gene_tuples, window_size=args.window_size, stride=args.window_stride,
        flank_bp=args.flank_bp,
    )
    logger.info("Generated %d gene-locus windows across %d genes", len(windows), len(gencode_by_gene_id))

    window_seqs = extract_window_sequences(
        fasta, [(w[0], w[1], w[2], w[3]) for w in windows]
    )
    window_ids = [w[0] for w in windows]
    sequences = [window_seqs[wid] for wid in window_ids]
    gene_labels = label_dna_windows_by_gene(windows)

    # ── Extract Evo 2 activations + encode through the SAE ───────────────────
    logger.info("Loading %s and extracting activations for %d windows...", args.model, len(sequences))
    import torch

    model = registry.load_model(args.model, device=args.device)

    all_z_chunks = []
    BS = args.batch_size
    for i in range(0, len(sequences), BS):
        batch_seqs = sequences[i : i + BS]
        acts = model.get_activations(batch_seqs, layer=args.layer, pooling="mean")
        acts = acts.to(args.device)
        with torch.no_grad():
            z = sae.encode(acts)
        all_z_chunks.append(z.cpu())

    Z = torch.cat(all_z_chunks, dim=0)  # (N, d_sae)
    logger.info("Feature activations shape: %s", tuple(Z.shape))

    # ── Probe: which feature best discriminates "window belongs to gene X" ───
    report = probe_genomic_annotations(
        feature_acts=Z,
        window_ids=window_ids,
        dna_labels=gene_labels,
        go_names=None,
        min_positives=args.min_positives,
        max_go_terms=args.max_candidate_genes,
        model_name=args.model,
        layer=args.layer,
        sae_variant="topk",
        compute_held_out_baselines=args.held_out_baselines,
        held_out_fraction=args.held_out_fraction,
        go_dag=None,  # no DAG structure over an eQTL candidate gene list
        # This script only ever consumes best_feature_idx (from
        # single_feature_auroc) downstream — multivariate_auroc is never
        # read by run_geuvadis_case_study.py's --naive-feature-results
        # loader. Skipping it saves one expensive d_sae-dimensional
        # LogisticRegression fit per candidate gene (real cost: job 9701717
        # timed out with probing barely started, see --window-stride's
        # help text for the other half of that fix).
        compute_multivariate_auroc=False,
    )

    # NOTE: key is literally "go_id" (not e.g. "gene_id") to match
    # scripts/run_geuvadis_case_study.py's naive_results = {r["go_id"]:
    # r["best_feature_idx"] for r in json.load(f)} — go_id holds the gene_id
    # string here, same "reuse the field name, not rename it" choice
    # probe_genomic_annotations' own docstring already makes for cCRE classes.
    results_data = [
        {
            "go_id": r.go_id,
            "n_positives": r.n_positives,
            "n_total": r.n_total,
            "single_feature_auroc": r.single_feature_auroc,
            "best_feature_idx": r.best_feature_idx,
            "multivariate_auroc": r.multivariate_auroc,
            "held_out_auroc": r.held_out_auroc,
            "auroc_inflation": r.auroc_inflation,
        }
        for r in report.results
    ]
    _save_json(results_data, Path(args.output))
    logger.info(
        "Completed naive-feature probing for %d/%d candidate genes (>= %d positives each)",
        len(results_data), len(gencode_by_gene_id), args.min_positives,
    )


if __name__ == "__main__":
    main()
