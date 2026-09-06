"""
Genomic-annotation linear probing for a trained Evo 2 SAE: builds the
genomic-annotation probing eval (GENCODE/ENCODE cCREs) and runs the same
style of probing used for the protein SAEs, on both Evo 2 SAEs.

Structurally the DNA analog of scripts/run_eval.py's GO-term probing: same
`min_positives`/held-out/hard-negative/permuted-label-control machinery
(biolens.eval.probing is domain-agnostic — probe_genomic_annotations is a
thin wrapper around the identical probe_go_terms engine used for GO terms
and Bias-in-Bios text concepts), just with ENCODE cCRE classes as the label
set and DNA windows as the unit instead of proteins.

Does NOT read from the SAE's own training ActivationCache (which stores
whichever windows were extracted for training, sharded across many array
tasks, with no guarantee eval wants the same subset) — instead generates a
dedicated eval window set fresh via generate_tiling_windows(), extracts
Evo 2 activations for exactly those windows, and encodes them through the
trained SAE. Mirrors run_eval.py's own choice to re-extract a fresh
Swiss-Prot eval subset rather than reuse the SAE's training cache.

Usage:
  python scripts/run_experiment1b_evo2_genomic.py \\
      --model evo2_7b --layer 16 \\
      --sae-checkpoint /scratch/arjunc4/biolens/checkpoints/evo2_7b_layer16_topk/final.pt \\
      --reference-genome /scratch/arjunc4/biolens/data/reference/hg38.fa \\
      --output-dir /scratch/arjunc4/biolens/eval/evo2_7b_L16_genomic \\
      --held-out-baselines --permuted-label-control
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.data.genomic_annotations import label_dna_windows, load_encode_ccres  # noqa: E402
from biolens.data.reference_genome import (  # noqa: E402
    extract_window_sequences,
    generate_tiling_windows,
    load_reference_genome,
)
from biolens.eval.probing import (  # noqa: E402
    decompose_noise_from_confounding_multi_seed,
    probe_genomic_annotations,
)
from biolens.models.registry import ModelRegistry  # noqa: E402
from biolens.sae.train import build_sae_from_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _save_json(data: object, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %s", path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="evo2_7b")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--sae-checkpoint", required=True)
    p.add_argument(
        "--expansion-factor", type=int, default=None,
        help="Only needed for checkpoints predating saved sae_cfg metadata — see run_eval.py.",
    )
    p.add_argument("--reference-genome", required=True)
    p.add_argument(
        "--chroms",
        default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,"
        "chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX",
        help="Comma-separated chromosome list — matches extract_activations_evo2.sh's "
             "training-extraction default so the eval windows are drawn from the same "
             "genomic scope the SAE was trained on.",
    )
    p.add_argument("--window-size", type=int, default=4096)
    p.add_argument(
        "--window-stride", type=int, default=200_000,
        help="Much larger than --window-size (non-overlapping tiling would use "
             "stride==window-size and yield ~700K windows genome-wide — intractable "
             "for a single eval forward pass). A large stride subsamples evenly "
             "across the genome instead of concentrating windows in one region, "
             "which --max-eval-windows alone would do if windows were generated "
             "with the default small stride and then truncated.",
    )
    p.add_argument(
        "--max-eval-windows", type=int, default=10_000,
        help="Cap on eval windows after generation — matches run_eval.py's "
             "--max-eval-seqs default for the same tractability reason.",
    )
    p.add_argument(
        "--window-offset", type=int, default=0,
        help="Shift the tiling grid's starting position by this many bp (must "
             "satisfy 0 <= offset < --window-stride). Draws a genuinely "
             "different, non-overlapping-with-offset=0 eval window set from "
             "the same tiling scheme — for a replication check on a fresh, "
             "independently-drawn sample (e.g. re-running after a finding "
             "like the PLS confounding-estimate anomaly) "
             "without needing a different reference genome or a random draw.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-positives", type=int, default=50)
    p.add_argument("--max-cre-classes", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--held-out-baselines", action="store_true",
        help="Compute held-out AUROC baseline alongside the naive max. "
             "Hard-negative AUROC is "
             "not computed here — GO-DAG sibling structure doesn't apply to "
             "ENCODE cCRE classes, which are a flat label set, not a hierarchy.",
    )
    p.add_argument("--held-out-fraction", type=float, default=0.5)
    p.add_argument(
        "--permuted-label-control", action="store_true",
        help="Also run the identical sweep on window-to-label-permuted annotations "
             "and report the noise-vs-confounding decomposition, same as "
             "run_eval.py / run_experiment4_nonbio.py.",
    )
    p.add_argument(
        "--n-permutations", type=int, default=20,
        help="Independent permutation draws averaged into the noise-vs-confounding "
             "decomposition (only used with --permuted-label-control). Added "
             "2026-07-15 after a single fixed-seed draw produced an unreliable "
             "confounding estimate for this experiment's low-positive-count cCRE "
             "classes — see run_eval.py's flag of the same name.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    if args.permuted_label_control:
        args.held_out_baselines = True
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load SAE ──────────────────────────────────────────────────────────────
    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    d_model = model_cfg.hidden_dim

    sae, step = build_sae_from_checkpoint(
        args.sae_checkpoint, device=args.device, variant="topk",
        d_model=d_model, expansion_factor=args.expansion_factor,
    )
    logger.info("Loaded SAE from step %d (d_sae=%d)", step, sae.cfg.d_sae)

    # ── Generate eval DNA windows ────────────────────────────────────────────
    logger.info("Loading reference genome %s...", args.reference_genome)
    fasta = load_reference_genome(args.reference_genome)
    chroms = args.chroms.split(",")

    windows = generate_tiling_windows(
        fasta, window_size=args.window_size, stride=args.window_stride, chroms=chroms,
        offset=args.window_offset,
    )
    if len(windows) > args.max_eval_windows:
        # Evenly-strided subsample (not a prefix slice) so the eval set still
        # spans every requested chromosome rather than collapsing onto
        # whichever chromosome(s) happen to be enumerated first.
        step_idx = len(windows) / args.max_eval_windows
        windows = [windows[int(i * step_idx)] for i in range(args.max_eval_windows)]
    logger.info("Eval window set: %d windows across %d chromosome(s)", len(windows), len(chroms))

    window_seqs = extract_window_sequences(fasta, windows)
    window_ids = [w[0] for w in windows]
    sequences = [window_seqs[wid] for wid in window_ids]

    # ── Label windows via ENCODE cCREs ───────────────────────────────────────
    logger.info("Loading ENCODE cCRE registry...")
    ccres = load_encode_ccres(chroms=set(chroms))
    dna_labels = label_dna_windows(windows, ccres)
    n_labeled = sum(1 for labels in dna_labels.values() if labels)
    logger.info(
        "Labeled %d/%d eval windows with >=1 cCRE class overlap", n_labeled, len(window_ids)
    )

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

    # ── Probe ─────────────────────────────────────────────────────────────────
    run_name = args.run_name or f"{args.model}_L{args.layer}_genomic"
    report = probe_genomic_annotations(
        feature_acts=Z,
        window_ids=window_ids,
        dna_labels=dna_labels,
        go_names=None,  # cCRE classes are already short human-readable codes (e.g. "PLS")
        min_positives=args.min_positives,
        max_go_terms=args.max_cre_classes,
        model_name=args.model,
        layer=args.layer,
        sae_variant="topk",
        compute_held_out_baselines=args.held_out_baselines,
        held_out_fraction=args.held_out_fraction,
        go_dag=None,  # no hard-negative sibling structure for a flat cCRE label set
    )

    results_data = [
        {
            "cre_class": r.go_id,
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
    _save_json(results_data, output_dir / "genomic_probing_results.json")

    summary = {
        "n_cre_classes_tested": report.n_go_terms_tested,
        "fraction_above_0.65": report.fraction_above_auroc(0.65),
        "fraction_above_0.70": report.fraction_above_auroc(0.70),
        "fraction_above_0.75": report.fraction_above_auroc(0.75),
        "fraction_above_0.80": report.fraction_above_auroc(0.80),
        "model": args.model,
        "layer": args.layer,
        "step": step,
        "n_eval_windows": len(window_ids),
        "n_labeled_windows": n_labeled,
        "held_out_baselines_computed": args.held_out_baselines,
    }
    if args.held_out_baselines:
        inflations = [r.auroc_inflation for r in report.results if r.auroc_inflation is not None]
        summary["mean_auroc_inflation"] = (
            float(np.mean(inflations)) if inflations else None
        )
        summary["n_cre_classes_with_held_out_auroc"] = len(inflations)
    _save_json(summary, output_dir / "genomic_probing_summary.json")

    # ── Permuted-label control + noise-vs-confounding decomposition ─────────
    if args.permuted_label_control:
        logger.info(
            "Running permuted-label control (window-to-label correspondence "
            "shuffled, marginal label distribution preserved) across %d independent "
            "permutation draws...",
            args.n_permutations,
        )
        probe_fn = functools.partial(
            probe_genomic_annotations,
            feature_acts=Z,
            window_ids=window_ids,
            min_positives=args.min_positives,
            max_go_terms=args.max_cre_classes,
            model_name=args.model,
            layer=args.layer,
            sae_variant="topk",
            compute_held_out_baselines=True,
            held_out_fraction=args.held_out_fraction,
        )
        decomposition = decompose_noise_from_confounding_multi_seed(
            report,
            lambda permuted_labels: probe_fn(dna_labels=permuted_labels),
            dna_labels, window_ids,
            n_permutations=args.n_permutations,
        )
        _save_json(
            [
                {
                    "cre_class": d.go_id,
                    "real_inflation": d.real_inflation,
                    "mean_permuted_inflation": d.mean_permuted_inflation,
                    "permuted_inflation_ci95": list(d.permuted_inflation_ci)
                    if d.permuted_inflation_ci else None,
                    "mean_confounding_estimate": d.mean_confounding_estimate,
                    "confounding_estimate_ci95": list(d.confounding_estimate_ci)
                    if d.confounding_estimate_ci else None,
                    "n_seeds_successful": d.n_seeds_successful,
                }
                for d in decomposition
            ],
            output_dir / "noise_confounding_decomposition.json",
        )

    print("\n" + str(report))
    logger.info("Genomic-annotation probing results written to %s", output_dir)


if __name__ == "__main__":
    main()
