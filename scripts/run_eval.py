"""
Run the standardized BioLens evaluation suite on a trained SAE.

Computes and logs:
  1. Reconstruction metrics (FVE, dead fraction, L0)
  2. GO term linear probing (single-feature and multivariate AUROC)
  3. (Phase 1) Causal ablation
  4. (Phase 2) Cross-model mediation

Usage:
  python scripts/run_eval.py \\
      --model esm2_8m --layer 5 \\
      --sae-checkpoint /path/to/checkpoint/final.pt \\
      --cache /path/to/activations \\
      --output-dir /path/to/eval_results \\
      [--wandb-project biolens]
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

from biolens.data.uniprot import (
    load_go_annotations,
    load_swissprot,
    resolve_go_dag,
    resolve_go_names,
)
from biolens.eval.probing import (
    decompose_noise_from_confounding_multi_seed,
    probe_go_terms,
)
from biolens.eval.reconstruction import compute_reconstruction_metrics
from biolens.eval.statistics import bootstrap_mean_auroc_inflation
from biolens.models.registry import ModelRegistry
from biolens.sae.dictionary import ActivationCache
from biolens.sae.train import build_sae_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="esm2_8m")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--sae-checkpoint", required=True)
    p.add_argument("--cache", required=True, help="ActivationCache directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-positives", type=int, default=50)
    p.add_argument("--max-go-terms", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--expansion-factor", type=int, default=None,
        help="Only needed for checkpoints that predate saved sae_cfg metadata "
             "(build_sae_from_checkpoint raises a clear ValueError naming this "
             "flag if one is passed without it) — e.g. the original Phase 0 "
             "esm2_8m_layer5_topk checkpoint. Ignored (harmless) for any "
             "checkpoint that already carries its own sae_cfg.",
    )
    p.add_argument(
        "--held-out-baselines", action="store_true",
        help=(
            "Compute held-out and hard-negative AUROC baselines alongside the "
            "naive max, to distinguish real signal from selection-induced "
            "inflation. Off by default — a distinct opt-in run mode for "
            "confirmatory sweep runs, not needed for a plain exit-criterion check."
        ),
    )
    p.add_argument(
        "--held-out-fraction", type=float, default=0.5,
        help="Fraction of the test split reserved for held-out estimation "
             "(only used with --held-out-baselines)",
    )
    p.add_argument(
        "--min-hard-negatives", type=int, default=10,
        help="Skip hard-negative AUROC below this many sibling-annotated "
             "negatives in the held-out set (only used with --held-out-baselines)",
    )
    p.add_argument(
        "--permuted-label-control", action="store_true",
        help=(
            "Also run the identical sweep on protein-to-label-permuted "
            "annotations and report the noise-vs-confounding decomposition. "
            "Implies --held-out-baselines (decomposition needs auroc_inflation)."
        ),
    )
    p.add_argument(
        "--n-permutations", type=int, default=20,
        help=(
            "Independent permutation draws averaged into the noise-vs-confounding "
            "decomposition (only used with --permuted-label-control). A single draw's "
            "permuted_inflation is a noisy point estimate, especially for GO terms with "
            "few positives — each extra draw reuses the same already-computed feature "
            "activations (no new GPU forward pass), just more CPU-bound re-probing."
        ),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    p.add_argument("--run-name", default=None)
    # Probing eval requires sequences — use eval subset of the cache
    p.add_argument("--max-eval-seqs", type=int, default=10000,
                   help="Max sequences for probing eval (subset of cache)")
    args = p.parse_args()
    if args.permuted_label_control:
        args.held_out_baselines = True
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model config + SAE ───────────────────────────────────────────────
    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    layer = args.layer if args.layer >= 0 else model_cfg.num_layers - 1
    d_model = model_cfg.hidden_dim

    # build_sae_from_checkpoint infers the real architecture (expansion_factor,
    # k, variant) from the checkpoint's own saved sae_cfg/sae_extra metadata —
    # NOT hardcoded defaults. The previous SAEConfig(d_model=d_model) +
    # TopKSAE(sae_cfg) construction always used SAEConfig's default
    # expansion_factor=8 regardless of what a checkpoint was actually trained
    # with, silently breaking on any non-default expansion-factor checkpoint:
    # a real RuntimeError (state_dict shape mismatch, e.g. W_enc
    # [320,1280] vs [320,2560]) hit every task of the x4/x16 expansion-factor
    # sweep variants (2026-07-03/04) — masked at the time by the same missing
    # `set -e` bug fixed elsewhere this session, so the jobs "completed"
    # while silently writing zero output. d_model/expansion_factor are passed
    # only as a fallback for checkpoints that predate saved sae_cfg metadata —
    # confirmed a real, not just hypothetical, case: the original Phase 0
    # esm2_8m_layer5_topk checkpoint has no sae_cfg key at all (trained before
    # that field existed), so --expansion-factor must be supplied explicitly
    # for that one specifically (2026-07-04) — every checkpoint trained since
    # (all of train_sae_multiscale.sh's outputs) already carries its own
    # sae_cfg and ignores this fallback entirely.
    sae, step = build_sae_from_checkpoint(
        args.sae_checkpoint, device=args.device, variant="topk",
        d_model=d_model, expansion_factor=args.expansion_factor,
    )
    logger.info("Loaded SAE from step %d (d_sae=%d)", step, sae.cfg.d_sae)

    # ── Load cache ────────────────────────────────────────────────────────────
    cache = ActivationCache(args.cache)
    logger.info("Cache: %d vectors, d=%d", len(cache), cache.d_model)

    # ── 1. Reconstruction metrics ─────────────────────────────────────────────
    logger.info("Computing reconstruction metrics...")
    recon = compute_reconstruction_metrics(
        sae=sae,
        activations=cache,
        batch_size=args.batch_size,
        device=args.device,
        normalize=True,
    )
    logger.info(str(recon))
    _save_json(
        {
            "frac_variance_explained": recon.frac_variance_explained,
            "mse": recon.mse,
            "mean_l0": recon.mean_l0,
            "mean_l0_frac": recon.mean_l0_frac,
            "dead_fraction": recon.dead_fraction,
            "n_samples": recon.n_samples,
        },
        output_dir / "reconstruction_metrics.json",
    )

    # ── 2. GO term linear probing ─────────────────────────────────────────────
    logger.info("Loading sequences and GO annotations for probing eval...")
    ids_all, seqs_all = load_swissprot(max_seqs=args.max_eval_seqs)
    protein_id_set = set(ids_all)

    go_labels, go_names = load_go_annotations(
        protein_ids=protein_id_set,
        evidence_codes={"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"},  # experimental only
    )
    logger.info("Resolving human-readable GO term names from the ontology...")
    go_names = resolve_go_names(go_names.keys())

    go_dag = None
    if args.held_out_baselines:
        logger.info("Resolving GO DAG for hard-negative sibling lookup...")
        go_dag = resolve_go_dag()

    # Extract SAE features for the eval proteins
    logger.info("Extracting SAE features for %d eval proteins...", len(ids_all))
    import torch

    model = registry.load_model(args.model, device=args.device)

    all_z_chunks = []
    eval_ids_with_acts = []
    BS = args.batch_size
    for i in range(0, len(seqs_all), BS):
        batch_seqs = seqs_all[i : i + BS]
        batch_ids = ids_all[i : i + BS]
        acts = model.get_activations(batch_seqs, layer=layer, pooling="mean")
        acts = acts.to(args.device)
        with torch.no_grad():
            z = sae.encode(acts)
        all_z_chunks.append(z.cpu())
        eval_ids_with_acts.extend(batch_ids)

    Z = torch.cat(all_z_chunks, dim=0)  # (N, d_sae)
    logger.info("Feature activations shape: %s", tuple(Z.shape))

    run_name = args.run_name or f"{args.model}_L{layer}"
    report = probe_go_terms(
        feature_acts=Z,
        protein_ids=eval_ids_with_acts,
        go_labels=go_labels,
        go_names=go_names,
        min_positives=args.min_positives,
        max_go_terms=args.max_go_terms,
        model_name=args.model,
        layer=layer,
        sae_variant="topk",
        return_per_feature_aurocs=False,
        compute_held_out_baselines=args.held_out_baselines,
        held_out_fraction=args.held_out_fraction,
        go_dag=go_dag,
        min_hard_negatives=args.min_hard_negatives,
    )

    # Check Phase 0 exit criterion
    phase0_threshold = 0.70
    passed = report.fraction_above_auroc(phase0_threshold) > 0
    logger.info(
        "Phase 0 exit criterion (any GO term AUROC > %.2f): %s",
        phase0_threshold, "PASS" if passed else "FAIL",
    )

    # Serialize report
    results_data = [
        {
            "go_id": r.go_id,
            "go_name": r.go_name,
            "n_positives": r.n_positives,
            "n_total": r.n_total,
            "single_feature_auroc": r.single_feature_auroc,
            "best_feature_idx": r.best_feature_idx,
            "multivariate_auroc": r.multivariate_auroc,
            "held_out_auroc": r.held_out_auroc,
            "hard_negative_held_out_auroc": r.hard_negative_held_out_auroc,
            "n_hard_negatives": r.n_hard_negatives,
            "auroc_inflation": r.auroc_inflation,
        }
        for r in report.results
    ]
    _save_json(results_data, output_dir / "go_probing_results.json")

    summary = {
        "n_go_terms_tested": report.n_go_terms_tested,
        "fraction_above_0.65": report.fraction_above_auroc(0.65),
        "fraction_above_0.70": report.fraction_above_auroc(0.70),
        "fraction_above_0.75": report.fraction_above_auroc(0.75),
        "fraction_above_0.80": report.fraction_above_auroc(0.80),
        "phase0_exit_criterion_passed": passed,
        "model": args.model,
        "layer": layer,
        "step": step,
        "held_out_baselines_computed": args.held_out_baselines,
    }
    if args.held_out_baselines:
        inflation_rows = [
            (r.auroc_inflation, r.best_feature_idx)
            for r in report.results if r.auroc_inflation is not None
        ]
        hard_neg_aurocs = [
            r.hard_negative_held_out_auroc for r in report.results
            if r.hard_negative_held_out_auroc is not None
        ]
        if inflation_rows:
            inflations, inflation_feat_idx = zip(*inflation_rows)
            summary["mean_auroc_inflation"] = float(sum(inflations) / len(inflations))
            # Cluster-bootstrap CI on the same feature_idx clustering variable
            # fit_verified_rate_sweep uses: a bare point estimate doesn't say
            # whether the mean is distinguishable from zero/noise at this
            # min_positives value, which matters most exactly where
            # n_go_terms_with_held_out_auroc is small (the high-min_positives
            # end of the sweep).
            ci_result = bootstrap_mean_auroc_inflation(
                np.array(inflations), np.array(inflation_feat_idx),
            )
            summary["mean_auroc_inflation_ci95"] = list(ci_result.ci)
            summary["mean_auroc_inflation_ci_n_clusters"] = ci_result.n_clusters
        else:
            summary["mean_auroc_inflation"] = None
            summary["mean_auroc_inflation_ci95"] = None
            summary["mean_auroc_inflation_ci_n_clusters"] = 0
        summary["n_go_terms_with_held_out_auroc"] = len(inflation_rows)
        summary["mean_hard_negative_held_out_auroc"] = (
            float(sum(hard_neg_aurocs) / len(hard_neg_aurocs)) if hard_neg_aurocs else None
        )
        summary["n_go_terms_with_hard_negative_auroc"] = len(hard_neg_aurocs)
    _save_json(summary, output_dir / "go_probing_summary.json")

    # ── 2b. Permuted-label control + noise-vs-confounding decomposition ────────
    if args.permuted_label_control:
        logger.info(
            "Running permuted-label control (protein-to-label correspondence "
            "shuffled, marginal label distribution preserved) across %d independent "
            "permutation draws...",
            args.n_permutations,
        )
        probe_fn = functools.partial(
            probe_go_terms,
            feature_acts=Z,
            protein_ids=eval_ids_with_acts,
            go_names=go_names,
            min_positives=args.min_positives,
            max_go_terms=args.max_go_terms,
            model_name=args.model,
            layer=layer,
            sae_variant="topk",
            compute_held_out_baselines=True,
            held_out_fraction=args.held_out_fraction,
            go_dag=go_dag,
            min_hard_negatives=args.min_hard_negatives,
        )
        decomposition = decompose_noise_from_confounding_multi_seed(
            report,
            lambda permuted_labels: probe_fn(go_labels=permuted_labels),
            go_labels, eval_ids_with_acts,
            n_permutations=args.n_permutations,
        )
        _save_json(
            [
                {
                    "go_id": d.go_id,
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
        confounding_estimates = [
            d.mean_confounding_estimate for d in decomposition
            if d.mean_confounding_estimate is not None
        ]
        permuted_inflations = [
            d.mean_permuted_inflation for d in decomposition
            if d.mean_permuted_inflation is not None
        ]
        logger.info(
            "Decomposition (%d GO terms matched, %d permutation draws each): "
            "mean real inflation=%.4f, mean permuted (pure-noise) inflation=%.4f, "
            "mean confounding estimate=%.4f",
            len(decomposition), args.n_permutations,
            float(np.mean([d.real_inflation for d in decomposition if d.real_inflation is not None]))
            if decomposition else float("nan"),
            float(np.mean(permuted_inflations)) if permuted_inflations else float("nan"),
            float(np.mean(confounding_estimates)) if confounding_estimates else float("nan"),
        )

    print("\n" + str(report))

    # ── WandB logging ─────────────────────────────────────────────────────────
    if args.wandb_project and not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=run_name + "_eval")
            wandb.log({
                "eval/frac_variance_explained": recon.frac_variance_explained,
                "eval/dead_fraction": recon.dead_fraction,
                "eval/mean_l0": recon.mean_l0,
                "eval/go_auroc_frac_above_0.70": report.fraction_above_auroc(0.70),
                "eval/go_auroc_frac_above_0.75": report.fraction_above_auroc(0.75),
                "eval/go_auroc_frac_above_0.80": report.fraction_above_auroc(0.80),
                "eval/phase0_passed": int(passed),
            })
            wandb.finish()
        except ImportError:
            pass

    logger.info("Eval results written to %s", output_dir)


def _save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
