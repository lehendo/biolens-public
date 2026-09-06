"""
Experiment 4 — non-biology domain replication.

Runs the identical min_positives sweep methodology used for GO-term and
ENCODE cCRE probing, against a public, pretrained, non-biological SAE
(Gemma Scope on Gemma-2-2B) and a public, non-biological concept-labeled
dataset (Bias-in-Bios profession classification) — demonstrating the
selection-bias artifact isn't biology-specific or TopK-architecture-specific
(Gemma Scope uses JumpReLU rather than TopK, so a matching pattern here
broadens the claim beyond a single SAE architecture).

Feature extraction (loading Gemma-2-2B + the Gemma Scope SAE, encoding every
text example) happens ONCE per invocation, then every --min-positives-list
value is swept over the SAME extracted features — min_positives is purely a
downstream filtering threshold on probe_text_concepts, not something that
changes what needs extracting. Looping the extraction itself per sweep point
(as a SLURM array job matching run_sweep_eval.sh's pattern would, if applied
here unmodified) would redundantly reload the 2B-parameter model and re-run
the full forward pass over every text example once per sweep point for no
benefit.

Output layout mirrors scripts/run_eval.py exactly (go_probing_results.json,
go_probing_summary.json, noise_confounding_decomposition.json under each
mp<N>/ subdirectory) so scripts/fit_sweep.py's existing aggregation works
unmodified against Experiment 4's output.

Usage:
  python scripts/run_experiment4_nonbio.py \\
      --sae-release gemma-scope-2b-pt-res \\
      --sae-id layer_12/width_16k/average_l0_82 \\
      --layer 12 \\
      --min-positives-list 10,20,50,100,150,200,300 \\
      --held-out-baselines \\
      --permuted-label-control \\
      --output-dir docs/experiment4_results
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

from biolens.data.saebench_datasets import load_bias_in_bios  # noqa: E402
from biolens.eval.probing import (  # noqa: E402
    decompose_noise_from_confounding_multi_seed,
    probe_text_concepts,
)
from biolens.eval.statistics import bootstrap_mean_auroc_inflation  # noqa: E402
from biolens.models.gemma_scope import GemmaScopeSAE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _save_json(data: object, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %s", path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="gemma-2-2b")
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res")
    p.add_argument(
        "--sae-id", default="layer_12/width_16k/average_l0_82",
        help="Gemma Scope sae_id — see sae_lens's pretrained-SAE directory for valid values "
             "per release (e.g. 'layer_N/width_16k/average_l0_M')",
    )
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--split", default="train")
    p.add_argument(
        "--max-examples", type=int, default=20000,
        help="Cap on Bias-in-Bios examples (full train split is 257K — a large "
             "cap keeps this tractable while still giving ample per-class counts)",
    )
    p.add_argument(
        "--min-positives-list", default="10,20,50,100,150,200,300",
        help="Comma-separated min_positives sweep points — matches "
             "configs/eval.yaml::probing.sweep_min_positives, the same sweep "
             "points used for the ESM2/Evo2 tracks, so results are directly "
             "comparable across domains.",
    )
    p.add_argument("--max-go-terms", type=int, default=28)  # all 28 profession classes
    p.add_argument(
        "--batch-size", type=int, default=128,
        help="Bumped from an earlier default of 16 after job 9322709 TIMEOUT'd at the full "
             "4h cap even on a non-MIG-sliced H100 (2026-07-04) — 16 is far too small for a "
             "2B-parameter model on an 80GB H100 at max_length=128, leaving most of the GPU "
             "idle. 128 is a reasonable middle ground; raise further if this still runs long.",
    )
    p.add_argument("--held-out-baselines", action="store_true")
    p.add_argument("--held-out-fraction", type=float, default=0.5)
    p.add_argument("--permuted-label-control", action="store_true")
    p.add_argument(
        "--n-permutations", type=int, default=20,
        help="Independent permutation draws averaged into the noise-vs-confounding "
             "decomposition (only used with --permuted-label-control) — see "
             "run_eval.py's flag of the same name.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _report_to_results_data(report: object) -> list[dict]:
    return [
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
        for r in report.results  # type: ignore[attr-defined]
    ]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    min_positives_list = [int(x) for x in args.min_positives_list.split(",")]

    logger.info("Loading Bias-in-Bios (%s split, max %d examples)...", args.split, args.max_examples)
    text_ids, texts, text_labels = load_bias_in_bios(split=args.split, max_examples=args.max_examples)

    logger.info(
        "Loading Gemma Scope SAE (%s / %s) on %s, layer %d...",
        args.sae_release, args.sae_id, args.model_name, args.layer,
    )
    gemma_sae = GemmaScopeSAE(
        model_name=args.model_name, sae_release=args.sae_release,
        sae_id=args.sae_id, layer=args.layer, device=args.device,
    )

    logger.info("Extracting SAE features for %d examples (once, reused across all sweep points)...", len(texts))
    features = gemma_sae.get_sae_features(texts, batch_size=args.batch_size)
    logger.info("Feature shape: %s", tuple(features.shape))

    for mp in min_positives_list:
        output_dir = output_root / f"mp{mp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("── min_positives=%d ──", mp)

        report = probe_text_concepts(
            feature_acts=features,
            text_ids=text_ids,
            text_labels=text_labels,
            min_positives=mp,
            max_go_terms=args.max_go_terms,
            model_name=f"{args.model_name}+{args.sae_release}",
            layer=args.layer,
            sae_variant="jumprelu",
            compute_held_out_baselines=args.held_out_baselines,
            held_out_fraction=args.held_out_fraction,
        )
        _save_json(_report_to_results_data(report), output_dir / "go_probing_results.json")

        summary = {
            "n_go_terms_tested": report.n_go_terms_tested,
            "fraction_above_0.65": report.fraction_above_auroc(0.65),
            "fraction_above_0.70": report.fraction_above_auroc(0.70),
            "fraction_above_0.75": report.fraction_above_auroc(0.75),
            "fraction_above_0.80": report.fraction_above_auroc(0.80),
            "model": f"{args.model_name}+{args.sae_release}",
            "layer": args.layer,
            "min_positives": mp,
            "held_out_baselines_computed": args.held_out_baselines,
        }
        if args.held_out_baselines:
            inflation_rows = [
                (r.auroc_inflation, r.best_feature_idx)
                for r in report.results if r.auroc_inflation is not None
            ]
            if inflation_rows:
                inflations, inflation_feat_idx = zip(*inflation_rows)
                summary["mean_auroc_inflation"] = float(np.mean(inflations))
                # Cluster-bootstrap CI — answers "is this decay real or noise"
                # instead of leaving a bare point estimate, which matters most
                # at high min_positives where only a handful of profession
                # classes remain eligible.
                ci_result = bootstrap_mean_auroc_inflation(
                    np.array(inflations), np.array(inflation_feat_idx),
                )
                summary["mean_auroc_inflation_ci95"] = list(ci_result.ci)
                summary["mean_auroc_inflation_ci_n_clusters"] = ci_result.n_clusters
            else:
                summary["mean_auroc_inflation"] = None
                summary["mean_auroc_inflation_ci95"] = None
                summary["mean_auroc_inflation_ci_n_clusters"] = 0
            summary["n_profession_classes_with_held_out_auroc"] = len(inflation_rows)
        _save_json(summary, output_dir / "go_probing_summary.json")

        if args.permuted_label_control:
            probe_fn = functools.partial(
                probe_text_concepts,
                feature_acts=features, text_ids=text_ids,
                min_positives=mp, max_go_terms=args.max_go_terms,
                model_name=f"{args.model_name}+{args.sae_release}", layer=args.layer,
                sae_variant="jumprelu", compute_held_out_baselines=True,
                held_out_fraction=args.held_out_fraction,
            )
            decomposition = decompose_noise_from_confounding_multi_seed(
                report,
                lambda permuted_labels: probe_fn(text_labels=permuted_labels),
                text_labels, text_ids,
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

        print("\n" + str(report))

    logger.info("Experiment 4 sweep complete: %s", output_root)


if __name__ == "__main__":
    main()
