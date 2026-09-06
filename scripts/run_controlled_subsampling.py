"""
Controlled subsampling variant — "pick GO terms with naturally large
positive counts, artificially downsample to 20/50/100/200 within the SAME
concept, and show the inflation pattern holds with concept difficulty held
fixed".

probe_go_terms' ordinary min_positives sweep compares across DIFFERENT
natural GO terms at each threshold — real winner's-curse evidence, but
confounded with concept difficulty (a 300-positive term isn't just "the same
concept with more data" than a 10-positive one; it may be intrinsically
easier or harder to linearly separate). This script isolates the pure
sample-size effect: pick ONE real GO term with a large natural positive
count in the eval set, and show the same naive-vs-held-out AUROC inflation
pattern holds as its VISIBLE positive count is artificially varied, holding
the concept itself fixed.

CPU-feasible, not a new GPU job: this operates entirely on an
ALREADY-EXTRACTED ActivationCache (raw ESM2 activations, from any existing
--cache directory) plus an already-trained SAE checkpoint — SAE encoding is
a small linear+TopK operation, not the expensive ESM2 forward pass, which
already happened when the cache was built. Same pattern
scripts/inspect_features.py already uses with --device cpu.

Usage:
  python scripts/run_controlled_subsampling.py \\
      --model esm2_8m --layer 5 \\
      --sae-checkpoint /scratch/arjunc4/biolens/checkpoints/esm2_8m_layer5_topk/esm2_8m_L5_topk_k32/final.pt \\
      --cache /scratch/arjunc4/biolens/data/activations/esm2_8m_layer5_mean \\
      --output docs/controlled_subsampling_esm2_8m.json \\
      --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.data.uniprot import load_go_annotations, resolve_go_dag, resolve_go_names
from biolens.eval.feature_inspection import load_cache_features
from biolens.eval.probing import probe_go_term_controlled_subsampling
from biolens.models.registry import ModelRegistry
from biolens.sae.dictionary import ActivationCache
from biolens.sae.train import build_sae_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="esm2_8m")
    p.add_argument("--sae-checkpoint", required=True)
    p.add_argument(
        "--expansion-factor", type=int, default=None,
        help="Only needed for checkpoints predating saved sae_cfg metadata — see run_eval.py.",
    )
    p.add_argument("--cache", required=True, help="ActivationCache directory")
    p.add_argument(
        "--max-seqs", type=int, default=100_000,
        help="Matches this project's standard 100K eval scale "
             "for direct comparability with the ordinary cross-term sweep.",
    )
    p.add_argument(
        "--go-id", default=None,
        help="Specific GO term to hold fixed. Default: auto-select the term with the "
             "largest natural positive count in the loaded eval set.",
    )
    p.add_argument(
        "--target-n-positives", default="20,50,100,200",
        help="Comma-separated subsample sizes to test.",
    )
    p.add_argument(
        "--n-repeats", type=int, default=20,
        help="Independent draws per target size — a mean + CI, not one noisy point.",
    )
    p.add_argument(
        "--min-hard-negatives", type=int, default=10,
        help="Same semantics as run_eval.py's flag of the same name.",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_n_positives = [int(x) for x in args.target_n_positives.split(",")]

    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    sae, step = build_sae_from_checkpoint(
        args.sae_checkpoint, device=args.device, variant="topk",
        d_model=model_cfg.hidden_dim, expansion_factor=args.expansion_factor,
    )
    logger.info("Loaded SAE from step %d (d_sae=%d)", step, sae.cfg.d_sae)

    cache = ActivationCache(args.cache)
    logger.info("Cache: %d vectors, d=%d", len(cache), cache.d_model)

    logger.info("Encoding cached activations through the SAE (max_seqs=%s)...", args.max_seqs)
    protein_ids, _sequences, Z = load_cache_features(
        cache, sae, max_seqs=args.max_seqs, device=args.device, feature_indices=None,
    )
    logger.info("Feature activations shape: %s", tuple(Z.shape))

    logger.info("Loading GO annotations for %d eval proteins...", len(protein_ids))
    protein_id_set = set(protein_ids)
    go_labels, go_names = load_go_annotations(
        protein_ids=protein_id_set,
        evidence_codes={"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"},
    )

    go_id = args.go_id
    if go_id is None:
        counts: dict[str, int] = {}
        for gos in go_labels.values():
            for g in gos:
                counts[g] = counts.get(g, 0) + 1
        if not counts:
            raise SystemExit("No GO annotations found for this eval set — cannot select a term.")
        go_id = max(counts, key=counts.get)
        logger.info("Auto-selected %s (%d natural positives)", go_id, counts[go_id])

    logger.info("Resolving human-readable name + GO DAG (for hard-negative sampling)...")
    go_name = resolve_go_names([go_id]).get(go_id, go_id)
    go_dag = resolve_go_dag()

    summaries = probe_go_term_controlled_subsampling(
        feature_acts=Z,
        protein_ids=protein_ids,
        go_labels=go_labels,
        go_id=go_id,
        go_name=go_name,
        target_n_positives=target_n_positives,
        n_repeats=args.n_repeats,
        compute_held_out_baselines=True,
        go_dag=go_dag,
        min_hard_negatives=args.min_hard_negatives,
        random_state=args.random_state,
    )

    output_data = {
        "model": args.model,
        "go_id": go_id,
        "go_name": go_name,
        "n_repeats": args.n_repeats,
        "results": [
            {
                "target_n_positives": s.target_n_positives,
                "n_natural_positives": s.n_natural_positives,
                "n_repeats_successful": s.n_repeats_successful,
                "mean_single_feature_auroc": s.mean_single_feature_auroc,
                "mean_held_out_auroc": s.mean_held_out_auroc,
                "mean_auroc_inflation": s.mean_auroc_inflation,
                "auroc_inflation_ci95": list(s.auroc_inflation_ci95) if s.auroc_inflation_ci95 else None,
            }
            for s in summaries
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output_data, indent=2))
    logger.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
