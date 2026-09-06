"""
Verify the Geuvadis naive-feature-probing results (scripts/run_geuvadis_
naive_probing.py, job 9712776's naive_feature_results.json) — builds the
first-ever Evo 2 verified_features registry, using a genomic-domain
verification protocol (biolens.eval.genomic_verification) distinct from the
ESM2/UniProt pattern (see that module's docstring for why).

For each unique best_feature_idx in the naive-probing results:
  1. Pull its real top-activating windows GENOME-WIDE from the standard
     whole-genome Evo 2 activation cache (the same cache used for cCRE
     probing) — not the artificially gene-locus-restricted window set the
     naive probing pass itself used.
  2. Look up which real gene each top window falls within, via GENCODE.
  3. Compare against every gene the naive-probing pass claimed this feature
     was the best discriminator for.

CPU-only (SAE encode + cache scan, no Evo 2 forward pass) — reuses the
already-extracted cache, same class of job as run_controlled_subsampling.py.
Run via scripts/slurm/verify_geuvadis_naive_features.sh, not directly on a
login node (see that script's header for why: a real, repeated 30-minute
login-node CPU-time cap has killed direct attempts at similarly-sized
CPU-only jobs in this project before).

Usage:
  python scripts/verify_geuvadis_naive_features.py \\
      --naive-results /scratch/arjunc4/biolens/eval/geuvadis_naive_probing/naive_feature_results.json \\
      --sae-checkpoint /scratch/arjunc4/biolens/checkpoints/evo2_7b_layer16_topk/evo2_7b_L16_topk_k32/final.pt \\
      --activation-cache /scratch/arjunc4/biolens/data/activations/evo2_7b_layer16_mean \\
      --output configs/verified_features/evo2_7b_layer16_topk_k32_geuvadis_naive.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml  # noqa: E402

from biolens.data.genomic_annotations import load_gencode_genes  # noqa: E402
from biolens.eval.feature_inspection import inspect_feature, load_cache_features  # noqa: E402
from biolens.eval.genomic_verification import verify_genomic_feature  # noqa: E402
from biolens.sae.dictionary import ActivationCache  # noqa: E402
from biolens.sae.train import build_sae_from_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_window_id(window_id: str) -> tuple[str, str, int, int] | None:
    """'chr1:12288-16384' -> ('chr1:12288-16384', 'chr1', 12288, 16384) —
    the standard whole-genome tiling cache's window_id format (biolens.data.
    reference_genome.generate_tiling_windows). Returns None if window_id
    doesn't parse (defensive — every real cache entry should parse; a
    malformed ID should be skipped, not crash the whole verification run)."""
    try:
        chrom, coords = window_id.rsplit(":", 1)
        start_s, end_s = coords.split("-")
        return window_id, chrom, int(start_s), int(end_s)
    except (ValueError, IndexError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--naive-results", required=True)
    p.add_argument("--sae-checkpoint", required=True)
    p.add_argument("--activation-cache", required=True)
    p.add_argument(
        "--top-k", type=int, default=20,
        help="Top-activating windows genome-wide to check per feature. Higher "
             "is more robust to noise but slower/more evidence to store; 20 "
             "matches the default top_k inspect_features.py otherwise uses.",
    )
    p.add_argument(
        "--max-claims-before-generic", type=int, default=5,
        help="See biolens.eval.genomic_verification.DEFAULT_MAX_CLAIMS_BEFORE_GENERIC.",
    )
    p.add_argument(
        "--confirmed-match-rate", type=float, default=0.5,
        help="See biolens.eval.genomic_verification.DEFAULT_CONFIRMED_MATCH_RATE.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True, help="Path to write the verified_features YAML registry")
    p.add_argument(
        "--evidence-output", default=None,
        help="Optional path to also write full per-window evidence as JSON "
             "(the YAML registry keeps only a compact summary — the evidence "
             "is what a human spot-check would want to see).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.naive_results) as f:
        naive_results = json.load(f)
    logger.info("Loaded %d naive-probing records from %s", len(naive_results), args.naive_results)

    claims_by_feature: dict[int, list[str]] = collections.defaultdict(list)
    for r in naive_results:
        claims_by_feature[r["best_feature_idx"]].append(r["go_id"])  # go_id holds the gene_id here
    feature_indices = sorted(claims_by_feature)
    logger.info("%d unique candidate features to verify", len(feature_indices))

    # ── SAE + whole-genome cache ─────────────────────────────────────────────
    sae, step = build_sae_from_checkpoint(args.sae_checkpoint, device=args.device, variant="topk")
    logger.info("Loaded SAE from step %d (d_sae=%d)", step, sae.cfg.d_sae)

    cache = ActivationCache(args.activation_cache)
    logger.info("Scanning whole-genome cache (%d windows)...", len(cache))
    ids, _sequences, feat_acts = load_cache_features(
        cache, sae, device=args.device, feature_indices=feature_indices,
    )
    logger.info("Loaded features for %d genome-wide windows", len(ids))

    # ── GENCODE genes: unfiltered — a top window's real gene may not be
    # among the small claimed-gene set the naive pass considered. ───────────
    logger.info("Loading full GENCODE gene set...")
    all_genes = load_gencode_genes()
    logger.info("Loaded %d GENCODE genes", len(all_genes))

    # ── Verify each candidate feature ────────────────────────────────────────
    verifications = []
    for column_idx, feat_idx in enumerate(feature_indices):
        report = inspect_feature(
            feat_idx, ids, [""] * len(ids), feat_acts, top_k=args.top_k, column_idx=column_idx,
        )
        ranked_windows = [
            parsed
            for ex in report.top_examples
            if (parsed := _parse_window_id(ex.protein_id)) is not None
        ]
        claimed_gene_ids = sorted(set(claims_by_feature[feat_idx]))

        result = verify_genomic_feature(
            feature_idx=feat_idx,
            claimed_gene_ids=claimed_gene_ids,
            ranked_windows=ranked_windows,
            genes=all_genes,
            max_claims_before_generic=args.max_claims_before_generic,
            confirmed_match_rate=args.confirmed_match_rate,
        )
        verifications.append(result)
        logger.info(
            "feature %-6d [%-12s] n_claims=%-3d match_rate=%.2f  %s",
            feat_idx, result.status, result.n_claims, result.match_rate, result.reason,
        )

    status_counts = collections.Counter(v.status for v in verifications)
    logger.info("Status tally: %s", dict(status_counts))

    # ── Write the verified_features registry (same schema convention as the
    # ESM2 registries, adapted: claimed_go -> claimed_gene_ids, real_identity
    # -> the reason string, evidence_accessions -> matched_windows). ────────
    registry = {
        "features": {
            v.feature_idx: {
                "status": v.status,
                "claimed_gene_ids": v.claimed_gene_ids,
                "n_claims": v.n_claims,
                "match_rate": round(v.match_rate, 4),
                "reason": v.reason,
                "matched_windows": v.matched_windows,
            }
            for v in verifications
        }
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(registry, sort_keys=False, default_flow_style=False))
    logger.info("Saved %s", out_path)

    if args.evidence_output:
        evidence_data = [
            {
                "feature_idx": v.feature_idx,
                "status": v.status,
                "claimed_gene_ids": v.claimed_gene_ids,
                "match_rate": v.match_rate,
                "evidence": [{"window_id": w, "overlapping_genes": g} for w, g in v.evidence],
            }
            for v in verifications
        ]
        Path(args.evidence_output).write_text(json.dumps(evidence_data, indent=2))
        logger.info("Saved evidence to %s", args.evidence_output)


if __name__ == "__main__":
    main()
