"""
Seed configs/verified_features/gemma_scope_2b_layer12_experiment4.yaml with
real candidates from Experiment 4's completed sweep.

Unlike ESM2 (which reuses a cached ActivationCache to look up a feature's
top-activating sequences with no GPU needed -- scripts/inspect_features.py),
no such cache exists for Gemma Scope/Bias-in-Bios: this requires a fresh
Gemma-2-2B + Gemma Scope forward pass over the real Bias-in-Bios corpus to
find each candidate feature's real top-activating biography examples. Uses
biolens.models.gemma_scope.GemmaScopeSAE.get_sae_features, which already
returns ALL d_sae=16384 features' activations per text in one pass -- so
this is a single forward pass over the corpus, not one pass per candidate
feature.

Candidates are the real best_feature_idx per profession class from the
already-completed sweep, at min_positives=10 -- the sweep point with full 28-class
coverage (mp10/20/50 are identical row sets; see run_experiment4_nonbio.py's
docstring on why the low end of the sweep ties for Bias-in-Bios's fixed
class space). A feature that's the top hit for MULTIPLE professions gets
one registry entry (features are keyed by feat_idx, matching every other
registry in this project), with all claiming professions recorded --
itself a promiscuity signal worth keeping visible, the text-domain analog
of the genomic registry's "claimed by N different genes" spurious
criterion.

Does NOT call the LLM verifier -- this only seeds evidence_example_ids and
claimed_concept with status left blank. Run
scripts/verify_experiment4_features.py --mode verify afterward to get real
judgments (requires ANTHROPIC_API_KEY, real small API spend -- kept as a
separate, explicit step rather than folded in here).

Usage (run on a GPU node -- this loads Gemma-2-2B):
  python scripts/select_experiment4_candidates.py \\
      --sweep-results /scratch/arjunc4/biolens/results/experiment4_gemma_scope_L12/mp10/go_probing_results.json \\
      --registry configs/verified_features/gemma_scope_2b_layer12_experiment4.yaml \\
      --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml  # noqa: E402

from biolens.data.saebench_datasets import (  # noqa: E402
    BIAS_IN_BIOS_PROFESSIONS,
    load_bias_in_bios,
)
from biolens.models.gemma_scope import GemmaScopeSAE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _profession_name(go_id: str) -> str:
    """'profession_21' -> 'professor' (or the raw go_id if the index is out
    of range -- defensive, not expected to fire against real sweep output)."""
    idx = int(go_id.removeprefix("profession_"))
    if 0 <= idx < len(BIAS_IN_BIOS_PROFESSIONS):
        return BIAS_IN_BIOS_PROFESSIONS[idx]
    return go_id


def select_candidates(
    sweep_results_path: Path,
    split: str,
    max_examples: int,
    sae_release: str,
    sae_id: str,
    layer: int,
    top_k: int,
    device: str,
    batch_size: int,
) -> dict:
    with open(sweep_results_path) as f:
        rows = json.load(f)

    # feat_idx -> list of go_ids that selected it as their best_feature_idx
    # (a feature claimed by multiple professions gets one entry, not several
    # -- see module docstring).
    claims: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        claims[r["best_feature_idx"]].append(r["go_id"])

    logger.info(
        "%d distinct candidate features across %d profession rows (%s)",
        len(claims), len(rows), sweep_results_path,
    )

    logger.info("Loading Bias-in-Bios (%s split, max %d examples)...", split, max_examples)
    text_ids, texts, _ = load_bias_in_bios(split=split, max_examples=max_examples)

    logger.info("Loading Gemma-2-2B + Gemma Scope SAE (%s / %s, layer %d)...", sae_release, sae_id, layer)
    sae = GemmaScopeSAE(sae_release=sae_release, sae_id=sae_id, layer=layer, device=device)

    logger.info("Running forward pass over %d texts (this is the slow step)...", len(texts))
    features = sae.get_sae_features(texts, batch_size=batch_size)  # (N, d_sae)
    logger.info("Done: features shape %s", tuple(features.shape))

    new_entries: dict[int, dict] = {}
    for feat_idx, go_ids in claims.items():
        col = features[:, feat_idx]
        top_indices = col.argsort(descending=True)[:top_k].tolist()
        evidence_example_ids = [text_ids[i] for i in top_indices]

        professions = sorted({_profession_name(gid) for gid in go_ids})
        claimed_concept = professions[0] if len(professions) == 1 else (
            f"{professions[0]} (feature ALSO claimed by: {', '.join(professions[1:])})"
        )

        new_entries[feat_idx] = {
            "claimed_concept": claimed_concept,
            "claimed_go": sorted(set(go_ids)),
            "evidence_example_ids": evidence_example_ids,
        }
        logger.info(
            "Feature %d: claimed_concept=%r, %d claiming profession(s), top activation=%.3f",
            feat_idx, claimed_concept, len(go_ids), float(col[top_indices[0]]),
        )

    return new_entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-results", required=True)
    p.add_argument("--registry", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--max-examples", type=int, default=20000)
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res")
    p.add_argument("--sae-id", default="layer_12/width_16k/average_l0_82")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--batch-size", type=int, default=128,
        # Matches run_experiment4_nonbio.py's own tuned default -- the
        # original 16 was confirmed (2026-07) to leave most of the GPU idle
        # on this exact model/hardware combination.
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    new_entries = select_candidates(
        sweep_results_path=Path(args.sweep_results),
        split=args.split,
        max_examples=args.max_examples,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        layer=args.layer,
        top_k=args.top_k,
        device=args.device,
        batch_size=args.batch_size,
    )

    registry_path = Path(args.registry)
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    registry.setdefault("features", {})

    already_present = set(registry["features"]) & set(new_entries)
    if already_present:
        logger.warning(
            "%d features already in the registry (possibly already verified) — "
            "NOT overwriting: %s", len(already_present), sorted(already_present),
        )
    for feat_idx, entry in new_entries.items():
        if feat_idx in already_present:
            continue
        registry["features"][feat_idx] = entry

    with open(registry_path, "w") as f:
        yaml.safe_dump(registry, f, sort_keys=False, allow_unicode=True, width=100)

    logger.info(
        "Seeded %d new candidates into %s (%d total now in registry)",
        len(new_entries) - len(already_present), registry_path, len(registry["features"]),
    )


if __name__ == "__main__":
    main()
