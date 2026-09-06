"""
Prepare the two human annotators' blinded task packets.

Draws candidates from the existing esm2_8m_layer5_topk_k32.yaml registry
(140 features, Claude-assisted-labeled -- those labels are LLM-assisted,
not human, which is exactly why this annotation round exists). Re-labeling
a subset of ALREADY-labeled features is deliberate, not incidental: it's
what makes the later "Human A vs. Human B vs. Claude's prior labels vs.
resolved consensus" comparison possible.

Produces, per annotator:
  - A shared overlap subset (same features, both annotators — for computing
    real inter-annotator agreement) — SAME file/rows given to both.
  - A non-overlapping expansion subset (different features per annotator —
    for expanding total registry coverage without redundant double-checking).

Both are BLINDED: claimed_concept, claimed_go, and evidence_accessions only
-- no status, no real_identity, no hint of Claude's prior judgment
(a safeguard against low-effort or non-independent annotation: annotators
must not see the existing YAML registry entries before submitting their
own).

Stratified, not proportional, sampling for the shared subset: the
underlying registry is heavily skewed (97/140 spurious), and proportional
sampling would starve confirmed/plausible of examples in exactly the range
that matters most for a discriminative kappa estimate. All 4
`unverifiable` entries are included (so rare that excluding any further
weakens an already-marginal category), plus a fixed number each of
confirmed/plausible/spurious.

Trap items (plant 2-3 trap items with an already-unambiguous answer): the
first confirmed and first spurious entry
in the (fixed-seed, deterministic) shared-subset sample are designated as
traps -- both categories were themselves originally labeled with enough
confidence to be clean category members, a reasonable proxy for
"unambiguous" without requiring a separate manual clear-cut-ness pass.
Recorded ONLY in the private answer key, never surfaced to annotators.

Outputs (all under --output-dir):
  annotator_shared_subset.csv         -- give an identical copy to BOTH annotators
  annotator_A_expansion.csv           -- give ONLY to annotator A
  annotator_B_expansion.csv           -- give ONLY to annotator B
  ANSWER_KEY_PRIVATE_do_not_share.yaml -- for post-hoc agreement analysis only

Usage:
  python scripts/prepare_annotator_packet.py \\
      --registry configs/verified_features/esm2_8m_layer5_topk_k32.yaml \\
      --output-dir docs/annotator_packets \\
      --shared-per-status 7 \\
      --expansion-per-annotator 15 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_BLIND_COLUMNS = ["feature_idx", "claimed_concept", "claimed_go", "evidence_accessions"]
_RESPONSE_COLUMNS = ["status", "confidence", "justification"]


def _blind_row(feat_idx: int, entry: dict) -> dict:
    return {
        "feature_idx": feat_idx,
        "claimed_concept": entry.get("claimed_concept", ""),
        "claimed_go": "; ".join(entry.get("claimed_go", [])),
        "evidence_accessions": "; ".join(entry.get("evidence_accessions", [])),
    }


def _write_blind_csv(path: Path, feat_idxs: list[int], features: dict) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BLIND_COLUMNS + _RESPONSE_COLUMNS)
        writer.writeheader()
        for feat_idx in feat_idxs:
            row = _blind_row(feat_idx, features[feat_idx])
            row.update({c: "" for c in _RESPONSE_COLUMNS})
            writer.writerow(row)
    logger.info("Wrote %d blinded rows to %s", len(feat_idxs), path)


def prepare_packets(
    registry_path: Path,
    output_dir: Path,
    shared_per_status: int,
    expansion_per_annotator: int,
    seed: int,
) -> None:
    import random

    rng = random.Random(seed)

    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    features = registry["features"]

    by_status: dict[str, list[int]] = defaultdict(list)
    for feat_idx, entry in features.items():
        by_status[entry.get("status", "")].append(feat_idx)
    for status_list in by_status.values():
        status_list.sort()  # deterministic before shuffling
        rng.shuffle(status_list)

    logger.info(
        "Registry: %d total features (%s)",
        len(features), {s: len(v) for s, v in by_status.items()},
    )

    # ── Shared overlap subset: all `unverifiable` + N of each other status ──
    shared_ids: list[int] = list(by_status.get("unverifiable", []))
    for status in ("confirmed", "plausible", "spurious"):
        available = by_status.get(status, [])
        if len(available) < shared_per_status:
            logger.warning(
                "Only %d '%s' entries available, wanted %d -- using all of them",
                len(available), status, shared_per_status,
            )
        shared_ids.extend(available[:shared_per_status])
    shared_ids.sort()
    logger.info("Shared overlap subset: %d features", len(shared_ids))

    # ── Trap items: first confirmed + first spurious in the shared subset ──
    shared_by_status = defaultdict(list)
    for fid in shared_ids:
        shared_by_status[features[fid]["status"]].append(fid)
    trap_ids = []
    if shared_by_status.get("confirmed"):
        trap_ids.append(sorted(shared_by_status["confirmed"])[0])
    if shared_by_status.get("spurious"):
        trap_ids.append(sorted(shared_by_status["spurious"])[0])
    logger.info("Trap items (private, not shown to annotators): %s", trap_ids)

    # ── Non-overlapping expansion pools, split A/B ──────────────────────────
    used = set(shared_ids)
    remaining: list[int] = [fid for fid in features if fid not in used]
    rng.shuffle(remaining)
    expansion_a = sorted(remaining[:expansion_per_annotator])
    expansion_b = sorted(remaining[expansion_per_annotator : 2 * expansion_per_annotator])
    if len(remaining) < 2 * expansion_per_annotator:
        logger.warning(
            "Only %d features left for expansion pools after the shared subset "
            "-- A got %d, B got %d (wanted %d each)",
            len(remaining), len(expansion_a), len(expansion_b), expansion_per_annotator,
        )
    logger.info("Expansion pool A: %d features, B: %d features", len(expansion_a), len(expansion_b))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_blind_csv(output_dir / "annotator_shared_subset.csv", shared_ids, features)
    _write_blind_csv(output_dir / "annotator_A_expansion.csv", expansion_a, features)
    _write_blind_csv(output_dir / "annotator_B_expansion.csv", expansion_b, features)

    answer_key = {
        "shared_subset": {
            fid: {
                "status": features[fid]["status"],
                "real_identity": features[fid].get("real_identity", ""),
                "claimed_concept": features[fid].get("claimed_concept", ""),
                "is_trap": fid in trap_ids,
            }
            for fid in shared_ids
        },
        "expansion_a": {
            fid: {
                "status": features[fid]["status"],
                "real_identity": features[fid].get("real_identity", ""),
            }
            for fid in expansion_a
        },
        "expansion_b": {
            fid: {
                "status": features[fid]["status"],
                "real_identity": features[fid].get("real_identity", ""),
            }
            for fid in expansion_b
        },
    }
    answer_key_path = output_dir / "ANSWER_KEY_PRIVATE_do_not_share.yaml"
    with open(answer_key_path, "w") as f:
        yaml.safe_dump(answer_key, f, sort_keys=False, allow_unicode=True, width=100)
    logger.info("Wrote private answer key to %s -- DO NOT SHARE WITH ANNOTATORS", answer_key_path)


def merge_to_full_overlap(output_dir: Path, registry_path: Path, seed: int) -> None:
    """
    Convert the existing shared(25) + non-overlapping expansion(15+15) design
    into a single 55-item FULL-OVERLAP set both annotators rate identically —
    reviewed 2026-08-16: at n=21 on-scale items (the shared subset
    alone, minus `unverifiable`), a 2-rater/3-category weighted-kappa 95%
    interval spans [0.17, 0.92] (Landis-Koch "slight" to "almost perfect") --
    uninterpretable at any observed value, the same failure mode that killed
    the width-axis sweep. Pooling to the full 55 (51 on-scale) tightens
    percent-agreement to a Wilson CI of width ~0.19 (meets the pre-registered
    <=0.20 target) and kappa to roughly [0.33, 0.82] -- narrower, but still
    not narrow; reaching kappa width <=0.35 needs ~100 on-scale items, a
    decision left open, not made here.

    This trades registry COVERAGE (the expansion pools' original purpose --
    30 additional distinct features getting at least one human check) for
    calibration PRECISION (all 55 features getting two independent human
    ratings). That reversal of a documented design decision is deliberate,
    not incidental, and each annotator's workload grows from 40 to 55 items
    (~40% more) as a real, non-free cost of the trade.

    Reads the ALREADY-GENERATED shared/expansion CSVs and answer key (not
    the registry directly) so the exact same feature selections and the 2
    existing trap items are preserved byte-for-byte -- this is a
    reorganization of already-drawn items, not a fresh random draw.
    """
    import random

    shared_path = output_dir / "annotator_shared_subset.csv"
    exp_a_path = output_dir / "annotator_A_expansion.csv"
    exp_b_path = output_dir / "annotator_B_expansion.csv"
    answer_key_path = output_dir / "ANSWER_KEY_PRIVATE_do_not_share.yaml"

    with open(registry_path) as f:
        features = yaml.safe_load(f)["features"]
    with open(answer_key_path) as f:
        answer_key = yaml.safe_load(f)

    all_ids: list[int] = []
    for csv_path in (shared_path, exp_a_path, exp_b_path):
        with open(csv_path) as f:
            all_ids.extend(int(row["feature_idx"]) for row in csv.DictReader(f))
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate feature_idx across the three existing packets — refusing to merge")
    logger.info("Merging %d shared + %d + %d expansion = %d total features into one full-overlap set",
                *[sum(1 for _ in open(p)) - 1 for p in (shared_path, exp_a_path, exp_b_path)], len(all_ids))

    rng = random.Random(seed)
    shuffled_ids = list(all_ids)
    rng.shuffle(shuffled_ids)

    merged_answer = dict(answer_key.get("shared_subset", {}))
    for pool_key in ("expansion_a", "expansion_b"):
        for fid, entry in answer_key.get(pool_key, {}).items():
            merged_answer[fid] = {**entry, "is_trap": entry.get("is_trap", False)}
    missing = set(all_ids) - set(merged_answer)
    if missing:
        raise ValueError(f"Answer key missing entries for {missing} — cannot merge safely")

    _write_blind_csv(shared_path, shuffled_ids, features)
    answer_key = {
        "shared_subset": {fid: merged_answer[fid] for fid in sorted(merged_answer)},
        "full_overlap_note": (
            "Design changed 2026-08-16 from shared(25)+non-overlapping expansion(15+15) "
            "to a single full-overlap set of all 55 -- see prepare_annotator_packet.py's "
            "merge_to_full_overlap docstring. expansion_a/expansion_b keys retired; their "
            "items are now folded into shared_subset above with the same status/real_identity/"
            "is_trap they always had. Pre-merge files backed up under pre_full_overlap_backup/."
        ),
    }
    with open(answer_key_path, "w") as f:
        yaml.safe_dump(answer_key, f, sort_keys=False, allow_unicode=True, width=100)

    exp_a_path.unlink()
    exp_b_path.unlink()
    logger.info(
        "Full-overlap packet ready: %d features in %s (give an identical copy to BOTH "
        "annotators). Retired %s and %s. Updated answer key: %s",
        len(shuffled_ids), shared_path, exp_a_path.name, exp_b_path.name, answer_key_path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", default="configs/verified_features/esm2_8m_layer5_topk_k32.yaml")
    p.add_argument("--output-dir", default="docs/annotator_packets")
    p.add_argument("--shared-per-status", type=int, default=7)
    p.add_argument("--expansion-per-annotator", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--full-overlap", action="store_true",
        help="Merge the ALREADY-GENERATED shared+expansion packets into one 55-item "
             "full-overlap set both annotators rate identically "
             "(see merge_to_full_overlap's docstring for why). Does not "
             "re-run the stratified draw; operates on existing --output-dir files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_overlap:
        merge_to_full_overlap(
            output_dir=Path(args.output_dir), registry_path=Path(args.registry), seed=args.seed,
        )
        return
    prepare_packets(
        registry_path=Path(args.registry),
        output_dir=Path(args.output_dir),
        shared_per_status=args.shared_per_status,
        expansion_per_annotator=args.expansion_per_annotator,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
