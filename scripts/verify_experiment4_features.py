"""
Run the LLM auto-verifier over an Experiment 4 (non-biology) verified-features
registry — the text-domain analog of scripts/auto_verify_features.py.

Same two modes as the biology version:
  --mode calibrate   Re-verify entries that ALREADY have a known status (from
                      human/Claude-assisted labeling) and report the
                      auto-verifier's accuracy against those known labels.
  --mode verify       Verify entries that do NOT yet have a status (new
                      candidates), producing fresh judgments.

Every feature is verified `--n-repeats` times (default 3) for run-to-run
consistency, exactly as in the biology track.

Unlike the biology version, there is no external "fetch" step (no UniProt
equivalent for Bias-in-Bios): the registry's evidence_example_ids are looked
up directly against a freshly-loaded copy of the same Bias-in-Bios split —
`--split`/`--max-examples` MUST match whatever was used for the original
Experiment 4 extraction (scripts/run_experiment4_nonbio.py's defaults:
split=train, max_examples=20000), since biolens.data.saebench_datasets.
load_bias_in_bios assigns IDs ("biobio_{i}") by iteration order over that
exact slice — a mismatched split/cap silently looks up the wrong examples.

Requires ANTHROPIC_API_KEY to be set.

Usage:
  python scripts/verify_experiment4_features.py \\
      --registry configs/verified_features/gemma_scope_2b_layer12_experiment4.yaml \\
      --mode calibrate \\
      --output docs/experiment4_auto_verifier_calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.data.saebench_datasets import load_bias_in_bios  # noqa: E402
from biolens.eval.auto_verify import (  # noqa: E402
    TextExampleRecord,
    consistency_rate,
    verify_text_feature,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", required=True, help="Path to a verified_features/*.yaml file")
    p.add_argument("--mode", choices=["calibrate", "verify"], default="calibrate")
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument(
        "--split", default="train",
        help="Bias-in-Bios split — MUST match the original Experiment 4 extraction run "
             "(see module docstring: example IDs are assigned by iteration order over "
             "this exact split/cap, not a stable dataset-native ID)",
    )
    p.add_argument(
        "--max-examples", type=int, default=20000,
        help="MUST match the original Experiment 4 extraction run's --max-examples",
    )
    p.add_argument("--output", default=None)
    p.add_argument(
        "--sleep-between-calls", type=float, default=0.5,
        help="Seconds to sleep between features, to be polite to the Anthropic API",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.registry) as f:
        registry = yaml.safe_load(f)
    features = registry.get("features") or {}
    logger.info("Loaded %d registry entries from %s", len(features), args.registry)

    if args.mode == "calibrate":
        targets = {k: v for k, v in features.items() if v.get("status")}
    else:
        targets = {k: v for k, v in features.items() if not v.get("status")}
    logger.info("Mode=%s: %d features to process", args.mode, len(targets))

    if not targets:
        logger.info("Nothing to do — exiting without loading Bias-in-Bios.")
        return

    logger.info(
        "Loading Bias-in-Bios (%s split, max %d examples) to resolve evidence_example_ids...",
        args.split, args.max_examples,
    )
    text_ids, texts, text_labels = load_bias_in_bios(split=args.split, max_examples=args.max_examples)
    id_to_text = dict(zip(text_ids, texts))

    results = []
    for feat_idx, entry in targets.items():
        example_ids = entry.get("evidence_example_ids") or []
        if not example_ids:
            logger.warning("Feature %s has no evidence_example_ids — skipping", feat_idx)
            continue

        missing = [eid for eid in example_ids if eid not in id_to_text]
        if missing:
            logger.warning(
                "Feature %s: %d/%d evidence_example_ids not found in this "
                "Bias-in-Bios slice (split/max-examples mismatch with the original "
                "extraction run?) — skipping those: %s",
                feat_idx, len(missing), len(example_ids), missing,
            )
        records = [
            TextExampleRecord(
                example_id=eid,
                text=id_to_text[eid],
                documented_professions=sorted(text_labels.get(eid, set())),
            )
            for eid in example_ids
            if eid in id_to_text
        ]
        if not records:
            logger.error("Feature %s: no resolvable evidence_example_ids — skipping", feat_idx)
            continue

        logger.info(
            "Verifying feature %s (claimed: %s)...", feat_idx, entry.get("claimed_concept", "")
        )
        try:
            judgments = verify_text_feature(
                claimed_concept=entry.get("claimed_concept", ""),
                example_records=records,
                n_repeats=args.n_repeats,
                model=args.model,
            )
        except Exception as exc:  # noqa: BLE001 — one feature's failure shouldn't abort the batch
            logger.error("Feature %s failed to verify: %s", feat_idx, exc)
            continue

        modal_status = max(
            {j.status for j in judgments}, key=lambda s: sum(j.status == s for j in judgments)
        )
        consistency = consistency_rate(judgments)
        known_status = entry.get("status")

        results.append(
            {
                "feature_idx": feat_idx,
                "claimed_concept": entry.get("claimed_concept", ""),
                "known_status": known_status,
                "verifier_modal_status": modal_status,
                "verifier_consistency": consistency,
                "verifier_agrees_with_known": (
                    (modal_status == known_status) if known_status else None
                ),
                "judgments": [
                    {"status": j.status, "confidence": j.confidence, "reasoning": j.reasoning}
                    for j in judgments
                ],
            }
        )
        time.sleep(args.sleep_between_calls)

    if args.mode == "calibrate":
        scored = [r for r in results if r["known_status"] is not None]
        n_correct = sum(1 for r in scored if r["verifier_agrees_with_known"])
        accuracy = n_correct / len(scored) if scored else float("nan")
        mean_consistency = (
            sum(r["verifier_consistency"] for r in results) / len(results) if results else float("nan")
        )
        logger.info(
            "Calibration: %d/%d correct (accuracy=%.3f), mean run-to-run consistency=%.3f",
            n_correct, len(scored), accuracy, mean_consistency,
        )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        logger.info("Saved %d results to %s", len(results), args.output)


if __name__ == "__main__":
    main()
