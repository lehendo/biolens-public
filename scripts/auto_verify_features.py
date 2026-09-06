"""
Run the LLM auto-verifier over a verified-features registry.

Two modes:
  --mode calibrate   Re-verify entries that ALREADY have a known status (from
                      human/Claude LLM-assisted labeling) and report the
                      auto-verifier's accuracy against those known labels —
                      this is the calibration step required before the
                      auto-verifier's output can be trusted for the headline
                      sweep curve.
  --mode verify       Verify entries that do NOT yet have a status (new
                      candidates), producing fresh judgments.

Every feature is verified `--n-repeats` times (default 3) to compute
run-to-run consistency alongside the judgment itself — "validated, scalable"
is an incomplete claim without knowing whether the verifier agrees with
itself.

Requires ANTHROPIC_API_KEY to be set.

Usage:
  python scripts/auto_verify_features.py \\
      --registry configs/verified_features/esm2_8m_layer5_topk_k32.yaml \\
      --mode calibrate \\
      --output docs/auto_verifier_calibration.json
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

from biolens.eval.auto_verify import consistency_rate, verify_feature  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", required=True, help="Path to a verified_features/*.yaml file")
    p.add_argument("--mode", choices=["calibrate", "verify"], default="calibrate")
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--output", default=None)
    p.add_argument(
        "--sleep-between-calls", type=float, default=0.5,
        help="Seconds to sleep between features, to be polite to the UniProt/Anthropic APIs",
    )
    p.add_argument(
        "--feature-ids", default=None,
        help="Optional comma-separated feature_idx list to restrict the run to (e.g. the "
             "annotator study's shared-overlap items) — avoids spending real Anthropic API "
             "cost re-verifying the rest of the registry when only a specific subset is needed "
             "('run auto_verify over the shared items, not the "
             "whole registry'). Omit to run over every eligible feature, as before.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.registry) as f:
        registry = yaml.safe_load(f)
    features = registry.get("features") or {}
    logger.info("Loaded %d registry entries from %s", len(features), args.registry)

    if args.feature_ids:
        wanted = {int(x.strip()) for x in args.feature_ids.split(",") if x.strip()}
        missing = wanted - set(features)
        if missing:
            raise SystemExit(f"--feature-ids includes IDs not in the registry: {sorted(missing)}")
        features = {k: v for k, v in features.items() if k in wanted}
        logger.info("Restricted to %d requested feature IDs", len(features))

    if args.mode == "calibrate":
        targets = {k: v for k, v in features.items() if v.get("status")}
    else:
        targets = {k: v for k, v in features.items() if not v.get("status")}
    logger.info("Mode=%s: %d features to process", args.mode, len(targets))

    results = []
    for feat_idx, entry in targets.items():
        accessions = entry.get("evidence_accessions") or []
        if not accessions:
            logger.warning("Feature %s has no evidence_accessions — skipping", feat_idx)
            continue

        logger.info(
            "Verifying feature %s (claimed: %s)...", feat_idx, entry.get("claimed_concept", "")
        )
        try:
            judgments = verify_feature(
                claimed_concept=entry.get("claimed_concept", ""),
                claimed_go=entry.get("claimed_go", []),
                accessions=accessions,
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
                "claimed_go": entry.get("claimed_go", []),
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
