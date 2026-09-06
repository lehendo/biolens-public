"""
Compute the full inter-annotator agreement report once both annotators have
returned their filled-in shared_subset CSVs.

Reports, per the pre-registered plan:
  - Annotator A vs. Annotator B (the real human-human baseline)
  - Claude's prior labels vs. the resolved human consensus
  - LLM auto-verifier vs. resolved consensus, if --auto-verifier-results is
    given (note: this is a DIFFERENT artifact from "Claude's prior labels"
    above -- the registry labels were produced by an earlier, separate
    labeling process, not by biolens.eval.auto_verify. Any "calibrated"
    accuracy claim about the auto-verifier is supported by this comparison,
    not the registry-labels one. Generate --auto-verifier-results via
    scripts/auto_verify_features.py --feature-ids <the shared-overlap IDs>
    --mode calibrate --output <path>, scoped to the study's shared items
    only -- do not spend API budget re-verifying the whole registry.)
Both percent agreement and weighted Cohen's kappa, plus the resolved-
consensus bookkeeping (how many items were excluded as genuinely
tied-confidence disagreements, per biolens.eval.statistics.resolve_consensus).

Also checks the 2 planted trap items (from the private answer key) and
flags outright if either annotator got one wrong -- a direct, cheap
low-effort/non-independence safeguard.

Usage:
  python scripts/compute_annotator_agreement.py \\
      --annotator-a annotator_shared_subset_A_completed.csv \\
      --annotator-b annotator_shared_subset_B_completed.csv \\
      --answer-key ANSWER_KEY_PRIVATE_do_not_share.yaml \\
      --output docs/annotator_agreement_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml  # noqa: E402

from biolens.eval.statistics import compute_agreement, resolve_consensus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_VALID_STATUSES = {"confirmed", "plausible", "spurious", "unverifiable"}
_VALID_CONFIDENCES = {"high", "medium", "low"}


def _load_completed_csv(path: Path) -> dict[int, dict]:
    """Returns feature_idx -> {"status": ..., "confidence": ..., "justification": ...},
    validating that every row was actually filled in and uses a legal value --
    a malformed or blank response must be caught explicitly, not silently
    treated as some default status."""
    rows: dict[int, dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            feat_idx = int(row["feature_idx"])
            status = row["status"].strip().lower()
            confidence = row["confidence"].strip().lower()
            if status not in _VALID_STATUSES:
                raise ValueError(
                    f"{path}, feature {feat_idx}: status {status!r} is not one of {_VALID_STATUSES}"
                )
            if confidence not in _VALID_CONFIDENCES:
                raise ValueError(
                    f"{path}, feature {feat_idx}: confidence {confidence!r} is not one of "
                    f"{_VALID_CONFIDENCES}"
                )
            if not row["justification"].strip():
                raise ValueError(f"{path}, feature {feat_idx}: empty justification")
            rows[feat_idx] = {
                "status": status, "confidence": confidence,
                "justification": row["justification"].strip(),
            }
    return rows


def build_report(
    annotator_a_path: Path, annotator_b_path: Path, answer_key_path: Path,
    auto_verifier_results_path: Path | None = None,
) -> dict:
    a_responses = _load_completed_csv(annotator_a_path)
    b_responses = _load_completed_csv(annotator_b_path)

    with open(answer_key_path) as f:
        answer_key = yaml.safe_load(f)
    shared_key = answer_key["shared_subset"]

    shared_ids = sorted(int(k) for k in shared_key)
    missing_a = [fid for fid in shared_ids if fid not in a_responses]
    missing_b = [fid for fid in shared_ids if fid not in b_responses]
    if missing_a or missing_b:
        raise ValueError(
            f"Incomplete responses -- annotator A missing {missing_a}, "
            f"annotator B missing {missing_b}. Both must fully complete the shared subset "
            f"before agreement can be computed."
        )

    # ── Trap check ────────────────────────────────────────────────────────
    trap_results = []
    for fid, entry in shared_key.items():
        if not entry.get("is_trap"):
            continue
        true_status = entry["status"]
        a_ok = a_responses[fid]["status"] == true_status
        b_ok = b_responses[fid]["status"] == true_status
        trap_results.append({
            "feature_idx": fid, "true_status": true_status,
            "annotator_a_status": a_responses[fid]["status"], "annotator_a_correct": a_ok,
            "annotator_b_status": b_responses[fid]["status"], "annotator_b_correct": b_ok,
        })
        if not a_ok:
            logger.warning("TRAP FAILED: annotator A missed trap item %d (true=%s, got=%s)",
                            fid, true_status, a_responses[fid]["status"])
        if not b_ok:
            logger.warning("TRAP FAILED: annotator B missed trap item %d (true=%s, got=%s)",
                            fid, true_status, b_responses[fid]["status"])

    # ── Human A vs. Human B ──────────────────────────────────────────────
    labels_a = [a_responses[fid]["status"] for fid in shared_ids]
    conf_a = [a_responses[fid]["confidence"] for fid in shared_ids]
    labels_b = [b_responses[fid]["status"] for fid in shared_ids]
    conf_b = [b_responses[fid]["confidence"] for fid in shared_ids]
    a_vs_b = compute_agreement(labels_a, conf_a, labels_b, conf_b)
    logger.info("Annotator A vs. Annotator B: %s", a_vs_b)

    # ── Resolved consensus per item ──────────────────────────────────────
    consensus: dict[int, dict] = {}
    for fid, la, ca, lb, cb in zip(shared_ids, labels_a, conf_a, labels_b, conf_b):
        resolved, reason = resolve_consensus(la, ca, lb, cb)
        consensus[fid] = {"resolved_label": resolved, "reason": reason}

    # ── Claude's prior labels vs. resolved human consensus ───────────────
    resolved_ids = [fid for fid in shared_ids if consensus[fid]["resolved_label"] is not None]
    excluded_ids = [fid for fid in shared_ids if consensus[fid]["resolved_label"] is None]
    claude_labels = [shared_key[fid]["status"] for fid in resolved_ids]
    consensus_labels = [consensus[fid]["resolved_label"] for fid in resolved_ids]
    # Claude's prior labels have no recorded confidence -- treat as "high"
    # (it was presented as a definitive registry entry, not hedged) purely
    # for resolve_consensus's tie-break math; this comparison itself only
    # needs percent-agreement/kappa, which compute_agreement gives regardless.
    claude_vs_consensus = compute_agreement(
        claude_labels, ["high"] * len(claude_labels),
        consensus_labels, ["high"] * len(consensus_labels),
    )
    logger.info("Claude's prior labels vs. resolved human consensus: %s", claude_vs_consensus)

    # ── LLM auto-verifier vs. resolved human consensus ────────────────────
    # This is the comparison the abstract's "calibrated" claim is actually
    # about -- biolens.eval.auto_verify is a distinct pipeline from whatever
    # process produced the registry's `status` field (see module docstring).
    auto_verifier_vs_consensus_report: dict | str
    if auto_verifier_results_path is not None:
        with open(auto_verifier_results_path) as f:
            auto_results = {r["feature_idx"]: r for r in json.load(f)}
        missing_verifier = [fid for fid in resolved_ids if fid not in auto_results]
        if missing_verifier:
            raise ValueError(
                f"--auto-verifier-results is missing judgments for resolved items "
                f"{missing_verifier} -- re-run auto_verify_features.py over the full "
                f"shared-overlap ID set before computing this comparison."
            )
        verifier_labels = [auto_results[fid]["verifier_modal_status"] for fid in resolved_ids]
        auto_vs_consensus = compute_agreement(
            verifier_labels, ["high"] * len(verifier_labels),
            consensus_labels, ["high"] * len(consensus_labels),
        )
        logger.info("LLM auto-verifier vs. resolved human consensus: %s", auto_vs_consensus)
        auto_verifier_vs_consensus_report = {
            "n_items": auto_vs_consensus.n_items,
            "percent_agreement": auto_vs_consensus.percent_agreement,
            "percent_agreement_on_scale": auto_vs_consensus.percent_agreement_on_scale,
            "weighted_kappa": auto_vs_consensus.weighted_kappa,
        }
    else:
        auto_verifier_vs_consensus_report = (
            "NOT COMPUTED -- pass --auto-verifier-results (see this script's module "
            "docstring for how to generate it via scripts/auto_verify_features.py "
            "--feature-ids). This is the comparison the abstract's 'calibrated' claim "
            "is actually about; without it, the study calibrates the registry labels, "
            "not the auto-verifier, and the abstract sentence remains unsupported."
        )

    return {
        "n_shared_items": len(shared_ids),
        "trap_results": trap_results,
        "all_traps_passed": all(t["annotator_a_correct"] and t["annotator_b_correct"] for t in trap_results),
        "annotator_a_vs_b": {
            "n_items": a_vs_b.n_items,
            "percent_agreement": a_vs_b.percent_agreement,
            "percent_agreement_on_scale": a_vs_b.percent_agreement_on_scale,
            "weighted_kappa": a_vs_b.weighted_kappa,
            "n_unverifiable_by_either": a_vs_b.n_unverifiable_by_either,
            "n_excluded_tied_disagreement": a_vs_b.n_excluded_tied_disagreement,
        },
        "resolved_consensus": {
            "n_resolved": len(resolved_ids),
            "n_excluded_as_genuinely_ambiguous": len(excluded_ids),
            "excluded_feature_ids": excluded_ids,
            "excluded_rate": len(excluded_ids) / len(shared_ids),
        },
        "claude_prior_labels_vs_resolved_consensus": {
            "n_items": claude_vs_consensus.n_items,
            "percent_agreement": claude_vs_consensus.percent_agreement,
            "percent_agreement_on_scale": claude_vs_consensus.percent_agreement_on_scale,
            "weighted_kappa": claude_vs_consensus.weighted_kappa,
        },
        "llm_auto_verifier_vs_resolved_consensus": auto_verifier_vs_consensus_report,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotator-a", required=True)
    p.add_argument("--annotator-b", required=True)
    p.add_argument("--answer-key", required=True)
    p.add_argument(
        "--auto-verifier-results", default=None,
        help="Optional path to scripts/auto_verify_features.py's --output JSON, scoped to "
             "the same shared-overlap feature IDs. Enables the auto-verifier-vs-consensus "
             "comparison the abstract's 'calibrated' claim actually needs.",
    )
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    auto_path = Path(args.auto_verifier_results) if args.auto_verifier_results else None
    report = build_report(
        Path(args.annotator_a), Path(args.annotator_b), Path(args.answer_key), auto_path,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved %s", out_path)

    if not report["all_traps_passed"]:
        logger.warning(
            "AT LEAST ONE TRAP ITEM WAS MISSED -- review annotator effort/independence "
            "before trusting the agreement numbers above."
        )


if __name__ == "__main__":
    main()
