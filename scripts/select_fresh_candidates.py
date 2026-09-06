"""
Select fresh (not-yet-verified) feature candidates from a confirmatory sweep,
for expanding a verified-features registry toward a pre-specified
n_clusters=100/model target.

Loads every mp<N>/go_probing_results.json under --sweep-dir, excludes any
row whose best_feature_idx is already present in --verified-features,
dedupes by feature_idx (keeping the highest-AUROC row per feature), and
prints candidates spread across min_positives values — drawing from more
than one min_positives pool matters: a registry built from a single sweep
point has near-zero variance in the model's own predictor and produces a
degenerate confirmatory fit (see esm2_35m's round-1 incident in
configs/verified_features/esm2_35m_layer5_topk_k32.yaml).

Usage:
  python scripts/select_fresh_candidates.py \\
      --sweep-dir /scratch/arjunc4/biolens/eval/sweep_esm2_8m_L5_k32_x8 \\
      --verified-features configs/verified_features/esm2_8m_layer5_topk_k32.yaml \\
      --n-per-mp 10 \\
      --min-auroc 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.eval.feature_inspection import load_verified_annotations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", required=True, help="Directory containing mp<N>/go_probing_results.json")
    p.add_argument("--verified-features", required=True, help="Path to verified_features/*.yaml")
    p.add_argument("--n-per-mp", type=int, default=10, help="Max fresh candidates to draw per min_positives value")
    p.add_argument("--min-auroc", type=float, default=0.0, help="Only consider rows with single_feature_auroc >= this")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    verified = load_verified_annotations(args.verified_features)
    already_checked = set(verified.keys())
    print(f"Registry already has {len(already_checked)} checked feature indices.\n")

    seen_this_run: set[int] = set()
    total_fresh = 0

    for subdir in sorted(sweep_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("mp"):
            continue
        try:
            mp = int(subdir.name[2:])
        except ValueError:
            continue

        results_path = subdir / "go_probing_results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            rows = json.load(f)

        # Dedupe by feature_idx within this mp value, keeping the highest-AUROC claim.
        best_by_feature: dict[int, dict] = {}
        for row in rows:
            fi = row["best_feature_idx"]
            if fi in already_checked or fi in seen_this_run:
                continue
            if row["single_feature_auroc"] < args.min_auroc:
                continue
            existing = best_by_feature.get(fi)
            if existing is None or row["single_feature_auroc"] > existing["single_feature_auroc"]:
                best_by_feature[fi] = row

        fresh = sorted(best_by_feature.values(), key=lambda r: -r["single_feature_auroc"])[: args.n_per_mp]
        if not fresh:
            print(f"mp{mp}: no fresh candidates")
            continue

        print(f"mp{mp}: {len(fresh)} fresh candidates")
        for row in fresh:
            fi = row["best_feature_idx"]
            seen_this_run.add(fi)
            print(
                f"    feature={fi:<6} auroc={row['single_feature_auroc']:.3f}  "
                f"{row['go_id']}  {row['go_name']}"
            )
        total_fresh += len(fresh)

    all_indices = sorted(seen_this_run)
    print(f"\nTotal fresh candidates this run: {total_fresh}")
    print("Feature indices for inspect_features.py --feature-idx:")
    print(" ".join(str(i) for i in all_indices))


if __name__ == "__main__":
    main()
