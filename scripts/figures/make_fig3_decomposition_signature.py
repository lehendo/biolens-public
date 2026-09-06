"""
Figure 3: the two-mechanism signature -- non-vanishing structural
confounding (hard-negative held-out AUROC gap, docs/experiment_hard_
negative_gap.json) stays flat and positive across min_positives, while pure
sampling noise (permuted-label gap, docs/experiment_permuted_inflation_vs_n
.json) collapses over the same range, on the same axis. This is the
decomposition's actual contribution -- a distinction not covered by prior
work on this failure mode.

Points with fewer than 10 distinct clusters (this project's own established
reliability threshold) are drawn with reduced opacity and an open marker --
not hidden, but visually distinguished. mp10/mp20 duplicate points (a fixed
hard-negative-candidate cap binding before min_positives does, same
mechanism as Fig 2) are drawn once, not twice.

Usage: python scripts/figures/make_fig3_decomposition_signature.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plotstyle import (  # noqa: E402
    INK_MUTED,
    SCALE_COLOR,
    SCALE_LABEL,
    apply_style,
    style_axis,
)

REPO_ROOT = Path(__file__).parent.parent.parent
RELIABLE_CLUSTER_THRESHOLD = 10


def plot_hard_negative_gap(ax, data: dict) -> None:
    for model in ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]:
        per_mp = data[model]
        mps_sorted = sorted((int(k) for k in per_mp), key=int)
        color = SCALE_COLOR[model]
        seen_signature = None
        plotted_mp, plotted_y, plotted_lo, plotted_hi, plotted_reliable = [], [], [], [], []
        for mp in mps_sorted:
            row = per_mp[str(mp)]
            sig = (row["n_paired"], row["mean_gap"])
            if sig == seen_signature:
                continue  # byte-identical duplicate of the previous point
            seen_signature = sig
            plotted_mp.append(mp)
            plotted_y.append(row["mean_gap"])
            plotted_lo.append(row["mean_gap"] - row["ci_lo"])
            plotted_hi.append(row["ci_hi"] - row["mean_gap"])
            plotted_reliable.append(row["n_clusters"] >= RELIABLE_CLUSTER_THRESHOLD)

        plotted_mp = np.array(plotted_mp)
        plotted_y = np.array(plotted_y)
        yerr = np.array([plotted_lo, plotted_hi])
        reliable = np.array(plotted_reliable)

        ax.errorbar(plotted_mp, plotted_y, yerr=yerr, fmt="-", color=color, linewidth=1.2,
                     capsize=2, elinewidth=0.9, zorder=2)
        ax.scatter(plotted_mp[reliable], plotted_y[reliable], color=color, s=22,
                    edgecolor="white", linewidth=0.5, zorder=3, label=SCALE_LABEL[model])
        if (~reliable).any():
            ax.scatter(plotted_mp[~reliable], plotted_y[~reliable], facecolor="white",
                        edgecolor=color, linewidth=1.2, s=22, zorder=3, alpha=0.85)

    ax.axhline(0, color=INK_MUTED, linewidth=0.8, linestyle="-", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{min\_positives}$")
    ax.set_ylabel("Held-out AUROC gap\n(hard-negative siblings)")
    ax.set_title("A. Non-vanishing structural confounding", loc="left", fontweight="bold")
    ax.set_ylim(bottom=-0.02)
    style_axis(ax)


def plot_permuted_noise(ax, data: dict) -> None:
    for model in ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]:
        points = [p for p in data["per_model_points"][model] if p["duplicate_of_min_positives"] is None]
        mp = np.array([p["min_positives"] for p in points])
        y = np.array([p["mean_permuted_inflation"] for p in points])
        lo = np.array([p["mean_permuted_inflation"] - p["ci95"][0] for p in points])
        hi = np.array([p["ci95"][1] - p["mean_permuted_inflation"] for p in points])
        color = SCALE_COLOR[model]
        ax.errorbar(mp, y, yerr=[lo, hi], fmt="o-", color=color, linewidth=1.2, markersize=4.5,
                     capsize=2, elinewidth=0.9, markerfacecolor=color, markeredgecolor="white",
                     markeredgewidth=0.5, label=SCALE_LABEL[model])

    ax.axhline(0, color=INK_MUTED, linewidth=0.8, linestyle="-")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{min\_positives}$")
    ax.set_ylabel("Held-out AUROC gap\n(permuted labels)")
    ax.set_title("B. Vanishing sampling noise (same axis)", loc="left", fontweight="bold")
    ax.set_ylim(bottom=-0.02)
    style_axis(ax)


def main() -> None:
    apply_style()
    hard_neg = json.loads((REPO_ROOT / "docs/experiment_hard_negative_gap.json").read_text())
    permuted = json.loads((REPO_ROOT / "docs/experiment_permuted_inflation_vs_n.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    plot_hard_negative_gap(ax1, hard_neg)
    plot_permuted_noise(ax2, permuted)
    ax2.tick_params(axis="y", labelleft=True)  # sharey hides these by default; same scale, keep both

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.97),
               frameon=False, fontsize=7.6, columnspacing=1.2, handletextpad=0.5)
    fig.suptitle(
        "Figure 3. The two-mechanism signature: one gap stays flat and positive, "
        "the other collapses, on the same axis",
        fontsize=9.5, y=1.03, x=0.02, ha="left",
    )

    out_dir = REPO_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig3_decomposition_signature.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {out_dir}/fig3_decomposition_signature.{{pdf,png}}")


if __name__ == "__main__":
    main()
