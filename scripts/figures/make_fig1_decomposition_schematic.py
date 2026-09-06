"""
Figure 1: the decomposition claim, stated visually before any real data
appears (Panel A), then grounded in the synthetic winner's-curse simulation
that isolates the vanishing mechanism in isolation (Panel B).

Panel A is a schematic, not a data plot -- it exists to state the
decomposition's actual contribution (a mixture of a vanishing and a
non-vanishing bias component) before Fig 2 shows it's real. Panel B is real
data: the winner's-curse simulation output (docs/winners_curse_results.json),
pure selection-bias simulation with no real model or labels, showing metric
choice changes whether the classic decay shape is even visible (AUROC/AP
shrink, F1 grows, MCC flat).

Figure order (this and Fig 2/3/4/5) leads with the decomposition, not with
"a known failure mode is larger than you thought".

Usage: python scripts/figures/make_fig1_decomposition_schematic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from plotstyle import (  # noqa: E402
    CATEGORICAL,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    apply_style,
    style_axis,
)

REPO_ROOT = Path(__file__).parent.parent.parent


def panel_a_schematic(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A. Two mechanisms inflate reported feature quality", loc="left", fontweight="bold")

    box_style = dict(boxstyle="round,pad=0.4", linewidth=1.2)
    text_kw = dict(ha="center", va="center", fontsize=8.2, color=INK_PRIMARY)

    top = FancyBboxPatch((0.6, 7.6), 8.8, 1.6, facecolor="#f3f1ea",
                          edgecolor=INK_SECONDARY, **box_style)
    ax.add_patch(top)
    ax.text(5, 8.4, "Reported statistic: best single-feature AUROC\n"
                     "(max over $d$ candidate SAE features)",
            fontweight="bold", **text_kw)

    left = FancyBboxPatch((0.6, 4.3), 4.0, 2.6, facecolor=CATEGORICAL["blue"],
                           alpha=0.14, edgecolor=CATEGORICAL["blue"], **box_style)
    ax.add_patch(left)
    ax.text(2.6, 6.65, "Vanishing\nsampling-noise bias", fontweight="bold",
            color=CATEGORICAL["blue"], ha="center", va="top", fontsize=8.2)
    ax.text(2.6, 5.55, "Winner's curse over $d$ noisy\nper-feature AUROCs.\n"
                       r"Shrinks as $n \to \infty$" + " (Fig. 2).",
            ha="center", va="top", fontsize=7.6, color=INK_PRIMARY)

    right = FancyBboxPatch((5.4, 4.3), 4.0, 2.6, facecolor=CATEGORICAL["red"],
                            alpha=0.14, edgecolor=CATEGORICAL["red"], **box_style)
    ax.add_patch(right)
    ax.text(7.4, 6.65, "Non-vanishing\nstructural confounding", fontweight="bold",
            color=CATEGORICAL["red"], ha="center", va="top", fontsize=8.2)
    ax.text(7.4, 5.55, "Feature tracks a real but\nimprecise broader concept.\n"
                       "Stays positive at every $n$\ntested (Fig. 3).",
            ha="center", va="top", fontsize=7.6, color=INK_PRIMARY)

    bottom = FancyBboxPatch((2.0, 1.2), 6.0, 1.4, facecolor="#f3f1ea",
                             edgecolor=INK_SECONDARY, **box_style)
    ax.add_patch(bottom)
    ax.text(5, 1.9, "Field practice (audit of 15 papers):\nneither mechanism named or corrected",
            **text_kw)

    for x0 in [2.6, 7.4]:
        ax.add_patch(FancyArrowPatch((5, 7.6), (x0, 6.9), arrowstyle="-|>",
                                      mutation_scale=12, color=INK_MUTED, linewidth=1.1))
    for x0 in [2.6, 7.4]:
        ax.add_patch(FancyArrowPatch((x0, 4.3), (5, 2.6), arrowstyle="-|>",
                                      mutation_scale=12, color=INK_MUTED, linewidth=1.1))


def panel_b_synthetic_decay(ax) -> None:
    rows = json.loads((REPO_ROOT / "docs/winners_curse_results.json").read_text())
    rows = sorted(rows, key=lambda r: r["n_positive"])
    n = np.array([r["n_positive"] for r in rows])

    series = [
        ("auroc_inflation_above_null", "AUROC", CATEGORICAL["blue"], "o"),
        ("ap_inflation_above_null", "AP", CATEGORICAL["orange"], "s"),
        ("f1_inflation_above_null", "F1", CATEGORICAL["green"], "^"),
        ("mcc_inflation_above_null", "MCC", CATEGORICAL["violet"], "D"),
    ]
    for key, label, color, marker in series:
        y = np.array([r[key] for r in rows])
        ax.plot(n, y, color=color, marker=marker, markersize=4.5, label=label,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"$n_{\mathrm{positive}}$ (candidates per group)")
    ax.set_ylabel("Inflation above null\n(max$-$mean, synthetic labels)")
    ax.set_title("B. Metric choice changes whether the decay is even visible", loc="left",
                  fontweight="bold")
    ax.legend(loc="upper right", ncol=1)
    style_axis(ax)


def main() -> None:
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.05, 1]})
    panel_a_schematic(ax_a)
    panel_b_synthetic_decay(ax_b)
    fig.suptitle(
        "Figure 1. SAE feature-evaluation inflation decomposes into a vanishing and a "
        "non-vanishing component", fontsize=9.5, y=1.03, x=0.02, ha="left",
    )
    fig.tight_layout()

    out_dir = REPO_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig1_decomposition_schematic.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {out_dir}/fig1_decomposition_schematic.{{pdf,png}}")


if __name__ == "__main__":
    main()
