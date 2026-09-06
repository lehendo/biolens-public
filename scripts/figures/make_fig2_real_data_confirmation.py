"""
Figure 2: real-data confirmation of the vanishing-bias mechanism, two
independent statistical routes, four ESM2 scales.

Route 1 (docs/experiment_auroc_inflation_vs_n.json): held-out AUROC gap on
real GO-labeled data. Route 2 (docs/experiment_permuted_inflation_vs_n.json):
same computation on permuted labels, isolating pure sampling noise (closer
to the formal lemma's i.i.d. assumption). Both are log(inflation) ~
log(min_positives) fits; theoretical slope -0.5. The Knapp-Hartung
random-effects pooled interval (docs/experiment_re_pooled_meta.json) is the
primary reported pooled CI -- NOT
the naive row-level pooled fit baked into each *_vs_n.json's own
"pooled_fit" key, which implicitly weights each model by how many
min_positives points it contributes rather than by estimate precision, and
is the one pooling choice that excludes -0.5 for both routes.

Usage: python scripts/figures/make_fig2_real_data_confirmation.py
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
    INK_PRIMARY,
    SCALE_COLOR,
    SCALE_LABEL,
    apply_style,
    style_axis,
)

REPO_ROOT = Path(__file__).parent.parent.parent
THEORETICAL_SLOPE = -0.5


def _dedup_points(points: list[dict]) -> list[dict]:
    """Drop points flagged as byte-identical duplicates of an earlier
    min_positives (a fixed candidate-pool cap binding before min_positives
    does at the low end -- see fit_auroc_inflation_curve.py) so the plotted
    markers don't silently double-draw the same measurement."""
    return [p for p in points if p["duplicate_of_min_positives"] is None]


def plot_route(
    ax, data: dict, title: str, kh_ci: tuple[float, float], kh_point: float, y_key: str,
) -> None:
    for model in ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]:
        points = _dedup_points(data["per_model_points"][model])
        mp = np.array([p["min_positives"] for p in points])
        y = np.array([p[y_key] for p in points])
        color = SCALE_COLOR[model]
        ax.plot(mp, y, "o", color=color, markersize=4.5, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.4, label=SCALE_LABEL[model], zorder=3)

        fit = data["per_model_fits"][model]
        x_line = np.array([mp.min(), mp.max()])
        y_line = np.exp(fit["intercept"]) * x_line ** fit["slope"]
        ax.plot(x_line, y_line, "-", color=color, linewidth=1.0, alpha=0.55, zorder=2)

    # Theoretical n^(-1/2) reference, anchored near the pooled KH point at
    # a representative min_positives so it's visually comparable, not
    # arbitrarily placed.
    x_ref = np.array([10, 300])
    anchor_mp, anchor_y = 50, 0.06
    y_ref = anchor_y * (x_ref / anchor_mp) ** THEORETICAL_SLOPE
    ax.plot(x_ref, y_ref, "--", color=INK_MUTED, linewidth=1.2, zorder=1,
            label=r"theoretical $n^{-1/2}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{min\_positives}$")
    ax.set_title(title, loc="left", fontweight="bold")
    style_axis(ax)

    kh_lo, kh_hi = kh_ci
    ax.text(
        0.03, 0.05,
        f"pooled (Knapp-Hartung): slope={kh_point:.3f}\n95% CI [{kh_lo:.3f}, {kh_hi:.3f}]",
        transform=ax.transAxes, fontsize=7, color=INK_PRIMARY, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=INK_MUTED, linewidth=0.6),
    )


def main() -> None:
    apply_style()
    route1 = json.loads((REPO_ROOT / "docs/experiment_auroc_inflation_vs_n.json").read_text())
    route2 = json.loads((REPO_ROOT / "docs/experiment_permuted_inflation_vs_n.json").read_text())
    meta = json.loads((REPO_ROOT / "docs/experiment_re_pooled_meta.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)

    r1 = meta["route1_auroc_real_label"]
    plot_route(ax1, route1, "A. Route 1 — held-out AUROC gap (real labels)",
               tuple(r1["ci_kh"]), r1["theta_re"], y_key="mean_auroc_inflation")
    ax1.set_ylabel("Mean AUROC inflation\n(naive max $-$ held-out)")

    r2 = meta["route2_permuted_label"]
    plot_route(ax2, route2, "B. Route 2 — held-out AUROC gap (permuted labels)",
               tuple(r2["ci_kh"]), r2["theta_re"], y_key="mean_permuted_inflation")

    fig.tight_layout(rect=(0, 0, 1, 0.86))

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.97),
               frameon=False, fontsize=7.6, columnspacing=1.2, handletextpad=0.5)

    fig.suptitle(
        "Figure 2. The vanishing sampling-noise bias shrinks as "
        r"$n^{-1/2}$, confirmed across four scales and two independent routes",
        fontsize=9.5, y=1.03, x=0.02, ha="left",
    )

    out_dir = REPO_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig2_real_data_confirmation.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {out_dir}/fig2_real_data_confirmation.{{pdf,png}}")


if __name__ == "__main__":
    main()
