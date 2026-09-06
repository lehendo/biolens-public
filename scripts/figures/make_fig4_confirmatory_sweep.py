"""
Figure 4: confirmatory sweep (verified_rate ~ log(min_positives)), four
per-scale slopes with cluster-bootstrap CIs, and the cross-scale trend.

This figure went through a leverage check (2026-08-29): draw four
slopes with CIs -- most of which cross zero -- rather than a fitted line
through four points that reads as an established scaling law, and lead the
caption with the trend estimate rather than the heterogeneity statistic Q.
The trend panel here uses the REAL bootstrap numbers independently
reproduced on the server that day (all four / drop esm2_8m / drop
esm2_650m), not the earlier normal-approximation numbers --
only the all-four bootstrap CI excludes zero; both leave-one-out bootstrap
CIs include zero, which the caption states plainly rather than the more
flattering but less-supported "the decline survives a leverage check."

Usage: python scripts/figures/make_fig4_confirmatory_sweep.py
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

# Leave-one-out bootstrap numbers, independently reproduced on cc-login5,
# 2026-08-29 -- not stored in any committed JSON
# because they're a one-off sensitivity check, not a primary result, so
# they're hardcoded here with their provenance rather than silently
# re-derived with no record of where they came from.
LEAVE_ONE_OUT_TREND = {
    "all four scales": {"trend": -0.4690, "ci": (-0.7614, -0.1535), "p": 0.009},
    "drop esm2_8m": {"trend": -0.4465, "ci": (-0.9333, 0.0289), "p": 0.07},
    "drop esm2_650m": {"trend": -0.3647, "ci": (-0.9303, 0.2368), "p": 0.167},
}


def plot_per_scale_slopes(ax, fit: dict) -> None:
    scales = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m"]  # fixed order, largest effect first
    y_positions = np.arange(len(scales))[::-1]

    for y, scale in zip(y_positions, scales):
        slope = fit["slope"][scale]
        lo, hi = fit["slope_ci_bootstrap"][scale]
        color = SCALE_COLOR[scale]
        ax.plot([lo, hi], [y, y], "-", color=color, linewidth=1.6, solid_capstyle="round")
        ax.plot(slope, y, "o", color=color, markersize=7, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=3)
        p = fit["slope_p_value_cluster"][scale]
        ax.text(hi + 0.06, y, f"{slope:+.3f}  (p={p:.4f})" if p < 0.001 else f"{slope:+.3f}  (p={p:.2f})",
                 va="center", ha="left", fontsize=7.2, color=INK_PRIMARY)

    ax.axvline(0, color=INK_MUTED, linewidth=0.9, linestyle="-", zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([SCALE_LABEL[s] for s in scales])
    ax.set_xlabel(r"Slope of verified_rate on $\log(\mathrm{min\_positives})$")
    ax.set_xlim(-1.0, 1.65)
    ax.set_title("A. Per-scale confirmatory-sweep slopes\n(bootstrap 95% CI)", loc="left",
                 fontweight="bold")
    ax.grid(True, axis="x", alpha=1.0)
    ax.grid(False, axis="y")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.set_ylim(-0.7, len(scales) - 0.3)


def plot_trend_sensitivity(ax) -> None:
    rows = list(LEAVE_ONE_OUT_TREND.items())
    y_positions = np.arange(len(rows))[::-1]

    for y, (name, r) in zip(y_positions, rows):
        lo, hi = r["ci"]
        excludes_zero = lo > 0 or hi < 0
        color = INK_PRIMARY if excludes_zero else INK_MUTED
        ax.plot([lo, hi], [y, y], "-", color=color, linewidth=1.6, solid_capstyle="round")
        ax.plot(r["trend"], y, "s", color=color, markersize=7, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=3)
        sig_note = "excludes 0" if excludes_zero else "includes 0"
        ax.text(hi + 0.06, y, f"{r['trend']:+.3f}  (bootstrap p={r['p']:.3f}, {sig_note})",
                 va="center", ha="left", fontsize=7.2, color=INK_PRIMARY)

    ax.axvline(0, color=INK_MUTED, linewidth=0.9, linestyle="-", zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([name for name, _ in rows])
    ax.set_xlabel("Trend: slope decline per log10(params)")
    ax.set_xlim(-1.5, 2.3)
    ax.set_title("B. Cross-scale trend, leave-one-out\n(real bootstrap, not normal approx.)",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="x", alpha=1.0)
    ax.grid(False, axis="y")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.set_ylim(-0.7, len(rows) - 0.3)


def main() -> None:
    apply_style()
    fit = json.loads((REPO_ROOT / "docs/experiment3_pooled_fit_with_trend.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    plot_per_scale_slopes(ax1, fit)
    plot_trend_sensitivity(ax2)

    fig.suptitle(
        "Figure 4. The confirmatory-sweep slope is significant only at the smallest scale, and "
        "significance of its decline depends on including all four",
        fontsize=9.2, y=1.1, x=0.02, ha="left",
    )
    fig.text(
        0.02, 1.0,
        f"Joint interaction test (one common slope across all four scales): "
        f"$\\chi^2$={fit['interaction_wald_stat']:.2f}, p={fit['interaction_p_value']:.4f} — rejected.",
        fontsize=7.4, color=INK_PRIMARY, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out_dir = REPO_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig4_confirmatory_sweep.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {out_dir}/fig4_confirmatory_sweep.{{pdf,png}}")


if __name__ == "__main__":
    main()
