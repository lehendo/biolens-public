"""
Shared matplotlib style for this project's figures.

Palette and mark-spec choices follow this project's dataviz skill (fixed-
order colorblind-safe categorical hues, solid hairline gridlines, thin
marks, no dual-axis, selective direct labels over legends-only). The
categorical palette here is the skill's own validated default
(references/palette.md) rather than an eyeballed set -- see that file for
the CVD/contrast validation this ordering already passed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Fixed-order categorical palette (light-surface variant -- these are static
# print/PDF figures, not a themed UI, so only one mode is needed).
CATEGORICAL = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
CATEGORICAL_ORDER = ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]

# The four ESM2 scales get fixed slots throughout every figure -- an entity
# keeps its color across figures/panels, never reassigned by local sort order.
SCALE_COLOR = {
    "esm2_8m": CATEGORICAL["blue"],
    "esm2_35m": CATEGORICAL["orange"],
    "esm2_150m": CATEGORICAL["aqua"],
    "esm2_650m": CATEGORICAL["yellow"],
}
SCALE_LABEL = {
    "esm2_8m": "ESM2-8M",
    "esm2_35m": "ESM2-35M",
    "esm2_150m": "ESM2-150M",
    "esm2_650m": "ESM2-650M",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
# Verification-status colors for the Geuvadis/registry figures -- reserved,
# never reused for a plain series.
VERIFICATION_COLOR = {
    "confirmed": STATUS["good"],
    "plausible": CATEGORICAL["yellow"],
    "spurious": STATUS["critical"],
    "unverifiable": INK_MUTED,
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.grid": True,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # embed as real text, not outlined paths
        "ps.fonttype": 42,
    })


def style_axis(ax) -> None:
    """Strip top/right spines, keep hairline bottom/left -- standard
    scientific-figure cleanup, applied uniformly instead of per-figure."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(True, axis="y", alpha=1.0)
    ax.set_axisbelow(True)
