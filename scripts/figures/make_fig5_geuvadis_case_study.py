"""
Figure 5: Geuvadis causal-mediation case study -- the naive (unverified)
SAE feature-selection pipeline's mediation estimates, alongside independent
genomic verification and two baselines that answer "versus what?".

Spec corrected 2026-08-29 after review found the original two-arm ("naive
vs. verified") spec did not match data that has only one arm. There is no
second, verified-feature mediation re-run: every candidate the naive
pipeline selects is independently flagged spurious by genomic verification before
any statistical test is run, so there is nothing left to run a second arm
on. This figure shows that directly: Panel A is a forest plot of the 19
ACME-reportable loci's naive-arm mediation estimates (0 significant after
BH correction, 3 at raw p<0.05, all 19 independently verified spurious).
Panel B answers "is the underlying genetic signal even real?" with the
TWAS-style and raw-embedding baselines already stored per locus in
docs/geuvadis_case_study_results.json -- yes, strongly (this is why the
naive SAE result being null is informative rather than just underpowered).

Usage: python scripts/figures/make_fig5_geuvadis_case_study.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plotstyle import (  # noqa: E402
    CATEGORICAL,
    INK_MUTED,
    INK_PRIMARY,
    STATUS,
    apply_style,
    style_axis,
)

REPO_ROOT = Path(__file__).parent.parent.parent
ACME_CI_PATTERN = re.compile(r"ACME=(-?[\d.]+) \(95% CI \[(-?[\d.]+), (-?[\d.]+)\]")
R_SQUARED_PATTERN = re.compile(r"R²=([\d.]+)")


def _r_squared(baseline_str: str) -> float:
    m = R_SQUARED_PATTERN.search(baseline_str)
    assert m is not None, f"R² not found in baseline string: {baseline_str}"
    return float(m.group(1))


def load_reportable_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r["naive_acme_reportable"]:
            continue
        m = ACME_CI_PATTERN.search(r["naive_mediation"])
        assert m is not None, f"ACME CI not found in mediation string for {r['variant_id']}"
        ci_lo, ci_hi = float(m.group(2)), float(m.group(3))
        out.append({**r, "acme_ci_lo": ci_lo, "acme_ci_hi": ci_hi})
    return out


def plot_forest(ax, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["naive_acme_estimate"])
    y = np.arange(len(rows))

    for i, r in enumerate(rows):
        lo, hi = r["acme_ci_lo"], r["acme_ci_hi"]
        est = r["naive_acme_estimate"]
        raw_sig = r["naive_acme_p"] < 0.05
        color = CATEGORICAL["blue"] if raw_sig else INK_MUTED
        ax.plot([lo, hi], [i, i], "-", color=color, linewidth=1.1, alpha=0.85, zorder=2)
        ax.plot(est, i, "o", color=color, markersize=4.2, markeredgecolor="white",
                 markeredgewidth=0.4, zorder=3)

    ax.axvline(0, color=INK_PRIMARY, linewidth=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["gene_id"].split(".")[0] for r in rows], fontsize=6.0, fontfamily="monospace")
    ax.set_xlabel("Naive-arm ACME estimate (95% CI)")
    ax.set_title(
        "A. All 19 ACME-reportable loci — 0/19 significant after BH,\n"
        "3/19 at raw p<0.05 (blue), all 19 independently flagged spurious",
        loc="left", fontweight="bold", fontsize=8.6,
    )
    style_axis(ax)
    ax.grid(False, axis="y")
    ax.set_ylim(-1, len(rows))


def plot_baselines(ax, rows: list[dict]) -> None:
    twas_r2 = [_r_squared(r["twas_style_baseline"]) for r in rows if r.get("twas_style_baseline")]
    raw_r2 = [_r_squared(r["raw_embedding_baseline"]) for r in rows if r.get("raw_embedding_baseline")]

    n_total_sig = sum(1 for r in rows if r["naive_total_effect_p"] < 0.05)
    n_acme_raw_sig = sum(1 for r in rows if r["naive_acme_p"] < 0.05)
    n_acme_bh_sig = sum(1 for r in rows if r["naive_acme_significant_after_bh"])

    bp = ax.boxplot(
        [twas_r2, raw_r2], positions=[0, 1], widths=0.5, patch_artist=True,
        showfliers=False,  # every point is already drawn as a jittered dot below -- no double-encoding
        medianprops=dict(color=INK_PRIMARY, linewidth=1.4),
    )
    for patch, color in zip(bp["boxes"], [CATEGORICAL["orange"], CATEGORICAL["aqua"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for element in ("whiskers", "caps"):
        for artist in bp[element]:
            artist.set_color(INK_MUTED)

    rng = np.random.default_rng(0)
    for pos, vals, color in [(0, twas_r2, CATEGORICAL["orange"]), (1, raw_r2, CATEGORICAL["aqua"])]:
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.array([pos] * len(vals)) + jitter, vals, s=14, color=color,
                    edgecolor="white", linewidth=0.4, zorder=3, alpha=0.9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["TWAS-style\n(genotype -> expression)", "Raw ESM2 embedding\n(mean-pool)"])
    ax.set_ylabel(r"$R^2$ (genotype/embedding -> expression)")
    ax.set_title(
        f"B. The underlying genetic signal is real and strong —\n"
        f"total-effect significant at {n_total_sig}/19 loci (raw p<0.05).\n"
        f"Naive-arm ACME: {n_acme_raw_sig}/19 raw-significant, {n_acme_bh_sig}/19 after BH.",
        loc="left", fontweight="bold", fontsize=8.6,
    )
    style_axis(ax)
    ax.grid(False, axis="x")


def main() -> None:
    apply_style()
    rows_all = json.loads((REPO_ROOT / "docs/geuvadis_case_study_results.json").read_text())
    rows = load_reportable_rows(rows_all)
    assert len(rows) == 19, f"expected 19 ACME-reportable loci, got {len(rows)}"
    assert all(r["naive_feature_verification_status"] == "spurious" for r in rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    plot_forest(ax1, rows)
    plot_baselines(ax2, rows)

    fig.suptitle(
        "Figure 5. The naive SAE-mediation pipeline finds nothing survivable, "
        "against a real and strongly-detected genetic signal",
        fontsize=9.2, y=1.04, x=0.02, ha="left",
    )
    fig.tight_layout()

    out_dir = REPO_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig5_geuvadis_case_study.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {out_dir}/fig5_geuvadis_case_study.{{pdf,png}}")


if __name__ == "__main__":
    main()
