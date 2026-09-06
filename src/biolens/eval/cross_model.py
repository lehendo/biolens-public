"""
Cross-modality causal mediation analysis (Phase 2).

Methodology (see design doc Section 3 for full derivation):

Phase 2a — DNA → expression (two-hop):
  Treatment:   DNA regulatory variant (allele dosage, from GTEx individual-level data)
  Mediator:    SAE feature activation shift in Evo 2 / HyenaDNA between alleles
  Outcome:     Measured expression change (GTEx normalized effect size, NES)
  Test:        Imai-Keele-Tingley (2010) causal mediation framework with
               sensitivity analysis for sequential ignorability violations.
               NOT classic Baron-Kenny (known limitations with treatment-mediator
               interaction and unmeasured confounding).
  Baseline:    Raw-residual-stream-embedding regression on same loci, same layer.
  Exit:        Spearman(SAE-predicted NES, GTEx NES) not significantly below
               Spearman(embedding-predicted NES, GTEx NES) — Wilcoxon paired,
               p < 0.05, n ≥ 500 held-out loci — AND significant path attenuation
               in Imai-Keele-Tingley test.

Phase 2b — DNA → protein → expression (three-hop):
  Adds protein abundance (pQTL data; verify sample size before committing)
  as a middle mediator via ESM2 SAE features.
  Granularity: aggregate all models to gene/locus level before comparison.

Phase 0/1 stub: data access application to GTEx dbGaP should be started
DURING Phase 0-1 engineering, not when Phase 2 is ready to begin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class MediationConfig:
    gtex_data_path: Path | None = None
    pqtl_data_path: Path | None = None
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    min_held_out_loci: int = 500
    wilcoxon_alpha: float = 0.05


@dataclass
class MediationResult:
    locus: str
    spearman_sae: float
    spearman_embedding: float
    wilcoxon_p: float
    mediation_proportion: float    # proportion of effect mediated through SAE features
    mediation_p: float             # Imai-Keele-Tingley test p-value
    sensitivity_rho: float         # sensitivity param at which mediation result breaks


def run_phase2a_mediation(
    dna_model,
    dna_sae,
    gtex_variants: pd.DataFrame,
    cfg: MediationConfig | None = None,
) -> list[MediationResult]:
    """
    Phase 2a — DNA → expression mediation via GTEx eQTLs.

    Requires:
      - Approved GTEx individual-level genotype + expression data (dbGaP DUA)
      - Evo 2 / HyenaDNA SAE trained and loaded
      - R package 'mediation' (Imai et al.) callable via rpy2, or a Python
        implementation of the IKT framework

    Phase 1/2 — not yet implemented.
    """
    raise NotImplementedError(
        "Phase 2 — cross-modality causal mediation. "
        "Prerequisites: (1) dbGaP data use agreement approved for GTEx "
        "individual-level genotype data; (2) Phase 1 multi-model support complete. "
        "Start the dbGaP application during Phase 0-1 engineering."
    )


def run_phase2b_mediation(
    dna_model, dna_sae,
    protein_model, protein_sae,
    pqtl_data: pd.DataFrame,
    cfg: MediationConfig | None = None,
) -> list[MediationResult]:
    """
    Phase 2b — DNA → protein → expression mediation via colocalized eQTL/pQTL pairs.

    IMPORTANT: verify colocalized pair count from the approved dataset before
    committing to this as a primary (non-pilot) result.  If n < 500 colocalized
    pairs, label explicitly as power-limited pilot in the writeup.
    """
    raise NotImplementedError("Phase 2b — see Phase 2a first.")
