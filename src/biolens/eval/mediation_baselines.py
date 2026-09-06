"""
Baselines for the Geuvadis case study's mediation result: raw-embedding-
difference regression, TWAS-style genotype regression, and pretrained
Borzoi inference — all three run together, since no single classical
baseline covers the full comparison space against the SAE-feature-based
result. All three are real, complete implementations.

Borzoi's package/checkpoint choice (`borzoi-pytorch`, github.com/johahi/
borzoi-pytorch) and every I/O constant below (one-hot base order, input
length, output shape, bin resolution, track indices) were verified directly
against primary sources on 2026-07-28 — not assumed from memory — see
`borzoi_baseline`'s docstring for exactly what was checked and against what.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── Borzoi I/O constants, all verified 2026-07-28 against primary sources ──
# (not the borzoi-pytorch README, which omits them) — see borzoi_baseline's
# docstring for exactly which source confirmed each value.
_BORZOI_SEQ_LENGTH = 524288          # exact input length; confirmed via the
                                       # package's own wt_seq.npy fixture shape
_BORZOI_BIN_SIZE = 32                 # bp per output bin
_BORZOI_OUTPUT_BINS = 6144            # confirmed via BorzoiConfig.bins_to_return
                                       # AND the example notebook's real output
                                       # shape (1, 6144, 3)
_BORZOI_OUTPUT_COVERED_BP = _BORZOI_OUTPUT_BINS * _BORZOI_BIN_SIZE  # 196608
_BORZOI_CROP_BP = (_BORZOI_SEQ_LENGTH - _BORZOI_OUTPUT_COVERED_BP) // 2  # 163840
                                       # per side — the model's output covers
                                       # only the CENTER of its input window,
                                       # standard Enformer/Borzoi-lineage design
# Human GM12878 (the exact EBV-transformed lymphoblastoid cell line Geuvadis
# itself assays: a single tissue, EBV-transformed LCLs) RNA-seq (not CAGE)
# track indices, both strands.
# Read directly from the package's own borzoi_pytorch/precomputed/targets.txt
# (row index == output-tensor track index; description column == "RNA:GM12878"),
# not guessed or averaged over an unrelated tissue.
_BORZOI_GM12878_RNA_TRACK_INDICES = [
    6112, 6113, 6114, 6115, 6116, 6117, 6118, 6119, 6120, 6121,
    6204, 6205, 6340, 6434, 6514, 6515, 6668, 6669, 7306, 7307, 7334, 7394,
]

# Byte-value -> one-hot-row lookup for _dna_one_hot's vectorized encoding
# (ord('A')=65 .. ord('T')=84 all fit in a 256-entry table indexed directly
# by byte value — avoids a per-base Python loop over a 524288-length
# string, which is genuinely slow at Borzoi's fixed input length and would
# be called twice per locus in any real run).
_BASE_TO_ONEHOT_ROW = np.full(256, -1, dtype=np.int8)
_BASE_TO_ONEHOT_ROW[ord("A")] = 0
_BASE_TO_ONEHOT_ROW[ord("C")] = 1
_BASE_TO_ONEHOT_ROW[ord("G")] = 2
_BASE_TO_ONEHOT_ROW[ord("T")] = 3


@dataclass
class BaselineResult:
    method: str
    r_squared: float
    coefficient: float
    p_value: float
    n_obs: int

    def __str__(self) -> str:
        return (
            f"{self.method}: R²={self.r_squared:.4f}, coef={self.coefficient:.4f}, "
            f"p={self.p_value:.4g} (n={self.n_obs})"
        )


def raw_embedding_baseline(
    embedding_diff: np.ndarray,
    outcome: np.ndarray,
    covariates: pd.DataFrame | None = None,
) -> BaselineResult:
    """
    Raw-embedding-difference regression baseline: regress real expression
    directly on the FULL residual-stream embedding difference between
    ref/alt allele sequences (not the SAE feature — the point of comparison
    is "does going through the SAE lose signal relative to using the dense
    embedding directly").

    Args:
        embedding_diff: (n,) some scalar summary of the dense embedding
            difference (e.g. its L2 norm, or its projection onto the first
            principal component across the eval set) — a full (n, d_model)
            embedding matrix would need dimensionality reduction before a
            single-coefficient regression like this one; that reduction is
            the caller's responsibility (kept out of this function so the
            regression itself stays a one-line, auditable OLS fit).
        outcome:        (n,) real expression values.
        covariates:     Optional additional covariates.

    Returns:
        BaselineResult from an OLS fit of outcome ~ embedding_diff (+ covariates).
    """
    n = len(embedding_diff)
    if len(outcome) != n:
        raise ValueError("embedding_diff and outcome must be the same length")

    data = pd.DataFrame({"X": embedding_diff, "Y": outcome})
    covariate_cols: list[str] = []
    if covariates is not None:
        for col in covariates.columns:
            data[col] = covariates[col].to_numpy()
            covariate_cols.append(col)

    formula = "Y ~ X" + ("".join(f" + {c}" for c in covariate_cols))
    fit = sm.OLS.from_formula(formula, data).fit()

    return BaselineResult(
        method="raw_embedding",
        r_squared=float(fit.rsquared),
        coefficient=float(fit.params["X"]),
        p_value=float(fit.pvalues["X"]),
        n_obs=n,
    )


def twas_style_baseline(
    genotype_dosage: np.ndarray,
    outcome: np.ndarray,
    covariates: pd.DataFrame | None = None,
) -> BaselineResult:
    """
    TWAS-style baseline: the classical-genetics baseline, regressing
    expression directly on GENOTYPE dosage, not any model-derived
    representation, matching the standard eQTL/TWAS association-testing
    approach.

    This is a genuine, complete implementation (not a stub) — a TWAS-style
    genotype-on-expression OLS regression is exactly this simple; the "TWAS"
    framing refers to the analysis design (genotype as the sole predictor,
    no sequence-model mediator), not to needing external TWAS software
    (e.g. PrediXcan/FUSION, which build genotype-to-expression prediction
    weights from a separate reference panel — out of scope here since the
    comparison this baseline needs to support is "genotype alone vs.
    SAE-mediated genotype effect," not weighted-genotype gene-level TWAS
    imputation specifically).

    Args:
        genotype_dosage: (n,) 0/1/2 alt-allele dosage.
        outcome:         (n,) real expression values.
        covariates:      Optional additional covariates (e.g. genotype PCs).

    Returns:
        BaselineResult from an OLS fit of outcome ~ genotype_dosage (+ covariates).
    """
    n = len(genotype_dosage)
    if len(outcome) != n:
        raise ValueError("genotype_dosage and outcome must be the same length")

    data = pd.DataFrame({"G": genotype_dosage, "Y": outcome})
    covariate_cols: list[str] = []
    if covariates is not None:
        for col in covariates.columns:
            data[col] = covariates[col].to_numpy()
            covariate_cols.append(col)

    formula = "Y ~ G" + ("".join(f" + {c}" for c in covariate_cols))
    fit = sm.OLS.from_formula(formula, data).fit()

    return BaselineResult(
        method="twas_style",
        r_squared=float(fit.rsquared),
        coefficient=float(fit.params["G"]),
        p_value=float(fit.pvalues["G"]),
        n_obs=n,
    )


def _dna_one_hot(seq: str) -> "torch.Tensor":  # noqa: F821 — torch imported lazily below
    """
    A=0, C=1, G=2, T=3 one-hot encoding, shape (4, len(seq)) — channel-first,
    matching Borzoi's expected input layout (confirmed via the package's own
    example notebook: `sequence_one_hot.permute(1,0)[None, ...]` fed to the
    model, i.e. the model wants (batch, 4, seq_len)).

    Base order verified directly against calico/baskerville's own
    `dna_1hot()` (github.com/calico/baskerville/blob/main/src/baskerville/
    dna.py) — the reference implementation Borzoi itself was trained with,
    not assumed from a generic "alphabetical" convention. Any base outside
    ACGT (e.g. N) is left as an all-zero column, matching dna_1hot's
    default (non-n_sample) behavior.

    Vectorized via a byte-value lookup table (_BASE_TO_ONEHOT_ROW), not a
    per-base Python loop — at Borzoi's fixed 524288bp input length, a plain
    Python loop is slow enough to matter (this function is called twice per
    locus, ref and alt, in any real borzoi_baseline run).
    """
    import torch

    seq_bytes = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    rows = _BASE_TO_ONEHOT_ROW[seq_bytes]  # (L,), values in {-1, 0, 1, 2, 3}

    arr = np.zeros((4, len(seq)), dtype=np.float32)
    valid = rows >= 0
    arr[rows[valid], np.nonzero(valid)[0]] = 1.0
    return torch.from_numpy(arr)


def load_borzoi_model(checkpoint: str = "johahi/borzoi-replicate-0", device: str = "cpu"):
    """
    Load a Borzoi model once for reuse across many borzoi_baseline() calls.
    The Geuvadis case study calls one baseline per locus, potentially
    hundreds — reloading a ~0.2B-param checkpoint from scratch on every call
    would be wasteful and slow; the caller loads once via this function and
    passes the result to every borzoi_baseline call, the same "load once
    outside the per-locus loop" pattern scripts/run_geuvadis_case_study.py
    already uses for the Evo 2 SAE.

    Raises:
        ImportError: if `borzoi-pytorch` isn't installed (`pip install
            borzoi-pytorch`; see https://github.com/johahi/borzoi-pytorch).
    """
    try:
        from borzoi_pytorch import Borzoi
    except ImportError as exc:
        raise ImportError(
            "The 'borzoi-pytorch' package is required for the Borzoi baseline "
            "but is not installed. Install with `pip install -e '.[borzoi]'` "
            "(pins a transformers<5.0.0 range this package needs — a bare "
            "`pip install borzoi-pytorch` can silently leave an incompatible "
            "transformers already installed; see https://github.com/johahi/"
            "borzoi-pytorch)."
        ) from exc

    model = Borzoi.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    return model


def extract_borzoi_ref_alt_window(fasta, chrom: str, pos: int, ref_allele: str, alt_allele: str):
    """
    Convenience wrapper around `biolens.data.reference_genome.
    extract_ref_alt_window` that bakes in Borzoi's fixed input length —
    callers (e.g. scripts/run_geuvadis_case_study.py) shouldn't need to
    import the private `_BORZOI_SEQ_LENGTH` constant themselves just to
    extract a correctly-sized window; that's Borzoi-specific knowledge this
    module already owns (see `borzoi_baseline`'s docstring for where the
    524288 figure comes from).

    Returns:
        (ref_sequence, alt_sequence, window_start) — same as
        extract_ref_alt_window, with window_size fixed to
        `_BORZOI_SEQ_LENGTH`.
    """
    from biolens.data.reference_genome import extract_ref_alt_window

    return extract_ref_alt_window(
        fasta, chrom, pos, ref_allele, alt_allele, window_size=_BORZOI_SEQ_LENGTH
    )


def borzoi_baseline(
    ref_sequence: str,
    alt_sequence: str,
    window_start: int,
    gene_span: tuple[int, int],
    genotype_dosage: np.ndarray,
    outcome: np.ndarray,
    model=None,
    device: str = "cpu",
    checkpoint: str = "johahi/borzoi-replicate-0",
) -> BaselineResult:
    """
    Borzoi baseline: the ML-recognized SOTA baseline (Linder et al. 2023),
    predicting expression directly from ref/alt sequence using Borzoi's
    pretrained weights (inference only, not retraining).

    ONE locus at a time, matching raw_embedding_baseline/twas_style_baseline's
    existing per-locus calling convention at this baseline's actual call
    site (scripts/run_geuvadis_case_study.py's `_run_one_locus`): the DNA
    sequence is identical for every sample at a given allele (this is the
    reference genome plus one substituted SNP, not per-individual whole-
    genome sequences), so Borzoi only needs to run ONCE per allele per
    locus — not once per sample. The predicted ref->alt expression shift is
    then scaled by each sample's genotype dosage (0/1/2 alt-allele copies)
    and regressed against real per-sample expression, exactly mirroring how
    `embedding_diff_norm * treatment` is used for the raw-embedding
    baseline at that same call site.

    Package/checkpoint choice and every I/O constant were verified directly
    against primary sources on 2026-07-28, not assumed:
      - Package: `borzoi-pytorch` (github.com/johahi/borzoi-pytorch, MIT
        licensed, actively maintained — includes the newer Flashzoi variant
        too) chosen over the Genentech/HuggingFace PyTorch-Lightning
        checkpoint (requires the heavier `grelu` library) for a simpler,
        self-contained `.from_pretrained()` load matching this codebase's
        existing Evo2Model/gemma_scope.py pattern.
      - Input length (524288bp) and output shape ((1, 7611, 6144)):
        confirmed by downloading the package's own `wt_seq.npy` fixture
        (shape (524288, 4)) and reading its example notebook's actual
        printed output shape, not assumed from the README (which omits
        these numbers entirely).
      - Bin resolution (32bp) and crop (163840bp per side, i.e. the model's
        output covers only the CENTER 196608bp of its 524288bp input):
        derived from `BorzoiConfig.bins_to_return=6144` together with the
        confirmed output shape (6144 * 32 = 196608), the standard Enformer/
        Borzoi-lineage "long input, cropped output" design.
      - GM12878 (Geuvadis's own assayed EBV-transformed lymphoblastoid cell
        line) RNA-seq track indices: read directly from the package's own
        `borzoi_pytorch/precomputed/targets.txt` (real per-track metadata
        shipped with the package), not averaged over an unrelated tissue.
      - Track-to-gene-body aggregation: reuses `borzoi_pytorch.gene_utils.
        Gene.output_slice` verbatim — the original Calico/baskerville
        reference implementation (Apache-2.0, vendored into borzoi-pytorch),
        not a reimplementation, so its exon/overlap-boundary edge cases
        don't need to be independently re-derived here.

    Args:
        ref_sequence, alt_sequence: EXACTLY `_BORZOI_SEQ_LENGTH` (524288)
            bp each — use
            `biolens.data.reference_genome.extract_ref_alt_window(...,
            window_size=524288)`, NOT `extract_ref_alt_context` (which can
            only produce odd-length windows and can never hit this exact
            even length).
        window_start: The 0-indexed genomic coordinate of `ref_sequence[0]`
            — the third return value of `extract_ref_alt_window`. Needed to
            map the model's output bins back to genomic coordinates for the
            gene-body aggregation.
        gene_span: (gene_start, gene_end) — 0-indexed, half-open, genomic
            coordinates of the target gene's overall span (e.g. from
            `biolens.data.genomic_annotations.GencodeGene.start/.end`).
            Aggregation uses the gene's overall span (`Gene.output_slice(...,
            span=True)`), not per-exon boundaries — exon-level GTF data
            isn't threaded through the Geuvadis pipeline's existing gene
            lookup, and span-level aggregation is the documented, supported
            simpler mode of the same upstream utility.
        genotype_dosage: (n_samples,) 0/1/2 alt-allele dosage — same role as
            raw_embedding_baseline's `treatment` argument at the call site.
        outcome: (n_samples,) real per-sample expression values.
        model: An already-loaded model from `load_borzoi_model()`. If None,
            loads one internally via `checkpoint`/`device` — convenient for
            a single call or a test, but wasteful across many loci; the
            real call site loads once and passes it in.
        device: "cpu" or "cuda". Ignored if `model` is given (assumed
            already on the right device).
        checkpoint: Which Borzoi replicate to use, if `model` isn't given.
            A single replicate, not the full ensemble the original paper
            averages — documented, deliberate scope: this baseline exists
            as one comparison point among several, not to reproduce
            Borzoi's own paper-grade accuracy.

    Returns:
        BaselineResult from an OLS fit of
        outcome ~ (predicted_expression_shift * genotype_dosage).

    Raises:
        ImportError: if `borzoi-pytorch` isn't installed and `model` wasn't
            given (`pip install -e '.[borzoi]'`).
        ValueError: if ref/alt aren't exactly 524288bp, genotype_dosage and
            outcome aren't the same length, or the target gene doesn't
            overlap Borzoi's output window at all for this locus (nothing
            to predict — check gene_span/window_start are in the same
            genomic coordinate convention, 0-indexed half-open).
    """
    try:
        from borzoi_pytorch.gene_utils import Gene
    except ImportError as exc:
        raise ImportError(
            "The 'borzoi-pytorch' package is required for the Borzoi baseline "
            "but is not installed. Install with `pip install -e '.[borzoi]'` "
            "(pins a transformers<5.0.0 range this package needs — a bare "
            "`pip install borzoi-pytorch` can silently leave an incompatible "
            "transformers already installed; see https://github.com/johahi/"
            "borzoi-pytorch)."
        ) from exc
    import torch

    if len(genotype_dosage) != len(outcome):
        raise ValueError("genotype_dosage and outcome must be the same length")
    for label, seq in (("ref_sequence", ref_sequence), ("alt_sequence", alt_sequence)):
        if len(seq) != _BORZOI_SEQ_LENGTH:
            raise ValueError(
                f"{label} has length {len(seq)}, expected exactly "
                f"{_BORZOI_SEQ_LENGTH} (Borzoi's fixed input length) — use "
                f"biolens.data.reference_genome.extract_ref_alt_window(..., "
                f"window_size={_BORZOI_SEQ_LENGTH})."
            )

    gene_start, gene_end = gene_span
    output_window_start = window_start + _BORZOI_CROP_BP
    gene = Gene(chrom=None, strand="+", kv={})
    gene.add_exon(gene_start, gene_end)
    bin_idx = gene.output_slice(
        seq_start=output_window_start,
        seq_len=_BORZOI_OUTPUT_COVERED_BP,
        model_stride=_BORZOI_BIN_SIZE,
        span=True,
    )
    if len(bin_idx) == 0:
        raise ValueError(
            f"gene_span={gene_span} does not overlap Borzoi's output window "
            f"[{output_window_start}, {output_window_start + _BORZOI_OUTPUT_COVERED_BP}) "
            f"for this locus (window_start={window_start}) — nothing to predict. "
            f"Check gene_span/window_start are in the same genomic coordinate "
            f"convention (0-indexed, half-open)."
        )

    if model is None:
        model = load_borzoi_model(checkpoint=checkpoint, device=device)
    else:
        # Docstring's documented contract: device is "ignored if model is
        # given (assumed already on the right device)" — but the code
        # didn't actually honor that until now. Real production failure,
        # job 9727443, 2026-07-31: run_geuvadis_case_study.py loads Borzoi
        # once on cuda and passes it in via model=, never setting device=
        # (correctly, per the documented contract) — device then silently
        # defaulted to "cpu", moving ref_1h/alt_1h to CPU while model
        # stayed on CUDA, crashing on the classic "Input type
        # (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)"
        # mismatch. Deriving device from the model itself makes the code
        # match what the docstring already promised.
        device = next(model.parameters()).device

    with torch.no_grad():
        ref_1h = _dna_one_hot(ref_sequence).unsqueeze(0).to(device)
        alt_1h = _dna_one_hot(alt_sequence).unsqueeze(0).to(device)
        ref_out = model(ref_1h)[0, _BORZOI_GM12878_RNA_TRACK_INDICES, :].cpu().numpy()
        alt_out = model(alt_1h)[0, _BORZOI_GM12878_RNA_TRACK_INDICES, :].cpu().numpy()

    predicted_shift = float(alt_out[:, bin_idx].sum() - ref_out[:, bin_idx].sum())

    mediator = predicted_shift * np.asarray(genotype_dosage)
    data = pd.DataFrame({"X": mediator, "Y": np.asarray(outcome)})
    fit = sm.OLS.from_formula("Y ~ X", data).fit()

    return BaselineResult(
        method="borzoi",
        r_squared=float(fit.rsquared),
        coefficient=float(fit.params["X"]),
        p_value=float(fit.pvalues["X"]),
        n_obs=len(outcome),
    )
