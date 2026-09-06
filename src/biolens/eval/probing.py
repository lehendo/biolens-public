"""
Linear probing: do SAE features linearly predict known biological labels?

Methodology follows Adams et al. (2024) and InterPLM (Simon & Zou 2024),
applied uniformly across all model families so results are directly comparable.

For each GO term with >= min_positives examples in the eval set:
  1. single_feature_auroc: best AUROC achieved by any single SAE feature alone.
     A value >> 0.5 is evidence of a monosemantic, interpretable feature.
  2. multivariate_auroc:   AUROC of logistic regression on all SAE features.
     Upper bound on linear separability.

Phase 0 exit criterion (configs/eval.yaml::probing.phase0_exit_auroc):
  Any GO term with single_feature_auroc > 0.70.

This module also supports subcellular localization and thermostability probing
(for proteins), motif enrichment probing (for genomic models) — added as the
corresponding annotation datasets are integrated.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import Tensor

if TYPE_CHECKING:
    from biolens.data.uniprot import GODag

logger = logging.getLogger(__name__)


@dataclass
class GOProbingResult:
    """Results for a single GO term."""
    go_id: str
    go_name: str
    n_positives: int
    n_total: int
    single_feature_auroc: float       # max AUROC across individual SAE features
    best_feature_idx: int             # which feature achieved it
    multivariate_auroc: float         # logistic regression on all features
    # Distribution of single-feature AUROCs (for diagnostic plots)
    per_feature_aurocs: np.ndarray | None = field(default=None, repr=False)
    # ── Selection-bias baselines — only populated when probe_go_terms is
    # called with compute_held_out_baselines=True. held_out_auroc de-biases
    # the *number* (classical sample-split fix); hard_negative_held_out_auroc
    # additionally stress-tests whether that cheap fix alone would have
    # caught the same structural confounding a verifier catches, without
    # needing an LLM pipeline.
    held_out_auroc: float | None = None
    hard_negative_held_out_auroc: float | None = None
    n_hard_negatives: int | None = None

    @property
    def auroc_inflation(self) -> float | None:
        """`single_feature_auroc - held_out_auroc` — the real-data AUROC-
        inflation-vs-n quantity the winner's-curse lemma actually predicts
        (as opposed to `verified_rate`, a different, atheoretical curve).
        None if held_out_auroc wasn't computed for this result."""
        if self.held_out_auroc is None:
            return None
        return self.single_feature_auroc - self.held_out_auroc


@dataclass
class GOProbingReport:
    """Aggregate probing results across all GO terms."""
    results: list[GOProbingResult]
    n_go_terms_tested: int
    model_name: str
    layer: int
    sae_variant: str

    def fraction_above_auroc(self, threshold: float) -> float:
        if not self.results:
            return 0.0
        above = sum(r.single_feature_auroc >= threshold for r in self.results)
        return above / len(self.results)

    def top_results(self, n: int = 10) -> list[GOProbingResult]:
        return sorted(self.results, key=lambda r: r.single_feature_auroc, reverse=True)[:n]

    def __str__(self) -> str:
        lines = [
            f"GO probing — {self.model_name} layer {self.layer} ({self.sae_variant})",
            f"  GO terms tested:        {self.n_go_terms_tested}",
            f"  AUROC > 0.65:           {self.fraction_above_auroc(0.65)*100:.1f}%",
            f"  AUROC > 0.70:           {self.fraction_above_auroc(0.70)*100:.1f}%",
            f"  AUROC > 0.75:           {self.fraction_above_auroc(0.75)*100:.1f}%",
            f"  AUROC > 0.80:           {self.fraction_above_auroc(0.80)*100:.1f}%",
            "",
            "  Top 10 GO terms:",
        ]
        for r in self.top_results(10):
            lines.append(
                f"    {r.go_id:12s} {r.go_name[:35]:35s}  "
                f"feat={r.single_feature_auroc:.3f}  multi={r.multivariate_auroc:.3f}  "
                f"n+={r.n_positives}"
            )
        return "\n".join(lines)


def _probe_single_term(
    Z: np.ndarray,
    protein_ids: list[str],
    y: np.ndarray,
    go_id: str,
    go_name: str,
    go_labels: dict[str, set[str]],
    splitter: StratifiedShuffleSplit,
    held_out_splitter: StratifiedShuffleSplit,
    min_positives: int,
    compute_held_out_baselines: bool,
    go_dag: "GODag | None",
    min_hard_negatives: int,
    return_per_feature_aurocs: bool,
    random_state: int,
    compute_multivariate_auroc: bool = True,
) -> "GOProbingResult | None":
    """
    Run the full single-feature/multivariate/held-out/hard-negative probing
    pipeline for ONE already-constructed label array `y` against one term
    identity (`go_id`/`go_name`, used only for logging/attribution — the
    actual positives are whatever `y` says, not re-derived from go_labels).

    compute_multivariate_auroc: Set False to skip the LogisticRegression fit
        entirely (multivariate_auroc comes back NaN) — added 2026-07-29 after
        a real production timeout: scripts/run_geuvadis_naive_probing.py only
        ever consumes best_feature_idx (from single_feature_auroc), never
        multivariate_auroc, but was paying for ~500 expensive d_sae=32768
        multivariate fits anyway (job 9701717, TIMEOUT at the 4h wall-clock
        limit with probing barely started) — a real, not hypothetical, waste
        for that specific caller. Defaults True so every other existing
        caller's behavior/output is unchanged.

    Extracted from probe_go_terms' per-term loop body (previously inline,
    unchanged in every particular) so probe_go_term_controlled_subsampling
    (a controlled-subsampling variant that isolates sample size from
    concept-difficulty confounding) can reuse the identical
    selection/held-out/multivariate logic against an
    artificially-subsampled `y` for a single fixed GO term, instead of
    duplicating ~150 carefully-tuned lines (see the C=0.01/max_iter=500
    comments below for why this isn't safe to casually re-derive).

    Returns None if `y` doesn't have enough positives/negatives for a valid
    split (mirrors probe_go_terms' previous inline `continue` behavior).
    """
    n_pos = int(y.sum())
    if n_pos < min_positives or n_pos >= len(y) - min_positives:
        return None

    # Train/test split stratified by label
    try:
        train_idx, test_idx = next(splitter.split(Z, y))
    except ValueError:
        return None

    Z_train, Z_test = Z[train_idx], Z[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ── Selection-bias baselines: carve the test split into a disjoint
    # "selection" subset (the AUROC-maximizing candidate-selection step,
    # matching single_feature_auroc's methodology exactly) and a
    # "held-out" subset (used only to estimate the ALREADY-selected
    # feature's AUROC — never to re-select). Falls back to using the
    # full test split for selection (byte-for-byte the pre-existing
    # behavior) when the flag is off. ────────────────────────────────
    held_out_auroc: float | None = None
    hard_negative_auroc: float | None = None
    n_hard_negatives: int | None = None

    if compute_held_out_baselines:
        try:
            sel_idx, ho_idx = next(held_out_splitter.split(Z_test, y_test))
        except ValueError:
            sel_idx = np.arange(len(y_test))
            ho_idx = np.array([], dtype=int)
        Z_sel, y_sel = Z_test[sel_idx], y_test[sel_idx]
        Z_ho, y_ho = Z_test[ho_idx], y_test[ho_idx]
        ho_protein_ids = [protein_ids[test_idx[i]] for i in ho_idx]
    else:
        Z_sel, y_sel = Z_test, y_test
        Z_ho, y_ho, ho_protein_ids = None, None, []

    # ── Single-feature AUROC (selection step) ──────────────────────────
    per_feature_aurocs = _vectorized_auroc(Z_sel, y_sel)
    best_feat_idx = int(np.argmax(per_feature_aurocs))
    single_feat_auroc = float(per_feature_aurocs[best_feat_idx])

    if compute_held_out_baselines and Z_ho is not None and len(y_ho) > 0:
        # Direction is fixed by the selection step, never re-optimized on
        # held-out data — re-reflecting here would silently reintroduce
        # the exact selection bias this baseline exists to measure.
        positive_direction = _raw_auroc(
            Z_sel[:, best_feat_idx], y_sel
        ) >= 0.5

        if y_ho.sum() > 0 and (len(y_ho) - y_ho.sum()) > 0:
            raw = _raw_auroc(Z_ho[:, best_feat_idx], y_ho)
            held_out_auroc = float(raw if positive_direction else 1.0 - raw)

        if go_dag is not None:
            hard_negative_auroc, n_hard_negatives = _hard_negative_auroc(
                scores=Z_ho[:, best_feat_idx],
                labels=y_ho,
                protein_ids=ho_protein_ids,
                go_id=go_id,
                go_labels=go_labels,
                go_dag=go_dag,
                positive_direction=positive_direction,
                min_hard_negatives=min_hard_negatives,
            )

    # ── Multivariate AUROC ────────────────────────────────────────────
    # Standardize before fitting: lbfgs's convergence rate depends heavily
    # on feature conditioning, and unscaled SAE activations can vary
    # wildly in per-feature magnitude — especially for JumpReLU SAEs
    # (e.g. Gemma Scope) which, unlike TopK, have no fixed per-example
    # sparsity count and a much denser, more varied activation pattern.
    # This caused real, repeated `lbfgs failed to converge` warnings and
    # multi-hour runtimes on the Gemma Scope (non-biology) probing
    # (2026-07-04/05) that never showed up on any TopK-SAE (ESM2/Evo2)
    # run — those are sparse enough (exactly k nonzero per example) that
    # conditioning was never actually a problem there, which is why this
    # was never hit until now. max_iter is also lowered from 1000 to 200:
    # with standardization this should be more than sufficient, and it
    # caps the worst case (a single stubborn fit could otherwise consume
    # unbounded time across potentially hundreds of fits — one per
    # candidate concept, per min_positives sweep point, per real/
    # permuted-label arm).
    #
    # Standardization alone did NOT fully fix it (2026-07-06: still
    # non-converging, just faster — 4:54 instead of 6+ hours), and a
    # first attempt at C=0.1 (chosen from a small synthetic i.i.d.-
    # Gaussian test) also didn't fully fix it in production (245
    # ConvergenceWarnings in the real 7-point sweep, 2026-07-07). Root
    # cause, confirmed directly against real cached Gemma Scope features
    # (not synthetic data) via an interactive debug session: the design
    # matrix is severely rank-deficient relative to its nominal
    # d_sae=16384 columns — a randomized partial SVD on 2000 real
    # examples found 60% of even the top 1800 singular values already
    # below 1e-6 * the largest — because a JumpReLU SAE's dictionary is
    # a massively overcomplete code over Gemma-2-2B's much smaller
    # d_model=2304 residual stream, unlike i.i.d. Gaussian synthetic
    # features (which are essentially full-rank and therefore didn't
    # reproduce this). No TopK SAE run (ESM2/Evo2) has ever hit this:
    # their d_sae (<=10240) is small enough, and/or their training-set
    # sizes large enough, to stay well clear of it. Scaling doesn't fix
    # a near-singular Hessian; stronger L2 regularization does. C was
    # swept (1.0/0.1/0.01/0.001/0.0001/0.00001) against real features at
    # n_train=1600 (harder than production's n_train~=16000, i.e. this
    # should transfer at least as well): C=0.01 was the unique sweet
    # spot — converges in 67 lbfgs iterations (vs. never at C=1.0 within
    # 200) AND achieves the best held-out AUROC of any C tried (0.828 vs.
    # 0.778 at C=1.0/max_iter=1000) — heavier regularization here isn't
    # just a numerics fix, it's the statistically correct choice given
    # the rank deficiency, not merely a means to silence a warning. This
    # changes multivariate_auroc's value, not just whether a warning is
    # printed, so it must be documented as a methodology choice.
    #
    # C=0.01 alone left 35/~318 fits (2026-07-13, job 9457999) still
    # non-converging at max_iter=200 — but per-fit go_id attribution
    # (added specifically to answer this) showed it was always the same
    # 4 known classes (profession_21/2/19/13, the largest/most
    # information-rich of the 28), not a regularization-strength
    # problem: a targeted debug session against the real, full-scale
    # (n_train=16000) feature matrix for exactly these 4 classes showed
    # C=0.01 was already correct — they just needed >200 iterations to
    # actually finish (148-276), and their AUROC at 200 iterations
    # (non-converged) already matched their converged value to within
    # noise (e.g. profession_21: 0.8018 -> 0.8024), confirming these
    # were "almost converged," not divergent. max_iter=500 covers all
    # 4 with margin; 1000/2000 give identical results (nothing left to
    # gain), so 500 is the right cap, not an arbitrarily large one.
    if not compute_multivariate_auroc:
        multi_auroc = float("nan")
    else:
        scaler = StandardScaler()
        Z_train_scaled = scaler.fit_transform(Z_train)
        Z_test_scaled = scaler.transform(Z_test)

        lr = LogisticRegression(
            max_iter=500,
            C=0.01,
            solver="lbfgs",
            random_state=random_state,
        )
        # Explicit per-fit warning attribution, not just a bare .fit() call:
        # a bare ConvergenceWarning is anonymous in the log stream (one line
        # among ~300+ fits across all GO terms/min_positives/real-vs-permuted
        # arms), which is exactly why the residual ~11% of non-converging
        # fits after the C=0.01 fix (2026-07-12, job 9454861: 35/~318)
        # couldn't be attributed to any specific GO term or arm after the
        # fact — a targeted debug session then showed the "extreme class
        # imbalance" hypothesis was wrong (the smallest real classes,
        # n_positives=15-17, converged cleanly). Wraps (rather than
        # replaces) the active warning handler so the warning still
        # propagates normally to stderr and to any outer
        # `warnings.catch_warnings(record=True)` capture (e.g. this
        # module's own tests) — swallowing it here would silently break
        # those.
        old_showwarning = warnings.showwarning

        def _log_and_show(message, category, filename, lineno, file=None, line=None):
            if issubclass(category, ConvergenceWarning):
                logger.warning(
                    "multivariate_auroc: lbfgs did not converge for %s "
                    "(n_positives=%d, n_train=%d, d_sae=%d)",
                    go_id, n_pos, len(y_train), Z.shape[1],
                )
            old_showwarning(message, category, filename, lineno, file, line)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("always", category=ConvergenceWarning)
                warnings.showwarning = _log_and_show
                lr.fit(Z_train_scaled, y_train)
            y_score = lr.predict_proba(Z_test_scaled)[:, 1]
            multi_auroc = float(roc_auc_score(y_test, y_score))
        except Exception as exc:
            logger.debug("Logistic regression failed for %s: %s", go_id, exc)
            multi_auroc = float("nan")

    return GOProbingResult(
        go_id=go_id,
        go_name=go_name,
        n_positives=n_pos,
        n_total=len(y),
        single_feature_auroc=single_feat_auroc,
        best_feature_idx=best_feat_idx,
        multivariate_auroc=multi_auroc,
        per_feature_aurocs=(
            per_feature_aurocs if return_per_feature_aurocs else None
        ),
        held_out_auroc=held_out_auroc,
        hard_negative_held_out_auroc=hard_negative_auroc,
        n_hard_negatives=n_hard_negatives,
    )


def probe_go_terms(
    feature_acts: Tensor,
    protein_ids: list[str],
    go_labels: dict[str, set[str]],
    go_names: dict[str, str] | None = None,
    min_positives: int = 50,
    max_go_terms: int = 500,
    train_fraction: float = 0.8,
    model_name: str = "",
    layer: int = -1,
    sae_variant: str = "",
    return_per_feature_aurocs: bool = False,
    random_state: int = 42,
    compute_held_out_baselines: bool = False,
    held_out_fraction: float = 0.5,
    go_dag: GODag | None = None,
    min_hard_negatives: int = 10,
    compute_multivariate_auroc: bool = True,
) -> GOProbingReport:
    """
    Run GO term linear probing on SAE feature activations.

    Args:
        feature_acts:   (N, d_sae) SAE feature activation matrix.
        protein_ids:    N UniProt accession IDs corresponding to each row.
        go_labels:      protein_id → set of GO term IDs (e.g. {"GO:0005737", ...}).
        go_names:       Optional GO ID → human-readable name (for reporting).
        min_positives:  Skip GO terms with fewer positive examples.
        max_go_terms:   Cap number of GO terms (sample most-common if exceeded).
        train_fraction: Train/test split.
        return_per_feature_aurocs: If True, store (d_sae,) AUROC array per result.
        compute_multivariate_auroc: Set False to skip the (potentially expensive
            — a full LogisticRegression fit per term, d_sae-dimensional) multivariate
            fit entirely; multivariate_auroc comes back NaN. See
            _probe_single_term's docstring for why this exists (a real production
            timeout, not a hypothetical optimization). Defaults True, unchanged
            behavior for every existing caller.
        compute_held_out_baselines: If True, further split the test set into a
            "selection" subset (used exactly as `single_feature_auroc` already
            is — unchanged when this flag is off) and a disjoint "held-out"
            subset, then report the selected feature's AUROC on the held-out
            subset (the classical sample-split de-biasing baseline) and,
            if `go_dag` is given, its AUROC against a matched hard-negative
            set drawn from GO-DAG siblings of `go_id`. Off by default — a distinct,
            opt-in run mode, not a change to existing single_feature_auroc
            numbers when unset.
        held_out_fraction: Of the test split, the fraction reserved as the
            disjoint held-out-estimation subset (the rest remains the
            "selection" subset). Only used when compute_held_out_baselines.
        go_dag: Optional GODag (see biolens.data.uniprot.resolve_go_dag) for
            hard-negative sibling lookup. If None, hard_negative_held_out_auroc
            is left unset even when compute_held_out_baselines=True.
        min_hard_negatives: Skip the hard-negative AUROC for a GO term if
            fewer than this many sibling-annotated hard negatives are
            available in the held-out subset — an unreliable estimate from a
            handful of hard negatives is worse than reporting "not enough
            data" honestly.

    Returns:
        GOProbingReport with per-GO-term results.
    """
    Z = feature_acts.float().numpy()   # (N, d_sae)
    go_names = go_names or {}

    # Collect all GO terms and their positive counts
    all_go_ids: dict[str, int] = {}
    for gos in go_labels.values():
        for go_id in gos:
            all_go_ids[go_id] = all_go_ids.get(go_id, 0) + 1

    # Filter to GO terms with enough positives and that appear in our protein set
    eligible = sorted(
        [(go_id, cnt) for go_id, cnt in all_go_ids.items() if cnt >= min_positives],
        key=lambda x: -x[1],
    )
    if len(eligible) > max_go_terms:
        eligible = eligible[:max_go_terms]

    logger.info(
        "Probing %d GO terms (min_positives=%d) across %d proteins, d_sae=%d",
        len(eligible), min_positives, len(protein_ids), Z.shape[1],
    )

    results = []
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - train_fraction, random_state=random_state
    )
    held_out_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=held_out_fraction, random_state=random_state
    )

    for go_id, _ in eligible:
        y = np.array([
            1 if pid in go_labels and go_id in go_labels[pid] else 0
            for pid in protein_ids
        ])
        result = _probe_single_term(
            Z=Z,
            protein_ids=protein_ids,
            y=y,
            go_id=go_id,
            go_name=go_names.get(go_id, go_id),
            go_labels=go_labels,
            splitter=splitter,
            held_out_splitter=held_out_splitter,
            min_positives=min_positives,
            compute_held_out_baselines=compute_held_out_baselines,
            go_dag=go_dag,
            min_hard_negatives=min_hard_negatives,
            return_per_feature_aurocs=return_per_feature_aurocs,
            random_state=random_state,
            compute_multivariate_auroc=compute_multivariate_auroc,
        )
        if result is not None:
            results.append(result)

    report = GOProbingReport(
        results=results,
        n_go_terms_tested=len(results),
        model_name=model_name,
        layer=layer,
        sae_variant=sae_variant,
    )
    logger.info("\n%s", report)
    return report


@dataclass
class ControlledSubsamplingSummary:
    """One target_n_positives point of the controlled subsampling variant:
    pick a GO term with a naturally large positive count, artificially
    downsample it, and show the inflation-vs-n pattern holds with concept
    difficulty held fixed — isolating the pure sample-size effect from the
    concept-difficulty confound inherent in probe_go_terms' ordinary sweep,
    which compares across DIFFERENT natural GO terms at each min_positives
    value."""

    go_id: str
    go_name: str
    n_natural_positives: int
    target_n_positives: int
    n_repeats_successful: int
    mean_single_feature_auroc: float
    mean_held_out_auroc: float | None
    mean_auroc_inflation: float | None
    auroc_inflation_ci95: tuple[float, float] | None

    def __str__(self) -> str:
        ci = (
            f"[{self.auroc_inflation_ci95[0]:.4f}, {self.auroc_inflation_ci95[1]:.4f}]"
            if self.auroc_inflation_ci95 is not None else "n/a"
        )
        infl = f"{self.mean_auroc_inflation:.4f}" if self.mean_auroc_inflation is not None else "n/a"
        return (
            f"  n={self.target_n_positives:4d}  single={self.mean_single_feature_auroc:.4f}  "
            f"inflation={infl}  95% CI={ci}  ({self.n_repeats_successful} repeats)"
        )


def probe_go_term_controlled_subsampling(
    feature_acts: Tensor,
    protein_ids: list[str],
    go_labels: dict[str, set[str]],
    go_id: str,
    target_n_positives: list[int],
    n_repeats: int = 10,
    go_name: str | None = None,
    train_fraction: float = 0.8,
    held_out_fraction: float = 0.5,
    compute_held_out_baselines: bool = True,
    go_dag: "GODag | None" = None,
    min_hard_negatives: int = 10,
    confidence: float = 0.95,
    random_state: int = 42,
    compute_multivariate_auroc: bool = False,
) -> list[ControlledSubsamplingSummary]:
    """
    Controlled subsampling variant.

    probe_go_terms' ordinary min_positives sweep varies sample size by
    comparing across DIFFERENT natural GO terms at each threshold — real,
    but confounded with concept difficulty (a term with 300 positives isn't
    just "the same concept with more data" than one with 10; it's a
    different concept entirely, which may be intrinsically easier or harder
    to linearly separate). This function isolates the pure sample-size
    effect: pick ONE GO term with a large natural positive count, and for
    each target size in `target_n_positives`, run `n_repeats` independent
    draws that each keep exactly `target_n_positives[i]` of that term's
    REAL positives (the rest are treated as ordinary negatives for that
    draw — not removed from the dataset, just not revealed as positive,
    mirroring what "only having observed this many real examples" would
    actually look like) and the full, unchanged negative pool — same
    concept, same negatives, only the visible positive count varies.

    Reuses _probe_single_term (the exact per-term selection/held-out/
    multivariate logic probe_go_terms itself uses) unchanged for each draw,
    with a fresh, distinct random_state per (target_n, repeat) pair so
    repeats give a genuine resampling distribution, not the same split
    replayed with different labels.

    Args:
        feature_acts, protein_ids, go_labels: Same as probe_go_terms.
        go_id: The single GO term to hold fixed.
        target_n_positives: Subsample sizes to test (e.g. [20, 50, 100, 200])
            — every value must be <= the term's real natural positive count.
        n_repeats: Independent draws per target size, for a mean + CI
            rather than one noisy point estimate.
        go_name: Optional human-readable name, for logging/reporting only.
        confidence: Confidence level for the percentile CI on
            mean_auroc_inflation across repeats.
        random_state: Base seed — both which positives get sampled per
            repeat and each repeat's train/held-out splits derive from this
            deterministically.
        compute_multivariate_auroc: Defaults False, unlike probe_go_terms —
            ControlledSubsamplingSummary has no multivariate_auroc field at
            all, so computing it here was pure wasted compute (up to
            `n_repeats * len(target_n_positives)` full LogisticRegression
            fits per call, discarded every time) until this was added
            2026-07-29; the same waste `_probe_single_term`'s own
            `compute_multivariate_auroc` parameter was built to let callers
            opt out of.

    Returns:
        One ControlledSubsamplingSummary per target_n_positives value, in
        the same order given.

    Raises:
        ValueError: if go_id has fewer natural positives than the largest
            requested target_n_positives — there's nothing to subsample
            from at that size.
    """
    Z = feature_acts.float().numpy()
    resolved_go_name = go_name or go_id

    full_y = np.array([
        1 if pid in go_labels and go_id in go_labels[pid] else 0
        for pid in protein_ids
    ])
    n_natural_positives = int(full_y.sum())
    positive_indices = np.where(full_y == 1)[0]

    if target_n_positives and max(target_n_positives) > n_natural_positives:
        raise ValueError(
            f"{go_id} has only {n_natural_positives} natural positives in this "
            f"eval set, cannot subsample up to {max(target_n_positives)} — pick "
            f"a GO term with a larger natural positive count, or lower the "
            f"largest requested target_n_positives."
        )

    rng = np.random.default_rng(random_state)
    alpha = 1.0 - confidence

    summaries: list[ControlledSubsamplingSummary] = []
    for target_n in target_n_positives:
        inflations: list[float] = []
        single_aurocs: list[float] = []
        held_out_aurocs: list[float] = []

        for repeat in range(n_repeats):
            sampled_pos = rng.choice(positive_indices, size=target_n, replace=False)
            y = np.zeros(len(full_y), dtype=int)
            y[sampled_pos] = 1

            # Distinct split per repeat (not the base random_state reused
            # every time) — otherwise every repeat at a given target_n would
            # share the exact same train/held-out partition and only differ
            # in which positives were masked, understating real resampling
            # variance rather than estimating it.
            repeat_seed = int(rng.integers(0, 2**31 - 1))
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=1.0 - train_fraction, random_state=repeat_seed
            )
            held_out_splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=held_out_fraction, random_state=repeat_seed
            )

            result = _probe_single_term(
                Z=Z,
                protein_ids=protein_ids,
                y=y,
                go_id=go_id,
                go_name=resolved_go_name,
                go_labels=go_labels,
                splitter=splitter,
                held_out_splitter=held_out_splitter,
                # min_positives is a no-op filter here: y is constructed with
                # EXACTLY target_n positives, so n_pos == target_n always —
                # 1 just satisfies _probe_single_term's signature without
                # rejecting any valid draw.
                min_positives=1,
                compute_held_out_baselines=compute_held_out_baselines,
                go_dag=go_dag,
                min_hard_negatives=min_hard_negatives,
                return_per_feature_aurocs=False,
                random_state=repeat_seed,
                compute_multivariate_auroc=compute_multivariate_auroc,
            )
            if result is None:
                continue

            single_aurocs.append(result.single_feature_auroc)
            if result.held_out_auroc is not None:
                held_out_aurocs.append(result.held_out_auroc)
            if result.auroc_inflation is not None:
                inflations.append(result.auroc_inflation)

        ci: tuple[float, float] | None = None
        if len(inflations) >= 2:
            lo, hi = np.percentile(inflations, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            ci = (float(lo), float(hi))

        summaries.append(ControlledSubsamplingSummary(
            go_id=go_id,
            go_name=resolved_go_name,
            n_natural_positives=n_natural_positives,
            target_n_positives=target_n,
            n_repeats_successful=len(single_aurocs),
            mean_single_feature_auroc=float(np.mean(single_aurocs)) if single_aurocs else float("nan"),
            mean_held_out_auroc=float(np.mean(held_out_aurocs)) if held_out_aurocs else None,
            mean_auroc_inflation=float(np.mean(inflations)) if inflations else None,
            auroc_inflation_ci95=ci,
        ))

    logger.info(
        "Controlled subsampling for %s (%s, %d natural positives):\n%s",
        go_id, resolved_go_name, n_natural_positives,
        "\n".join(str(s) for s in summaries),
    )
    return summaries


def probe_genomic_annotations(
    feature_acts: Tensor,
    window_ids: list[str],
    dna_labels: dict[str, set[str]],
    **kwargs: object,
) -> GOProbingReport:
    """
    Genomic-annotation analog of probe_go_terms: labels come from GENCODE /
    ENCODE cCREs instead of GO terms, built for Evo 2.

    The underlying statistical machinery (selection over many noisy
    per-feature estimates, held-out/hard-negative baselines, clustering,
    permuted-label control) doesn't care whether the labels are GO terms or
    ENCODE cCRE classes — `probe_go_terms` is already fully domain-generic in
    its actual implementation. This wrapper exists purely so DNA-probing call
    sites don't read as "probing GO terms" on genomic data, which would be
    confusing to a future reader — not because the engine underneath differs.

    Args:
        feature_acts: (N, d_sae) SAE feature activations for N DNA windows.
        window_ids:   N window IDs (e.g. "chr1:1000-2000"), playing the same
                      role `protein_ids` plays for GO probing.
        dna_labels:   window_id -> set of ENCODE cCRE class labels, from
                      biolens.data.genomic_annotations.label_dna_windows.
        **kwargs:     Forwarded to probe_go_terms (min_positives, go_dag for
                      hard-negative sampling if a genomic-annotation hierarchy
                      is later added, compute_held_out_baselines, etc.) — note
                      the returned report's field names still say "go_id" /
                      "go_name" (they hold cCRE class labels here instead);
                      not renamed, to keep this a thin, low-maintenance
                      wrapper around one shared, well-tested implementation
                      rather than a parallel dataclass hierarchy.

    Returns:
        GOProbingReport — field names are inherited from the GO-probing
        engine (see Args note above), values are genomic-annotation results.
    """
    return probe_go_terms(
        feature_acts=feature_acts,
        protein_ids=window_ids,
        go_labels=dna_labels,
        **kwargs,  # type: ignore[arg-type]
    )


def probe_text_concepts(
    feature_acts: Tensor,
    text_ids: list[str],
    text_labels: dict[str, set[str]],
    **kwargs: object,
) -> GOProbingReport:
    """
    Non-biological analog of probe_go_terms, used as a non-biology control
    domain: concept labels here are e.g.
    Bias-in-Bios profession classes (biolens.data.saebench_datasets) rather
    than GO terms or ENCODE cCREs, and features come from a pretrained
    Gemma Scope SAE (biolens.models.gemma_scope) rather than one of this
    project's own trained SAEs — but the selection-bias mechanism under
    study doesn't care about any of that, so this is, again, a thin naming
    wrapper around the one shared, well-tested sweep engine, not a
    reimplementation. See probe_genomic_annotations's docstring for the
    same argument made once already for the DNA case.

    Args:
        feature_acts: (N, d_sae) SAE feature activations for N text examples.
        text_ids:     N example IDs, playing the same role `protein_ids`
                      plays for GO probing.
        text_labels:  text_id -> set of concept labels (e.g.
                      {"profession_21"}), from
                      biolens.data.saebench_datasets.load_bias_in_bios.
        **kwargs:     Forwarded to probe_go_terms — see probe_genomic_
                      annotations's docstring for the same note about
                      inherited "go_id"/"go_name" field names.

    Returns:
        GOProbingReport — field names inherited from the GO-probing engine,
        values are non-biological concept-probing results.
    """
    return probe_go_terms(
        feature_acts=feature_acts,
        protein_ids=text_ids,
        go_labels=text_labels,
        **kwargs,  # type: ignore[arg-type]
    )


def _raw_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUROC for a single feature column, WITHOUT the below-0.5 reflection
    `_vectorized_auroc` applies. Held-out estimation must evaluate the exact
    direction chosen during selection — reflecting again on held-out data
    would silently let a mediocre feature "flip direction" to look good,
    reintroducing the same selection bias this baseline exists to measure.
    """
    pos_mask = labels == 1
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    from scipy.stats import rankdata

    ranks = rankdata(scores)
    rank_sum_pos = ranks[pos_mask].sum()
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * n_neg))


def _hard_negative_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    protein_ids: list[str],
    go_id: str,
    go_labels: dict[str, set[str]],
    go_dag: GODag,
    positive_direction: bool,
    min_hard_negatives: int,
) -> tuple[float | None, int | None]:
    """
    AUROC of the selected feature restricted to a matched hard-negative set:
    positives unchanged (proteins annotated with go_id), negatives restricted
    to proteins annotated with a GO-DAG sibling of go_id but NOT go_id itself
    (e.g. other transmembrane-channel proteins as negatives for an
    acetylcholine-channel feature, rather than the generic negative pool).

    The strongest classical stress-test of whether a feature has learned
    the specific claimed concept or a
    broader structural class it happens to overlap with — a feature that
    only distinguishes the claimed class from *unrelated* proteins, but not
    from its own siblings, is exactly the "multi-pass transporter flagged as
    acetylcholine receptor" failure mode.

    Returns (auroc, n_hard_negatives), or (None, n_hard_negatives) if there
    aren't enough hard negatives in the held-out set to estimate reliably.
    """
    siblings = go_dag.siblings(go_id)
    if not siblings:
        return None, 0

    positive_mask = labels == 1
    hard_negative_mask = np.array(
        [
            (not positive_mask[i])
            and bool(go_labels.get(pid, set()) & siblings)
            for i, pid in enumerate(protein_ids)
        ]
    )
    n_hard_negatives = int(hard_negative_mask.sum())
    if n_hard_negatives < min_hard_negatives or positive_mask.sum() == 0:
        return None, n_hard_negatives

    mask = positive_mask | hard_negative_mask
    raw = _raw_auroc(scores[mask], labels[mask])
    auroc = raw if positive_direction else 1.0 - raw
    return float(auroc), n_hard_negatives


def _vectorized_auroc(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute AUROC for every column of scores independently.

    Uses the Wilcoxon rank-sum / Mann-Whitney U relationship:
        AUROC = U / (n_pos * n_neg)
    where U is computed via argsort of each feature column.

    Much faster than calling sklearn.roc_auc_score in a loop for large d_sae.

    Args:
        scores: (N, d_sae) float array — one column per feature.
        labels: (N,) binary int array.

    Returns:
        (d_sae,) float array of AUROCs, one per feature.
        Values < 0.5 are reflected to [0.5, 1.0] (we care about any linear
        separation, positive or negative correlation).
    """
    pos_mask = labels == 1
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], 0.5)

    # Rank each column independently.
    # NOTE: scipy.stats.rankdata(scores, axis=0) — ranking all columns in one
    # call — was benchmarked as a "vectorization" here and measured ~1.7x
    # SLOWER than this per-column loop at realistic scale (d_sae=2560,
    # N=50000, ~1.25% nonzero per column matching TopK's k/d_sae sparsity):
    # scipy's axis-wise tie-ranking path is less optimized than its 1-D path.
    # Keep the loop — it's the faster option here, confirmed empirically.
    from scipy.stats import rankdata

    d_sae = scores.shape[1]
    aurocs = np.empty(d_sae)

    for i in range(d_sae):
        ranks = rankdata(scores[:, i])
        rank_sum_pos = ranks[pos_mask].sum()
        U = rank_sum_pos - n_pos * (n_pos + 1) / 2
        auroc = U / (n_pos * n_neg)
        aurocs[i] = max(auroc, 1.0 - auroc)  # reflect below 0.5

    return aurocs


# ── Permuted-label control / noise-vs-confounding decomposition ──────────────
# The winner's-curse lemma proves selection-over-noisy-estimates is
# upward-biased in general; real-data inflation is a MIXTURE of
# (a) vanishing sampling-noise bias (shrinks toward 0 as n grows) and (b)
# non-vanishing structural confounding (a feature genuinely, if imprecisely,
# predictive of a broader class — does not shrink with more data). Shuffling
# protein-to-label correspondence destroys any real structural relationship
# while preserving each GO term's positive count exactly, isolating (a) in
# pure form; `(real inflation) − (permuted inflation)` then estimates (b).

def permute_go_labels(
    go_labels: dict[str, set[str]],
    protein_ids: list[str],
    random_state: int = 42,
) -> dict[str, set[str]]:
    """
    Randomly reassign each protein's ENTIRE label set to a different protein,
    breaking the true protein-identity <-> label correspondence while
    preserving the marginal label distribution exactly: every GO term's
    positive count is unchanged (we're permuting which protein holds which
    already-existing label set, not resampling labels), and per-protein
    label co-occurrence structure (e.g. two GO terms that tend to co-occur on
    the same protein) is preserved too — just reattached to a different
    protein. This is deliberately a stronger, more conservative permutation
    than reshuffling each GO term's positive/negative labels independently,
    which would destroy real co-occurrence structure the naive selection
    procedure might otherwise (spuriously) exploit.

    Proteins with no GO annotations in `go_labels` are left unlabeled (empty
    set) in the permuted output, same as in the input — permuting an absent
    key would silently invent annotations that were never there.
    """
    rng = np.random.default_rng(random_state)
    ids_with_labels = [pid for pid in protein_ids if go_labels.get(pid)]
    label_sets = [go_labels[pid] for pid in ids_with_labels]

    shuffled_order = rng.permutation(len(ids_with_labels))
    permuted = {
        ids_with_labels[i]: label_sets[shuffled_order[i]]
        for i in range(len(ids_with_labels))
    }
    return permuted


@dataclass
class DecompositionResult:
    """Noise-vs-confounding decomposition for one GO term."""
    go_id: str
    real_inflation: float | None            # single_feature_auroc - held_out_auroc, real labels
    permuted_inflation: float | None        # same quantity, permuted labels (pure noise)
    confounding_estimate: float | None      # real_inflation - permuted_inflation


def decompose_noise_from_confounding(
    real_report: GOProbingReport,
    permuted_report: GOProbingReport,
) -> list[DecompositionResult]:
    """
    Pair up real-label and permuted-label probing results by go_id and
    compute the noise-vs-confounding decomposition. Both reports must come
    from `probe_go_terms(..., compute_held_out_baselines=True)` runs (same
    min_positives / go_dag / split settings) — one on real go_labels, one on
    `permute_go_labels(...)`'s output — or `auroc_inflation` won't be
    populated and every result here will be None.

    GO terms present in only one report (can happen at the margin if
    permutation shifts a term's eligibility) are skipped rather than
    guessed at.
    """
    permuted_by_id = {r.go_id: r for r in permuted_report.results}
    out = []
    for real_r in real_report.results:
        permuted_r = permuted_by_id.get(real_r.go_id)
        if permuted_r is None:
            continue
        real_infl = real_r.auroc_inflation
        permuted_infl = permuted_r.auroc_inflation
        confounding = (
            real_infl - permuted_infl
            if real_infl is not None and permuted_infl is not None
            else None
        )
        out.append(
            DecompositionResult(
                go_id=real_r.go_id,
                real_inflation=real_infl,
                permuted_inflation=permuted_infl,
                confounding_estimate=confounding,
            )
        )
    return out


@dataclass
class MultiSeedDecompositionResult:
    """
    Noise-vs-confounding decomposition for one label (GO term / cCRE class /
    text concept), averaged over multiple independent permutation draws
    rather than read off a single fixed seed.

    A single permuted-label draw's held-out AUROC is itself a noisy point
    estimate — the smaller a label's positive count, the noisier — so one
    draw can produce a `permuted_inflation` value that looks like a real,
    reproducible pattern purely by chance (e.g. two unrelated labels
    coincidentally landing on nearly the same value from a single shared
    seed) when it's actually just single-draw sampling noise. This surfaced
    concretely in genomic cCRE probing (2026-07-15): two
    cCRE classes with the smallest positive counts (~300-800, versus
    typically thousands for protein GO terms) produced nearly identical
    permuted_inflation values (~0.148 each) under the single fixed seed
    used by decompose_noise_from_confounding — a pattern much more
    plausibly explained by high single-draw variance at low n than by any
    real shared property of those two classes.

    Averaging over many independent permutation seeds requires no new
    model forward pass — every draw reuses the SAME already-computed
    feature_acts, re-probing only against a freshly reshuffled label set —
    so this is CPU-only, the same "cheap because activations are cached"
    property bootstrap_mean_auroc_inflation already relies on.
    """

    go_id: str
    real_inflation: float | None
    mean_permuted_inflation: float | None
    permuted_inflation_ci: tuple[float, float] | None
    mean_confounding_estimate: float | None
    confounding_estimate_ci: tuple[float, float] | None
    n_seeds_successful: int


def decompose_noise_from_confounding_multi_seed(
    real_report: GOProbingReport,
    probe_fn: Callable[[dict[str, set[str]]], GOProbingReport],
    labels: dict[str, set[str]],
    ids: list[str],
    n_permutations: int = 20,
    confidence: float = 0.95,
    base_random_state: int = 123,
) -> list[MultiSeedDecompositionResult]:
    """
    Multi-seed generalization of decompose_noise_from_confounding: runs
    `n_permutations` independent permuted-label draws (each a fresh call to
    permute_go_labels with a distinct seed), re-probes each via `probe_fn`,
    and reports the MEAN and percentile CI of permuted_inflation /
    confounding_estimate per label — not a single-draw point estimate.

    Args:
        real_report: report from probing on the TRUE labels.
        probe_fn: callable taking one permuted labels dict and returning a
            GOProbingReport — typically a functools.partial binding
            probe_go_terms/probe_genomic_annotations/probe_text_concepts to
            the already-computed feature_acts and every other kwarg, so
            only the labels argument varies between calls (no new model
            forward pass per draw).
        labels, ids: the REAL label/id data, passed to permute_go_labels to
            generate each draw (the true label sets are what get shuffled,
            not the report).
        n_permutations: number of independent permutation draws.
        confidence: confidence level for the percentile CI.
        base_random_state: seeds used are base_random_state,
            base_random_state + 1, ..., base_random_state + n_permutations - 1.

    Returns:
        One MultiSeedDecompositionResult per label that appeared in at
        least one permuted draw's decomposition — a label that never
        clears min_positives under any permutation is silently absent
        (skipped, not guessed at), same convention as
        decompose_noise_from_confounding.
    """
    per_id_permuted: dict[str, list[float]] = defaultdict(list)
    per_id_confounding: dict[str, list[float]] = defaultdict(list)
    per_id_real: dict[str, float | None] = {}

    for i in range(n_permutations):
        permuted_labels = permute_go_labels(labels, ids, random_state=base_random_state + i)
        permuted_report = probe_fn(permuted_labels)
        for d in decompose_noise_from_confounding(real_report, permuted_report):
            if d.permuted_inflation is None or d.confounding_estimate is None:
                continue
            per_id_permuted[d.go_id].append(d.permuted_inflation)
            per_id_confounding[d.go_id].append(d.confounding_estimate)
            per_id_real[d.go_id] = d.real_inflation

    alpha = 1.0 - confidence
    out = []
    for go_id, permuted_vals in per_id_permuted.items():
        confounding_vals = per_id_confounding[go_id]
        lo_p, hi_p = np.percentile(permuted_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        lo_c, hi_c = np.percentile(confounding_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out.append(
            MultiSeedDecompositionResult(
                go_id=go_id,
                real_inflation=per_id_real[go_id],
                mean_permuted_inflation=float(np.mean(permuted_vals)),
                permuted_inflation_ci=(float(lo_p), float(hi_p)),
                mean_confounding_estimate=float(np.mean(confounding_vals)),
                confounding_estimate_ci=(float(lo_c), float(hi_c)),
                n_seeds_successful=len(permuted_vals),
            )
        )
    return out
