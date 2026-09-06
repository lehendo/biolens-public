"""
Causal ablation / steering eval (Phase 1).

Tests whether clamping a feature on or off shifts model output toward/away from
the associated biological concept.  This test is MANDATORY before claiming a
feature is "interpretable" — feature-concept correlation does not guarantee
causal effect (established in the antibody LM SAE paper; cited in design doc).

Phase 0: stubs only.
Phase 1: implement for all four model families using the same intervention API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor


@dataclass
class CausalAblationConfig:
    n_steering_sequences: int = 100
    top_k_features: int = 10
    clamp_value: float = 20.0     # activation value to force during patching
    batch_size: int = 16


@dataclass
class CausalAblationResult:
    feature_idx: int
    go_term: str
    baseline_score: float          # model output score for concept without intervention
    clamped_on_score: float        # score after forcing feature active
    clamped_off_score: float       # score after zeroing feature
    effect_size: float             # |clamped_on - clamped_off| / baseline_std


def run_causal_ablation(
    sae,
    model,
    sequences: list[str],
    feature_indices: list[int],
    concept_scorer: Callable[[Tensor], Tensor],
    cfg: CausalAblationConfig | None = None,
    layer: int = -1,
) -> list[CausalAblationResult]:
    """
    Phase 1 — not yet implemented.

    For each (sequence, feature_idx) pair:
      1. Run forward pass, record baseline concept score.
      2. Clamp feature to clamp_value, re-run, record score.
      3. Clamp feature to 0, re-run, record score.
      4. Compute effect size.

    This requires architecture-specific activation patching hooks,
    which are built in Phase 1 alongside the multi-model registry adapters.
    See activation_hooks.py for the hook infrastructure.
    """
    raise NotImplementedError(
        "Causal ablation is Phase 1.  "
        "The hook infrastructure is in biolens.models.activation_hooks; "
        "this function will use it to patch activations mid-forward-pass."
    )
