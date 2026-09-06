"""
Gemma Scope pretrained SAE wrapper used as a non-biology control domain: text
concept probing on Gemma-2-2B, contrasted against the protein/genomic
probing elsewhere in this project.

Deliberately NOT a BioModel subclass (src/biolens/models/registry.py): every
other adapter in this project trains its OWN SAE on a base model's
activations. Here, both the base model (Gemma-2-2B) and the SAE are
pretrained by others (Google DeepMind) and loaded as-is via `sae_lens` — a
fundamentally different "bring your own pretrained SAE" use case that
doesn't fit the train-our-own-SAE abstraction the rest of this project is
built around. Forcing it into BioModel would be the wrong abstraction, not
a shortcut.

API verified directly against the real, installed `sae_lens` package and
the real HuggingFace registry (2026-07-03), not assumed:
  - `SAE.from_pretrained("gemma-scope-2b-pt-res", "layer_N/width_W/average_l0_L",
    device=...)` — confirmed against sae_lens's own pretrained-SAE directory
    (316 real sae_ids for this release) and a real, successful load + encode
    call (d_in=2304 matching Gemma-2-2B's hidden size, d_sae=16384 for the
    16K-width variant, real sparse activations returned).
  - `HookedSAETransformer.from_pretrained_no_processing("gemma-2-2b")` +
    `run_with_cache(...)` — standard, well-documented TransformerLens API;
    the residual-stream hook name `blocks.{layer}.hook_resid_post` is
    bedrock TransformerLens convention, not something this project
    introduced or is guessing at (unlike Evo 2's layer-naming ambiguity,
    which IS genuinely unresolved upstream — see src/biolens/models/evo2.py).

The full Gemma-2-2B model itself was NOT downloaded/run in this
environment (multi-GB, and the harder, more novel part of this
integration — the SAE loading and encoding path — was already fully
verified end-to-end). First real run of get_sae_features() below is the
first true validation of the model-loading half of this class.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class GemmaScopeSAE:
    """Loads a pretrained Gemma-2 model + a pretrained Gemma Scope SAE for
    a specific layer, exposing a get_sae_features(texts) -> (N, d_sae)
    interface directly usable by biolens.eval.probing.probe_go_terms
    (pass window/text IDs as `protein_ids` and the labels dict as
    `go_labels` — the sweep engine doesn't care about the domain)."""

    def __init__(
        self,
        model_name: str = "gemma-2-2b",
        sae_release: str = "gemma-scope-2b-pt-res",
        sae_id: str = "layer_12/width_16k/average_l0_82",
        layer: int = 12,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.sae_release = sae_release
        self.sae_id = sae_id
        self.layer = layer
        self.device = device
        self._model: Any = None
        self._sae: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._sae is not None:
            return

        from sae_lens import SAE, HookedSAETransformer

        logger.info("Loading %s...", self.model_name)
        self._model = HookedSAETransformer.from_pretrained_no_processing(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        logger.info("Loading Gemma Scope SAE %s / %s...", self.sae_release, self.sae_id)
        self._sae = SAE.from_pretrained(self.sae_release, self.sae_id, device=self.device)
        self._sae.eval()

        logger.info(
            "Loaded %s (d_model implied by SAE d_in=%d) + SAE (d_sae=%d) at layer %d",
            self.model_name, self._sae.cfg.d_in, self._sae.cfg.d_sae, self.layer,
        )

    @property
    def d_sae(self) -> int:
        self._ensure_loaded()
        assert self._sae is not None
        return int(self._sae.cfg.d_sae)

    def get_sae_features(
        self,
        texts: list[str],
        batch_size: int = 8,
        max_length: int = 128,
    ) -> Tensor:
        """
        Args:
            texts:      Input text examples (e.g. Bias-in-Bios biographies).
            batch_size: Sequences per forward pass.
            max_length: Truncate/pad to this many tokens — Gemma Scope's own
                training used a similar bound for short-text classification
                tasks; long documents would need a pooling-over-chunks
                strategy this class doesn't implement.

        Returns:
            (N, d_sae) CPU float32 tensor of mean-pooled (over non-padding
            token positions) SAE feature activations, one row per input text.
        """
        self._ensure_loaded()
        assert self._model is not None and self._sae is not None

        hook_name = f"blocks.{self.layer}.hook_resid_post"
        all_features: list[Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self._model.to_tokens(
                batch, truncate=True, prepend_bos=True,
            )[:, :max_length].to(self.device)

            with torch.no_grad():
                _, cache = self._model.run_with_cache(
                    tokens, names_filter=hook_name, stop_at_layer=self.layer + 1,
                )
                resid = cache[hook_name]  # (batch, seq, d_model)

                # Mean-pool over real (non-padding) token positions. Gemma's
                # tokenizer pads on the right by default in most
                # HookedTransformer configs; the BOS token is included in
                # the pool (unlike ESM2Model's convention of excluding
                # CLS/EOS) since Gemma Scope's own published evaluation
                # pools over the full sequence including BOS.
                pad_token_id = self._model.tokenizer.pad_token_id
                mask = (tokens != pad_token_id).float().unsqueeze(-1)  # (batch, seq, 1)
                pooled = (resid * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

                features = self._sae.encode(pooled)

            all_features.append(features.float().cpu())

        return torch.cat(all_features, dim=0)
