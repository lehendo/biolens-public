"""
Tests for the Gemma Scope pretrained-SAE wrapper (biolens.models.gemma_scope).

The full Gemma-2-2B model is a multi-GB download this environment doesn't
fetch — the model-loading half is tested here with a fake `sae_lens` module
injected via sys.modules (same pattern as test_models.py's TestEvo2Adapter),
verifying the pooling/hook-name/batching logic. The SAE-loading half
(`SAE.from_pretrained`) was independently verified against the REAL Gemma
Scope release in this session (real d_in=2304, d_sae=16384, real sparse
encode() output) — see the module docstring in
src/biolens/models/gemma_scope.py for that verification record; it isn't
re-mocked away here, it's a documented fact this test suite relies on
without re-deriving.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch


@pytest.fixture
def fake_sae_lens(monkeypatch):
    """Inject a fake `sae_lens` module so GemmaScopeSAE._ensure_loaded's
    deferred imports succeed without downloading the real 2B-parameter
    model."""
    fake_module = types.ModuleType("sae_lens")

    class FakeTokenizer:
        pad_token_id = 0

    class FakeSAEConfig:
        d_in = 8
        d_sae = 32

    class FakeSAE:
        def __init__(self):
            self.cfg = FakeSAEConfig()
            self.last_encode_input = None

        def eval(self):
            return self

        def encode(self, x):
            self.last_encode_input = x
            # Deterministic fake sparse output: identity-ish projection
            # padded/truncated to d_sae, so tests can check shape and that
            # real (non-garbage) input reached this point.
            out = torch.zeros(x.shape[0], self.cfg.d_sae)
            out[:, : x.shape[1]] = x
            return out

        @classmethod
        def from_pretrained(cls, release, sae_id, device="cpu"):
            instance = cls()
            instance.release = release
            instance.sae_id = sae_id
            return instance

    class FakeHookedSAETransformer:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.last_run_with_cache_kwargs = None

        def to(self, device):
            return self

        def eval(self):
            return self

        def to_tokens(self, texts, truncate=True, prepend_bos=True):
            # Fake tokenization: each text -> a fixed-length sequence of
            # token ids, padded with pad_token_id (0).
            max_len = 6
            batch = []
            for i, _text in enumerate(texts):
                real_len = 3 + (i % 2)  # vary length slightly, like real text would
                seq = [1] * real_len + [0] * (max_len - real_len)
                batch.append(seq)
            return torch.tensor(batch, dtype=torch.long)

        def run_with_cache(self, tokens, names_filter=None, stop_at_layer=None):
            self.last_run_with_cache_kwargs = {
                "names_filter": names_filter, "stop_at_layer": stop_at_layer,
            }
            batch, seq = tokens.shape
            d_model = 8
            resid = torch.ones(batch, seq, d_model)
            # Zero out padding positions so mean-pooling correctness is
            # actually exercised (non-padding positions contribute 1s).
            pad_mask = tokens == 0
            resid[pad_mask] = 999.0  # sentinel: must NOT survive pooling if masked correctly
            return None, {names_filter: resid}

        @classmethod
        def from_pretrained_no_processing(cls, model_name):
            return cls()

    fake_module.SAE = FakeSAE
    fake_module.HookedSAETransformer = FakeHookedSAETransformer
    monkeypatch.setitem(sys.modules, "sae_lens", fake_module)
    return fake_module


class TestGemmaScopeSAE:
    def test_lazy_loading(self, fake_sae_lens):
        from biolens.models.gemma_scope import GemmaScopeSAE

        wrapper = GemmaScopeSAE(layer=3, device="cpu")
        assert wrapper._model is None
        assert wrapper._sae is None
        wrapper._ensure_loaded()
        assert wrapper._model is not None
        assert wrapper._sae is not None

    def test_d_sae_property(self, fake_sae_lens):
        from biolens.models.gemma_scope import GemmaScopeSAE

        wrapper = GemmaScopeSAE(layer=3, device="cpu")
        assert wrapper.d_sae == 32  # FakeSAEConfig.d_sae

    def test_uses_correct_hook_name_for_layer(self, fake_sae_lens):
        from biolens.models.gemma_scope import GemmaScopeSAE

        wrapper = GemmaScopeSAE(layer=7, device="cpu")
        wrapper.get_sae_features(["some text"])
        assert wrapper._model.last_run_with_cache_kwargs["names_filter"] == "blocks.7.hook_resid_post"
        assert wrapper._model.last_run_with_cache_kwargs["stop_at_layer"] == 8

    def test_output_shape(self, fake_sae_lens):
        from biolens.models.gemma_scope import GemmaScopeSAE

        wrapper = GemmaScopeSAE(layer=3, device="cpu")
        features = wrapper.get_sae_features(["text one", "text two", "text three"])
        assert features.shape == (3, 32)  # (N, d_sae)

    def test_padding_excluded_from_mean_pool(self, fake_sae_lens):
        """The fake resid tensor sets padding positions to 999.0 specifically
        so this test fails loudly if the pooling mask is wrong — real
        (non-padding) positions are all 1.0, so a correctly-masked mean
        pool must equal exactly 1.0 in every dimension, not something
        contaminated by the 999.0 sentinel."""
        from biolens.models.gemma_scope import GemmaScopeSAE

        wrapper = GemmaScopeSAE(layer=3, device="cpu")
        # Bypass the SAE encode step to inspect the pooled representation
        # directly: FakeSAE.encode copies its input into the first columns
        # of the output, so we can recover the pooled vector from there.
        features = wrapper.get_sae_features(["a"])
        pooled_recovered = features[0, :8]  # d_model=8 in the fake
        assert torch.allclose(pooled_recovered, torch.ones(8)), (
            f"Padding contaminated the pooled representation: {pooled_recovered}"
        )

    def test_batching_processes_all_inputs(self, fake_sae_lens):
        from biolens.models.gemma_scope import GemmaScopeSAE

        wrapper = GemmaScopeSAE(layer=3, device="cpu")
        features = wrapper.get_sae_features([f"text {i}" for i in range(10)], batch_size=3)
        assert features.shape == (10, 32)


@pytest.mark.integration
class TestGemmaScopeSAERealSaeLoad:
    def test_real_sae_loads_and_encodes(self):
        """Real (network) test of the SAE-loading half only — the harder,
        more novel part of this integration, already independently
        verified in this session (real d_in=2304, d_sae=16384). Does NOT
        load the full 2B-parameter model, only the small SAE checkpoint."""
        from sae_lens import SAE

        sae = SAE.from_pretrained(
            "gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_82", device="cpu"
        )
        assert sae.cfg.d_in == 2304
        assert sae.cfg.d_sae == 16384

        x = torch.randn(2, sae.cfg.d_in)
        z = sae.encode(x)
        assert z.shape == (2, 16384)
        assert (z >= 0).all()  # JumpReLU features are non-negative
