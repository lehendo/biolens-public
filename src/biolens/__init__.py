"""
BioLens — SAE interpretability toolkit for biological foundation models.

Phase 0: ESM2 protein models, TopK SAE, reconstruction + GO-term probing.
Phase 1: Multi-model coverage (Evo 2, HyenaDNA, Geneformer), JumpReLU/Gated SAE variants.
Phase 2: Cross-modality causal mediation study (DNA→expression via GTEx eQTLs).
Phase 3: Public dashboard, pretrained weights on HuggingFace Hub.
"""

from biolens.models.registry import ModelRegistry
from biolens.sae.architectures import TopKSAE
from biolens.sae.dictionary import ActivationCache
from biolens.sae.train import SAETrainer, TrainingConfig

__version__ = "0.1.0"
__all__ = [
    "ModelRegistry",
    "TopKSAE",
    "ActivationCache",
    "SAETrainer",
    "TrainingConfig",
]
