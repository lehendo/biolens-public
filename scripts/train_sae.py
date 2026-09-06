"""
Train a sparse autoencoder on pre-cached activations.

Usage:
  python scripts/train_sae.py \\
      --model esm2_8m --layer 5 --variant topk \\
      --cache /scratch/arjunc4/biolens/data/activations/esm2_8m_layer5 \\
      --output-dir /scratch/arjunc4/biolens/checkpoints/esm2_8m_layer5_topk \\
      --wandb-project biolens

On SLURM (ic-express, 8h):  see scripts/slurm/train_sae.sh
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.models.registry import ModelRegistry
from biolens.sae.architectures import SAEConfig, TopKSAE
from biolens.sae.dictionary import ActivationCache
from biolens.sae.train import SAETrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="esm2_8m")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--variant", default="topk", choices=["topk", "jumprelu", "gated"])
    p.add_argument("--cache", required=True, help="Path to ActivationCache directory")
    p.add_argument("--output-dir", required=True, help="Checkpoint output directory")
    # SAE hyperparams
    p.add_argument("--expansion-factor", type=int, default=8)
    p.add_argument("--k", type=int, default=32, help="TopK k (ignored for jumprelu/gated)")
    p.add_argument("--n-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-4)
    # Logging
    p.add_argument("--wandb-project", default="biolens")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    # Misc
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    layer = args.layer if args.layer >= 0 else model_cfg.num_layers - 1
    d_model = model_cfg.hidden_dim

    run_name = args.run_name or f"{args.model}_L{layer}_{args.variant}_k{args.k}"
    output_dir = Path(args.output_dir) / run_name

    logger.info(
        "Training %s SAE on %s layer %d: d_model=%d, expansion=%d×, k=%d",
        args.variant, args.model, layer, d_model, args.expansion_factor, args.k,
    )

    # Build SAE
    sae_cfg = SAEConfig(d_model=d_model, expansion_factor=args.expansion_factor)
    sae = TopKSAE(sae_cfg, k=args.k)  # Phase 1: use make_sae(args.variant, ...)

    # Load cache
    cache = ActivationCache(args.cache)
    logger.info("Cache: %d activation vectors, d=%d", len(cache), cache.d_model)
    assert cache.d_model == d_model, (
        f"Cache d_model {cache.d_model} ≠ model d_model {d_model}"
    )

    # Build training config
    train_cfg = TrainingConfig(
        variant=args.variant,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        wandb_project=None if args.no_wandb else args.wandb_project,
        run_name=run_name,
        checkpoint_dir=output_dir,
        device=args.device,
        seed=args.seed,
        extra_config={
            "model": args.model,
            "layer": layer,
            "expansion_factor": args.expansion_factor,
            "k": args.k,
        },
    )

    trainer = SAETrainer(sae, train_cfg)
    trainer.train_from_cache(cache)

    logger.info("Training complete.  Checkpoint saved to %s", output_dir)


if __name__ == "__main__":
    main()
