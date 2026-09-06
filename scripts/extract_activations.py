"""
Extract and cache residual-stream activations from a biological foundation model.

Designed for SLURM array jobs: each task handles a disjoint slice of the dataset.
Writes shards atomically and registers them in a filelock-protected manifest.

Usage (single process):
  python scripts/extract_activations.py \\
      --model esm2_8m --layer 5 --pooling mean \\
      --output-dir /scratch/arjunc4/biolens/data/activations/esm2_8m_layer5 \\
      --data-source swissprot --max-seq-len 1024

Usage (SLURM array, see scripts/slurm/extract_activations.sh):
  python scripts/extract_activations.py \\
      --model esm2_8m --layer 5 --pooling mean \\
      --output-dir ... \\
      --shard-id $SLURM_ARRAY_TASK_ID --n-shards $N_SHARDS

After all shards are written, run:
  python scripts/extract_activations.py --finalize \\
      --output-dir /path/to/cache
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path so biolens is importable without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biolens.data.reference_genome import (
    extract_window_sequences,
    generate_tiling_windows,
    load_reference_genome,
)
from biolens.data.uniprot import load_swissprot
from biolens.models.registry import ModelRegistry
from biolens.sae.dictionary import ActivationCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="esm2_8m", help="Model name in configs/models.yaml")
    p.add_argument("--layer", type=int, default=-1, help="Layer index (-1 = last layer)")
    p.add_argument("--pooling", default="mean", choices=["mean", "cls", "none"])
    p.add_argument("--output-dir", required=False, help="Cache directory path")
    p.add_argument("--data-source", default="swissprot", choices=["swissprot", "genomic"])
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32, help="Sequences per GPU forward pass")
    # --data-source genomic uses the Evo 2 genomic-annotation probing path
    p.add_argument(
        "--reference-genome", default=None,
        help="Path to an indexed reference genome FASTA (required for --data-source genomic)",
    )
    p.add_argument("--window-size", type=int, default=4096, help="DNA window length in bp")
    p.add_argument(
        "--window-stride", type=int, default=4096,
        help="Stride between window starts (== window-size for non-overlapping tiling)",
    )
    p.add_argument(
        "--chroms", default=None,
        help="Comma-separated chromosome list (e.g. 'chr1,chr2'); default = every "
             "sequence in the reference FASTA",
    )
    # SLURM array sharding
    p.add_argument("--shard-id", type=int, default=0, help="This task's shard index")
    p.add_argument("--n-shards", type=int, default=1, help="Total number of shards")
    # Finalization
    p.add_argument("--finalize", action="store_true", help="Compute norm stats; run after all shards written")
    # Device
    p.add_argument("--device", default="cuda")
    # Max sequences (for dev/testing)
    p.add_argument("--max-seqs", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.finalize:
        if not args.output_dir:
            raise ValueError("--output-dir required for --finalize")
        cache = ActivationCache(args.output_dir)
        manifest = cache.finalize()
        logger.info(
            "Finalization complete: %d total vectors, scale=%.4f",
            manifest["norm"]["n_total"], manifest["norm"]["scale"],
        )
        return

    if not args.output_dir:
        raise ValueError("--output-dir required")

    # ── Load model ────────────────────────────────────────────────────────────
    registry = ModelRegistry()
    model_cfg = registry.get_config(args.model)
    layer = args.layer if args.layer >= 0 else model_cfg.num_layers - 1
    logger.info(
        "Extracting %s layer %d (%s pooling) → shard %d/%d",
        args.model, layer, args.pooling, args.shard_id, args.n_shards,
    )

    model = registry.load_model(args.model, device=args.device)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading %s sequences...", args.data_source)
    if args.data_source == "swissprot":
        ids, sequences = load_swissprot(
            max_seqs=args.max_seqs,
            max_seq_len=args.max_seq_len,
        )
    elif args.data_source == "genomic":
        if not args.reference_genome:
            raise ValueError("--reference-genome required for --data-source genomic")
        fasta = load_reference_genome(args.reference_genome)
        chroms = args.chroms.split(",") if args.chroms else None
        windows = generate_tiling_windows(
            fasta, window_size=args.window_size, stride=args.window_stride, chroms=chroms,
        )
        if args.max_seqs is not None:
            windows = windows[: args.max_seqs]
        window_seqs = extract_window_sequences(fasta, windows)
        ids = [w[0] for w in windows]
        sequences = [window_seqs[wid] for wid in ids]
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    # Assign this shard's slice
    shard_ids_seq = _shard_slice(ids, sequences, args.shard_id, args.n_shards)
    if not shard_ids_seq:
        logger.info("Shard %d: nothing to do (empty slice).", args.shard_id)
        return

    shard_protein_ids = [pid for pid, _ in shard_ids_seq]
    shard_sequences = [seq for _, seq in shard_ids_seq]
    logger.info("Shard %d: %d sequences", args.shard_id, len(shard_sequences))

    # ── Extract activations ───────────────────────────────────────────────────
    all_acts = model.get_activations(
        shard_sequences,
        layer=layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
    )
    logger.info("Extracted activations: shape %s", tuple(all_acts.shape))

    # ── Write shard ───────────────────────────────────────────────────────────
    metadata = {
        "model": args.model,
        "layer": layer,
        "pooling": args.pooling,
        "data_source": args.data_source,
    }
    if args.data_source == "genomic":
        metadata.update(
            {
                "reference_genome": args.reference_genome,
                "window_size": args.window_size,
                "window_stride": args.window_stride,
                "chroms": args.chroms,
            }
        )

    cache = ActivationCache(args.output_dir)
    cache.write_shard(
        shard_id=args.shard_id,
        activations=all_acts,
        ids=shard_protein_ids,
        sequences=shard_sequences,
        metadata=metadata,
    )
    logger.info("Shard %d written successfully.", args.shard_id)


def _shard_slice(
    ids: list[str], sequences: list[str], shard_id: int, n_shards: int
) -> list[tuple[str, str]]:
    """Return the subset of (id, seq) pairs assigned to this shard."""
    pairs = list(zip(ids, sequences))
    size = len(pairs)
    shard_size = (size + n_shards - 1) // n_shards
    start = shard_id * shard_size
    end = min(start + shard_size, size)
    return pairs[start:end]


if __name__ == "__main__":
    main()
