# BioLens

A unified, open-source toolkit for training, evaluating, and serving sparse autoencoder (SAE) features across open biological foundation models (protein, genomic, and single-cell), with a standardized interpretability benchmark and a feature-browser dashboard.

---

## Prior art

BioLens does not claim to be first to show that SAEs work on biology. The following results are established:

- SAEs on ESM2 recover GO terms, thermostability, and subcellular localization features: Adams et al. (2024) "From Mechanistic Interpretability to Mechanistic Biology"; Simon & Zou, "InterPLM" (2024).
- ProtSAE adds semantically-guided and disentangled variants.
- Arc Institute (Brixi et al.) has trained SAEs on Evo 2, finding features for phage insertions, operon structure, exon-intron boundaries, and protein secondary structure.
- SAEs on Geneformer and scGPT find organized biological knowledge: Kendiukhov (2026) runs causal circuit tracing across these two models.
- Embedding-level convergence across scientific foundation models is established ("Universally Converging Representations of Matter," Dec 2025).

What BioLens adds: a single toolkit covering multiple model families under one API, a standardized cross-family evaluation suite, and a feature browser for inspecting and verifying claimed feature-concept mappings against real reference data.

---

## Installation

```bash
git clone https://github.com/lehendo/biolens
cd biolens
pip install -e ".[dev,borzoi]"
```

Requires Python 3.10+. `torch` is pinned to an exact version and build (`torch==2.11.0`), not a floor: see `pyproject.toml` for why.

Two extra steps are needed for a fully working environment:

1. **Install `torch` from PyTorch's own CUDA index, not plain PyPI.** Plain PyPI can serve a differently-tagged CUDA build under the same version string:
   ```bash
   pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
   ```
2. **Build `flash-attn` from source against the installed torch**, rather than using the prebuilt wheel (required if you use `biolens.models.evo2`, since Evo 2's underlying Vortex/StripedHyena2 framework depends on it directly):
   ```bash
   pip install flash-attn==2.8.3.post1 --no-build-isolation --no-binary flash-attn \
       --no-cache-dir --force-reinstall --no-deps
   ```
   This is a from-source CUDA kernel compile: budget real time (1-3+ hours) and memory (128G+), and run it on a GPU node with a CUDA toolkit available (for example `module load cuda/12.8` on a SLURM cluster).

Verify the full chain:
```bash
python -c "import torch, evo2, biolens; print(torch.__version__); print(torch.cuda.is_available())"
```

Or run all steps in order automatically: `scripts/setup_env.sh` handles the base install and both fixes above, creating a fresh conda environment if needed. On a SLURM cluster, submit it as a job (`sbatch scripts/slurm/setup_env.sh`) rather than running it on a login node, since the from-source flash-attn build can take longer than most login nodes allow.

---

## Quick start

```bash
# Full pipeline: extract activations -> train SAE -> evaluate -> dashboard
bash scripts/run_phase0.sh

# Quick sanity check (1000 sequences, 500 steps, about 5 minutes on GPU)
bash scripts/run_phase0.sh --quick
```

**On a SLURM cluster:**

```bash
# 1. Extract activations (array job, sharded)
sbatch scripts/slurm/extract_activations.sh

# 2. Finalize normalization stats after all array tasks complete
python scripts/extract_activations.py --finalize \
    --output-dir /scratch/arjunc4/biolens/data/activations/esm2_8m_layer5_mean

# 3. Train the SAE
sbatch scripts/slurm/train_sae.sh

# 4. Evaluate
python scripts/run_eval.py \
    --model esm2_8m --layer 5 \
    --sae-checkpoint /scratch/arjunc4/biolens/checkpoints/.../final.pt \
    --cache /scratch/arjunc4/biolens/data/activations/esm2_8m_layer5_mean \
    --output-dir /scratch/arjunc4/biolens/eval/esm2_8m_layer5 \
    --min-positives 100 --device cuda
```

---

## Feature investigation: verify before you trust an AUROC

A high single-feature GO-term AUROC is a candidate signal, not proof. Before treating any feature as "this encodes concept X," check its real top-activating sequences:

```bash
# Headless: no GPU, no browser, works on a login node
python scripts/inspect_features.py \
    --sae-checkpoint <checkpoint>/final.pt --cache <cache_dir> \
    --feature-idx 1103 2175 \
    --go-probing-results <eval_dir>/go_probing_results.json \
    --verified-annotations configs/verified_features/esm2_8m_layer5_topk_k32.yaml
```

Then look up the printed UniProt accessions to confirm what the top-activating proteins actually are. Once checked, record the outcome in a `configs/verified_features/*.yaml` file. Both this CLI tool and the Gradio dashboard read that registry and show a feature's real `[CONFIRMED]` / `[SPURIOUS]` / `[PLAUSIBLE]` status instead of a raw, unverified AUROC claim.

Interactive browsing (same underlying logic, `biolens.eval.feature_inspection`):

```bash
python -m biolens.dashboard.app \
    --model esm2_8m --layer 5 \
    --sae-checkpoint <checkpoint>/final.pt --cache <cache_dir> \
    --go-probing-results <eval_dir>/go_probing_results.json \
    --verified-annotations configs/verified_features/esm2_8m_layer5_topk_k32.yaml
```

On a remote cluster, add `--share` for a public Gradio link, or use SSH port-forwarding.

---

## Results

Reference run: ESM2 8M, layer 5, TopK SAE (k=32).

| Metric | Value |
|---|---|
| Reconstruction FVE | 0.970 |
| Mean L0 | 32.0 (equals k, as designed) |
| Dead features | 23.4% strict / about 4% training-time EMA |
| GO terms with AUROC > 0.70 (`min_positives=100`) | 10/13 (76.9%) |
| Features manually verified against real top-activating sequences | 7/13 confirmed or plausible |

A raw single-feature AUROC can look far stronger than it is: at a lower `--min-positives` threshold, checking claimed feature-concept matches against actual UniProt annotations found only a small fraction were genuinely specific matches, with the rest sharing only a broad structural resemblance to the claimed GO term. Raising `--min-positives` to 100 fixed this, with no other change to the model, SAE, or code. This is why feature verification (above) is enforced in the tooling rather than left as an optional step.

---

## Python API

```python
from biolens import ModelRegistry, TopKSAE, SAETrainer, TrainingConfig, ActivationCache

# Load model
registry = ModelRegistry()
model = registry.load_model("esm2_8m", device="cuda")

# Extract activations
acts = model.get_activations(["MKTAYIAKQRQISFVK..."], layer=5, pooling="mean")
# acts: (N, 320) float32 CPU tensor

# Train SAE
from biolens.sae.architectures import SAEConfig
sae = TopKSAE(SAEConfig(d_model=320, expansion_factor=8), k=32)
trainer = SAETrainer(sae, TrainingConfig(n_steps=10000, device="cuda"))
trainer.train_from_tensors(acts)

# Evaluate
from biolens import eval as beval
metrics = beval.compute_reconstruction_metrics(sae, acts)
print(metrics)  # FVE=0.94  MSE=0.0031  L0=32.0  dead=0.8%
```

---

## Architecture

```
Open bio foundation models  ->  Activation extraction  ->  SAE training
  ESM2, Evo 2, HyenaDNA,          (hooks per arch;           (TopK / JumpReLU /
  scGPT, Geneformer               token + pooled)             Gated)

                                       |
                          Standardized eval suite
                       (reconstruction, GO probing,
                        causal ablation, cross-modality
                              mediation)

                                       |
                             Feature browser dashboard
                          (multi-modal, cross-model search)
```

---

## Development

```bash
# Run fast tests (no network, no GPU)
pytest -m "not integration" tests/

# Run all tests including integration (downloads ESM2 8M weights, ~30MB)
pytest tests/

# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/
```

---

## License

MIT. See [LICENSE](LICENSE).

Model weights carry their own licenses; check each before distributing derived SAE weights.
- ESM2: MIT (Meta)
- Evo 2: Apache 2.0 (Arc Institute)
- HyenaDNA: Apache 2.0 (Hazy Research)
- Geneformer / scGPT: check current license files
