"""
Non-biological concept-labeled datasets, used as a non-biology control
domain. Paired with Gemma Scope SAEs (biolens.models.gemma_scope) via the
same `probe_go_terms` sweep engine used for GO-term and ENCODE cCRE
probing — the engine is domain-agnostic; only the label source changes.

Dataset schema verified directly against the real HuggingFace dataset
(2026-07-03), not assumed from SAEBench's own config alone: `LabHC/
bias_in_bios` has columns `hard_text` (biography text), `profession`
(integer class, 28 professions), `gender` (0/1) — NOT the dataset ID
`LabHC/bias_in_bios_class_set1` referenced in SAEBench's own config, which
does not exist on the Hub; that config uses SAEBench's *internal*
per-class-subset naming convention, not the real underlying dataset's
actual HuggingFace repo ID — corrected here after direct verification.
Natural per-class counts range from 911 to 76,748 (confirmed directly),
comfortably spanning and exceeding this project's planned `min_positives`
sweep range (10-2000) without needing the controlled-subsampling step to
be load-bearing for feasibility — though that step remains valuable for
isolating concept-difficulty confounds, matching the existing GO-term
methodology.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BIAS_IN_BIOS_DATASET_ID = "LabHC/bias_in_bios"

# The 28 profession classes, in label-index order, per the Bias-in-Bios
# corpus (De-Arteaga et al. 2019, "Bias in Bios") — used only for
# human-readable names in reporting; the numeric label itself (0-27) is
# what's actually used for probing.
BIAS_IN_BIOS_PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]


def load_bias_in_bios(
    split: str = "train",
    max_examples: int | None = None,
) -> tuple[list[str], list[str], dict[str, set[str]]]:
    """
    Load the Bias-in-Bios profession-classification dataset in the same
    (ids, sequences, labels) shape biolens.eval.probing.probe_go_terms /
    probe_genomic_annotations expect (protein_ids/window_ids, sequences,
    go_labels/dna_labels) — the sweep engine is domain-agnostic, this just
    supplies a non-biological instance of its expected input shape.

    Args:
        split: "train" (257,478 examples), "test" (99,069), or "dev" (39,642).
        max_examples: Optional cap, for fast local iteration.

    Returns:
        ids:      "biobio_{i}" for each example (no natural stable ID in
                  the source dataset).
        texts:    The biography text for each example.
        labels:   id -> {"profession_{class}"} — a single-label set per
                  example, in the same dict-of-sets shape as multi-label GO
                  annotations (a text could in principle be given more than
                  one profession label by construction here, though the
                  underlying dataset is single-label).
    """
    from datasets import load_dataset

    ds = load_dataset(_BIAS_IN_BIOS_DATASET_ID, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    ids = [f"biobio_{i}" for i in range(len(ds))]
    texts = list(ds["hard_text"])
    labels = {
        ids[i]: {f"profession_{profession}"} for i, profession in enumerate(ds["profession"])
    }

    logger.info(
        "Loaded %d Bias-in-Bios examples (%s split), %d profession classes",
        len(ids), split, len(BIAS_IN_BIOS_PROFESSIONS),
    )
    return ids, texts, labels


def profession_class_counts(labels: dict[str, set[str]]) -> dict[str, int]:
    """Per-class positive counts across the loaded label dict — the
    natural-imbalance structure the min_positives sweep needs (see module
    docstring: confirmed range 911-76,748 on the full train split)."""
    counts: dict[str, int] = {}
    for label_set in labels.values():
        for label in label_set:
            counts[label] = counts.get(label, 0) + 1
    return counts
