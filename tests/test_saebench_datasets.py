"""
Tests for the non-biology control-domain dataset loader
(biolens.data.saebench_datasets).

The real-network test uses the actual `LabHC/bias_in_bios` dataset (small
slice) — this is the dataset ID that SAEBench's own config gets WRONG
(cites `LabHC/bias_in_bios_class_set1`, which doesn't exist on the Hub),
caught and corrected only by actually trying to load it directly rather
than trusting that config. That correction is worth pinning here as a
regression test, not just fixing silently.
"""

from __future__ import annotations

import pytest


class TestProfessionClassCounts:
    def test_counts_across_examples(self):
        from biolens.data.saebench_datasets import profession_class_counts

        labels = {
            "a": {"profession_0"},
            "b": {"profession_0"},
            "c": {"profession_5"},
        }
        counts = profession_class_counts(labels)
        assert counts == {"profession_0": 2, "profession_5": 1}

    def test_empty_labels_gives_empty_counts(self):
        from biolens.data.saebench_datasets import profession_class_counts

        assert profession_class_counts({}) == {}


@pytest.mark.integration
class TestLoadBiasInBiosReal:
    def test_loads_real_dataset_with_expected_schema(self):
        """Regression test for SAEBench's own config citing the wrong dataset
        ID: 'LabHC/bias_in_bios_class_set1', which does not exist on
        HuggingFace — the real ID is 'LabHC/bias_in_bios', confirmed here
        by actually loading it, not by re-reading old notes."""
        from biolens.data.saebench_datasets import (
            BIAS_IN_BIOS_PROFESSIONS,
            load_bias_in_bios,
            profession_class_counts,
        )

        ids, texts, labels = load_bias_in_bios(split="test", max_examples=500)

        assert len(ids) == len(texts) == len(labels) == 500
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)
        # Every label must be one of the 28 known profession classes.
        all_labels = {lbl for label_set in labels.values() for lbl in label_set}
        valid_labels = {f"profession_{i}" for i in range(len(BIAS_IN_BIOS_PROFESSIONS))}
        assert all_labels <= valid_labels

        counts = profession_class_counts(labels)
        assert len(counts) > 1  # more than one profession class actually present in the slice
        assert sum(counts.values()) == 500

    def test_natural_class_imbalance_spans_wide_range(self):
        """Confirms the real natural-imbalance claim in the module
        docstring (911-76,748 on the full train split) holds directionally
        even on a smaller slice — some classes are much more common than
        others, which is what the min_positives sweep needs."""
        from biolens.data.saebench_datasets import load_bias_in_bios, profession_class_counts

        _, _, labels = load_bias_in_bios(split="train", max_examples=5000)
        counts = profession_class_counts(labels)

        assert max(counts.values()) > 5 * min(counts.values()), (
            f"Expected real natural class imbalance, got counts: {counts}"
        )
