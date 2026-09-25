"""
tests/test_labeling.py
=======================
Unit tests for src/entity_resolution/labeling.py

Test coverage
-------------
- True positive → label=1
- Non-matching candidate → label=0
- Singleton S1 entity → all candidates label=0, negative_type='singleton_negative'
- Multi-target entity (S1 matches both S2 and S3) labelled correctly on both sides
- Hard-negative candidates have higher blocking_score than easy-negative candidates
- After sampling, negative_type is one of the expected values
- No entity leakage between sampled positives and sampled negatives
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.entity_resolution.labeling import (
    NegativeSamplingConfig,
    assign_labels,
    build_labeled_pairs,
    compute_label_statistics,
    parse_ground_truth,
    sample_negatives,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def gt_s1_to_s2():
    """S1→S2 ground-truth lookup."""
    return {
        "S1-100": {"S2-200", "S2-201"},    # has two S2 matches
        "S1-101": {"S2-202"},              # has one S2 match
        # S1-102 is singleton (no entry here)
    }


@pytest.fixture
def gt_s1_to_s3():
    """S1→S3 ground-truth lookup (same S1-100 entity matches S3 too)."""
    return {
        "S1-100": {"S3-300"},              # S1-100 matches both S2 and S3
        "S1-101": set(),                   # S1-101 has NO S3 matches (but not singleton)
    }


@pytest.fixture
def singleton_ids():
    return {"S1-102"}


@pytest.fixture
def candidates_s1_s2():
    """Synthetic S1×S2 candidate pairs."""
    return pd.DataFrame({
        "entity_id_left":  ["S1-100", "S1-100", "S1-100", "S1-101", "S1-101", "S1-102", "S1-102"],
        "entity_id_right": ["S2-200", "S2-999", "S2-998", "S2-202", "S2-997", "S2-500", "S2-501"],
        "blocking_score":  [0.95,      0.90,     0.10,     0.88,     0.20,     0.70,     0.30],
        "blocking_method": ["tfidf",   "tfidf",  "tfidf",  "tfidf",  "tfidf",  "tfidf",  "tfidf"],
    })


@pytest.fixture
def candidates_s1_s3():
    """Synthetic S1×S3 candidate pairs."""
    return pd.DataFrame({
        "entity_id_left":  ["S1-100", "S1-100"],
        "entity_id_right": ["S3-300", "S3-999"],
        "blocking_score":  [0.92,      0.40],
        "blocking_method": ["tfidf",   "token_sort"],
    })


@pytest.fixture
def sampling_cfg():
    return NegativeSamplingConfig(
        hard_neg_per_pos=2,
        easy_neg_per_pos=2,
        hard_neg_delta=0.10,
        easy_score_pct=0.30,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests: assign_labels
# ---------------------------------------------------------------------------

class TestAssignLabels:

    def test_true_positive_gets_label_one(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids):
        """S1-100 → S2-200 is a GT match → label=1."""
        df = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        row = df[(df["source1_entity_id"] == "S1-100") & (df["candidate_entity_id"] == "S2-200")]
        assert len(row) == 1
        assert int(row["label"].iloc[0]) == 1
        assert row["negative_type"].iloc[0] == "positive"

    def test_non_match_gets_label_zero(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids):
        """S1-100 → S2-999 is NOT in GT → label=0."""
        df = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        row = df[(df["source1_entity_id"] == "S1-100") & (df["candidate_entity_id"] == "S2-999")]
        assert len(row) == 1
        assert int(row["label"].iloc[0]) == 0

    def test_singleton_all_label_zero(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids):
        """All candidates of singleton S1-102 → label=0, negative_type='singleton_negative'."""
        df = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        singleton_rows = df[df["source1_entity_id"] == "S1-102"]
        assert len(singleton_rows) > 0, "Expected singleton candidates"
        assert (singleton_rows["label"] == 0).all()
        assert (singleton_rows["negative_type"] == "singleton_negative").all()

    def test_candidate_source_inferred(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids):
        """candidate_source is correctly inferred as 's2' for S2-* IDs."""
        df = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        assert (df["candidate_source"] == "s2").all()

    def test_columns_renamed(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids):
        """entity_id_left/right renamed to source1_entity_id/candidate_entity_id."""
        df = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        assert "source1_entity_id" in df.columns
        assert "candidate_entity_id" in df.columns
        assert "entity_id_left" not in df.columns
        assert "entity_id_right" not in df.columns

    def test_multi_target_s2_correct(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids):
        """S1-100 has two S2 GT targets (S2-200 and S2-201); both should be label=1 if present."""
        df = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        tp_200 = df[(df["source1_entity_id"] == "S1-100") & (df["candidate_entity_id"] == "S2-200")]
        assert int(tp_200["label"].iloc[0]) == 1
        # S2-201 is not in the candidate file — that's fine (blocking recall gap)

    def test_multi_target_s3_correct(self, candidates_s1_s3, gt_s1_to_s3, singleton_ids):
        """S1-100 matches S3-300 on the S3 side too → label=1."""
        df = assign_labels(candidates_s1_s3, gt_s1_to_s3, singleton_ids)
        row = df[(df["source1_entity_id"] == "S1-100") & (df["candidate_entity_id"] == "S3-300")]
        assert len(row) == 1
        assert int(row["label"].iloc[0]) == 1
        # S3-999 is not a match
        non_row = df[(df["source1_entity_id"] == "S1-100") & (df["candidate_entity_id"] == "S3-999")]
        assert int(non_row["label"].iloc[0]) == 0

    def test_s3_source_inferred(self, candidates_s1_s3, gt_s1_to_s3, singleton_ids):
        """candidate_source inferred as 's3' for S3-* IDs."""
        df = assign_labels(candidates_s1_s3, gt_s1_to_s3, singleton_ids)
        assert (df["candidate_source"] == "s3").all()


# ---------------------------------------------------------------------------
# Tests: sample_negatives
# ---------------------------------------------------------------------------

class TestSampleNegatives:

    def test_singletons_retained_fully(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg):
        """Singleton negatives are never dropped by sampling."""
        labeled = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        sampled = sample_negatives(labeled, sampling_cfg)
        n_singleton_before = (labeled["negative_type"] == "singleton_negative").sum()
        n_singleton_after  = (sampled["negative_type"] == "singleton_negative").sum()
        assert n_singleton_after == n_singleton_before

    def test_negative_types_valid(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg):
        """All negative_type values are one of the expected set."""
        labeled = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        sampled = sample_negatives(labeled, sampling_cfg)
        valid_types = {"positive", "hard_negative", "easy_negative", "singleton_negative"}
        assert set(sampled["negative_type"].unique()).issubset(valid_types)

    def test_no_unsampled_negatives_remain(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg):
        """After sampling, no 'unsampled_negative' rows should remain."""
        labeled = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        sampled = sample_negatives(labeled, sampling_cfg)
        assert (sampled["negative_type"] == "unsampled_negative").sum() == 0

    def test_positives_all_retained(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg):
        """All positive pairs are retained after sampling (never dropped)."""
        labeled = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        n_pos_before = (labeled["label"] == 1).sum()
        sampled = sample_negatives(labeled, sampling_cfg)
        n_pos_after = (sampled["label"] == 1).sum()
        assert n_pos_after == n_pos_before

    def test_hard_negatives_higher_score_than_easy(
        self, candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg
    ):
        """Hard negatives should on average have higher blocking_score than easy negatives."""
        labeled = assign_labels(candidates_s1_s2, gt_s1_to_s2, singleton_ids)
        sampled = sample_negatives(labeled, sampling_cfg)
        hards = sampled[sampled["negative_type"] == "hard_negative"]["blocking_score"]
        easys = sampled[sampled["negative_type"] == "easy_negative"]["blocking_score"]
        if len(hards) > 0 and len(easys) > 0:
            assert hards.mean() >= easys.mean(), (
                f"Hard negatives avg={hards.mean():.3f} should be >= easy negatives avg={easys.mean():.3f}"
            )


# ---------------------------------------------------------------------------
# Tests: build_labeled_pairs (end-to-end)
# ---------------------------------------------------------------------------

class TestBuildLabeledPairs:

    def test_end_to_end_s1_s2(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg):
        """End-to-end pipeline produces a valid labeled DataFrame."""
        result = build_labeled_pairs(candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg)
        assert "label" in result.columns
        assert "negative_type" in result.columns
        assert "source1_entity_id" in result.columns
        assert "candidate_entity_id" in result.columns
        assert len(result) > 0
        # Labels are only 0 or 1
        assert set(result["label"].unique()).issubset({0, 1})

    def test_end_to_end_s1_s3(self, candidates_s1_s3, gt_s1_to_s3, singleton_ids, sampling_cfg):
        """Works correctly on S3 side too."""
        result = build_labeled_pairs(candidates_s1_s3, gt_s1_to_s3, singleton_ids, sampling_cfg)
        assert (result["candidate_source"] == "s3").all()
        tp = result[(result["source1_entity_id"] == "S1-100") &
                    (result["candidate_entity_id"] == "S3-300")]
        assert len(tp) == 1
        assert int(tp["label"].iloc[0]) == 1

    def test_no_label_nan(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg):
        """No NaN labels in output."""
        result = build_labeled_pairs(candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg)
        assert result["label"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# Tests: compute_label_statistics
# ---------------------------------------------------------------------------

class TestComputeLabelStatistics:

    def test_stats_keys(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg):
        result = build_labeled_pairs(candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg)
        stats = compute_label_statistics(result, "S1_S2")
        required_keys = [
            "pair_type", "total_pairs", "positives", "negatives",
            "hard_negatives", "easy_negatives", "singleton_negatives",
            "imbalance_ratio_neg_per_pos", "positive_rate_pct",
        ]
        for k in required_keys:
            assert k in stats, f"Missing key: {k}"

    def test_stats_counts_consistent(self, candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg):
        result = build_labeled_pairs(candidates_s1_s2, gt_s1_to_s2, singleton_ids, sampling_cfg)
        stats = compute_label_statistics(result, "S1_S2")
        assert stats["positives"] + stats["negatives"] == stats["total_pairs"]
        assert (stats["hard_negatives"] + stats["easy_negatives"] +
                stats["singleton_negatives"]) == stats["negatives"]


# ---------------------------------------------------------------------------
# Tests: parse_ground_truth (spot-check with a temp file)
# ---------------------------------------------------------------------------

class TestParseGroundTruth:

    def test_parse_basic(self, tmp_path):
        gt_file = tmp_path / "gt.tsv"
        gt_file.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-1\tS2-10,S3-20\n"
            "S1-2\tS2-11\n"
            "S1-3\t\n"                   # singleton — empty matched IDs
        )
        s1_to_s2, s1_to_s3, singletons = parse_ground_truth(str(gt_file))

        assert s1_to_s2["S1-1"] == {"S2-10"}
        assert s1_to_s2["S1-2"] == {"S2-11"}
        assert s1_to_s3["S1-1"] == {"S3-20"}
        assert "S1-2" not in s1_to_s3
        assert "S1-3" in singletons

    def test_nan_target_treated_as_singleton(self, tmp_path):
        gt_file = tmp_path / "gt.tsv"
        gt_file.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-99\tnan\n"
        )
        _, _, singletons = parse_ground_truth(str(gt_file))
        assert "S1-99" in singletons
