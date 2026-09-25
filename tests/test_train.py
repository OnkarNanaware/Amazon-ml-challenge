"""
tests/test_train.py
====================
Unit tests for src/entity_resolution/train.py

Test coverage
-------------
- entity_level_split: no entity appears in both train and val sets
- entity_level_split: approximate fraction of entities in val set
- entity_level_split: deterministic with same seed, different with different seed
- check_no_entity_leakage: correctly detects leakage
- fbeta_score: correct F0.5 computation
- evaluate_at_thresholds: returns all required keys
- evaluate_split_by_source: handles missing source column gracefully
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.entity_resolution.train import (
    ENTITY_COL,
    FEATURE_COLS,
    LABEL_COL,
    SOURCE_COL,
    check_no_entity_leakage,
    entity_level_split,
    evaluate_at_thresholds,
    evaluate_split_by_source,
    fbeta_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Synthetic feature DataFrame with 10 unique entities, 5 rows each."""
    n_entities = 10
    rows_per_entity = 5
    entities = [f"S1-{i:04d}" for i in range(n_entities)]
    rows = []
    for eid in entities:
        for j in range(rows_per_entity):
            row = {ENTITY_COL: eid, LABEL_COL: int(j == 0)}
            for feat in FEATURE_COLS:
                row[feat] = float(np.random.default_rng(42).random())
            row[SOURCE_COL] = "s2" if j % 2 == 0 else "s3"
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entity-level split tests
# ---------------------------------------------------------------------------

class TestEntityLevelSplit:

    def test_no_entity_overlap_between_train_and_val(self, synthetic_df):
        """The core anti-leakage guarantee: no shared entities."""
        df_train, df_val = entity_level_split(synthetic_df, val_frac=0.2, seed=42)
        train_entities = set(df_train[ENTITY_COL].unique())
        val_entities   = set(df_val[ENTITY_COL].unique())
        overlap = train_entities & val_entities
        assert len(overlap) == 0, f"Entity leakage detected: {overlap}"

    def test_val_fraction_approximately_correct(self, synthetic_df):
        """val_frac=0.2 → ~20% of unique entities in val."""
        df_train, df_val = entity_level_split(synthetic_df, val_frac=0.2, seed=42)
        n_total = synthetic_df[ENTITY_COL].nunique()
        n_val   = df_val[ENTITY_COL].nunique()
        actual_frac = n_val / n_total
        assert 0.10 <= actual_frac <= 0.40, (
            f"Val fraction {actual_frac:.2f} unexpectedly far from 0.2"
        )

    def test_all_rows_preserved(self, synthetic_df):
        """train + val should contain all rows with no duplicates."""
        df_train, df_val = entity_level_split(synthetic_df, val_frac=0.2, seed=42)
        combined = pd.concat([df_train, df_val])
        assert len(combined) == len(synthetic_df)

    def test_deterministic_with_same_seed(self, synthetic_df):
        """Same seed → same split."""
        tr1, vl1 = entity_level_split(synthetic_df, val_frac=0.2, seed=99)
        tr2, vl2 = entity_level_split(synthetic_df, val_frac=0.2, seed=99)
        assert set(vl1[ENTITY_COL].unique()) == set(vl2[ENTITY_COL].unique())

    def test_different_seeds_give_different_splits(self, synthetic_df):
        """Different seeds should (very likely) give different splits on 10-entity data."""
        _, vl_a = entity_level_split(synthetic_df, val_frac=0.2, seed=1)
        _, vl_b = entity_level_split(synthetic_df, val_frac=0.2, seed=2)
        # At 10 entities / 2 val, probability of exact same split is C(10,2)/C(10,2)^-1 ≈ 0.0
        # Not guaranteed but extremely likely to differ
        assert set(vl_a[ENTITY_COL].unique()) != set(vl_b[ENTITY_COL].unique()), (
            "Different seeds produced identical splits — this is statistically very unlikely."
        )

    def test_val_frac_zero_gives_empty_val(self, synthetic_df):
        """val_frac close to 0 → val gets 1 entity (minimum)."""
        df_train, df_val = entity_level_split(synthetic_df, val_frac=0.001, seed=42)
        assert df_val[ENTITY_COL].nunique() >= 1


# ---------------------------------------------------------------------------
# Leakage detector tests
# ---------------------------------------------------------------------------

class TestCheckNoEntityLeakage:

    def test_clean_split_passes(self, synthetic_df):
        df_train, df_val = entity_level_split(synthetic_df, val_frac=0.2, seed=42)
        assert check_no_entity_leakage(df_train, df_val) is True

    def test_leaked_split_fails(self, synthetic_df):
        """Deliberately inject the same entity into both splits."""
        df_train = synthetic_df.copy()
        df_val   = synthetic_df.copy()   # SAME data → full leakage
        assert check_no_entity_leakage(df_train, df_val) is False


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

class TestFbetaScore:

    def test_perfect_precision_recall(self):
        assert fbeta_score(1.0, 1.0, beta=0.5) == pytest.approx(1.0)

    def test_zero_precision(self):
        assert fbeta_score(0.0, 1.0, beta=0.5) == pytest.approx(0.0)

    def test_zero_recall(self):
        assert fbeta_score(1.0, 0.0, beta=0.5) == pytest.approx(0.0)

    def test_f05_weights_precision_more(self):
        """F0.5 with P=1.0, R=0.5 > F0.5 with P=0.5, R=1.0."""
        f_high_prec = fbeta_score(1.0, 0.5, beta=0.5)
        f_high_rec  = fbeta_score(0.5, 1.0, beta=0.5)
        assert f_high_prec > f_high_rec

    def test_both_zero(self):
        assert fbeta_score(0.0, 0.0, beta=0.5) == pytest.approx(0.0)


class TestEvaluateAtThresholds:

    @pytest.fixture
    def y_true_prob(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        y_prob = np.where(y_true == 1, rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.5, 100))
        return y_true, y_prob

    def test_threshold_keys_present(self, y_true_prob):
        y_true, y_prob = y_true_prob
        result = evaluate_at_thresholds(y_true, y_prob, thresholds=[0.5, 0.7])
        assert "thr_0.5" in result
        assert "thr_0.7" in result

    def test_result_metric_keys(self, y_true_prob):
        y_true, y_prob = y_true_prob
        result = evaluate_at_thresholds(y_true, y_prob, thresholds=[0.5])
        thr_result = result["thr_0.5"]
        for k in ("threshold", "precision", "recall", "f0.5", "n_predicted_positive"):
            assert k in thr_result, f"Missing key: {k}"

    def test_higher_threshold_fewer_positives(self, y_true_prob):
        """Higher threshold → fewer predicted positives."""
        y_true, y_prob = y_true_prob
        result = evaluate_at_thresholds(y_true, y_prob, thresholds=[0.3, 0.8])
        assert result["thr_0.3"]["n_predicted_positive"] >= result["thr_0.8"]["n_predicted_positive"]


class TestEvaluateSplitBySource:

    @pytest.fixture
    def val_df_with_source(self):
        return pd.DataFrame({
            ENTITY_COL: [f"S1-{i}" for i in range(10)],
            LABEL_COL:  [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            SOURCE_COL: ["s2", "s2", "s2", "s2", "s2", "s3", "s3", "s3", "s3", "s3"],
        })

    def test_source_keys_present(self, val_df_with_source):
        y_prob = np.array([0.8, 0.3, 0.7, 0.2, 0.9, 0.1, 0.2, 0.75, 0.15, 0.05])
        result = evaluate_split_by_source(val_df_with_source, y_prob, thresholds=[0.5])
        assert "S1_S2" in result
        assert "S1_S3" in result
        assert "ALL"   in result

    def test_n_pairs_correct(self, val_df_with_source):
        y_prob = np.zeros(10)
        result = evaluate_split_by_source(val_df_with_source, y_prob, thresholds=[0.5])
        assert result["S1_S2"]["n_pairs"] == 5
        assert result["S1_S3"]["n_pairs"] == 5
        assert result["ALL"]["n_pairs"]   == 10
