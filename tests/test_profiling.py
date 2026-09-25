"""
tests/test_profiling.py
=======================
Unit tests for src/entity_resolution/profiling.py.

All tests use tiny synthetic DataFrames — no S3 / AWS credentials required.
Run with:
    pytest tests/test_profiling.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.entity_resolution.profiling import (
    profile_basic,
    profile_country,
    profile_duplicates,
    profile_entity_id,
    profile_ground_truth,
    profile_text_fields,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture()
def source_df() -> pd.DataFrame:
    """A tiny source DataFrame that mimics the real schema.

    * Rows 1 and 4 (0-indexed) are exact duplicates (E2 / Beta LLC / 2 Oak Ave / US)
      so duplicate_rows == 1 and duplicate (name,address,country) == 1.
    * Row 5 has country=None so that missing-value and __MISSING__ sentinel tests pass.
    """
    return pd.DataFrame(
        {
            "entity_id": ["E1", "E2", "E3", "E4", "E2", "E5"],
            "name": ["Acme Corp", "Beta LLC", "Gamma SA", "Delta GmbH", "Beta LLC", "Epsilon"],
            "address": ["1 Main St", "2 Oak Ave", "3 Rue de Rivoli", "4 Unter Linden", "2 Oak Ave", "5 Unknown Rd"],
            "country": ["US", "US", "FR", "DE", "US", None],  # row 5 has null country
        }
    )


@pytest.fixture()
def gt_df() -> pd.DataFrame:
    """A tiny ground-truth DataFrame."""
    return pd.DataFrame(
        {
            "entity_id_1": ["E1", "E2", "E3"],
            "entity_id_2": ["E4", "E3", "E4"],
            "label": [1, 1, 0],
        }
    )


@pytest.fixture()
def source_dfs(source_df: pd.DataFrame) -> dict:
    """Simulate the three-source dict passed to profile_ground_truth."""
    s1 = source_df[source_df["entity_id"].isin(["E1", "E2"])].copy()
    s2 = source_df[source_df["entity_id"].isin(["E3", "E4"])].copy()
    return {"train_source1": s1, "train_source2": s2}


# ---------------------------------------------------------------------------
# profile_basic
# ---------------------------------------------------------------------------

class TestProfileBasic:
    def test_row_count(self, source_df):
        result = profile_basic(source_df, "test_file")
        assert result["row_count"] == len(source_df)

    def test_columns_present(self, source_df):
        result = profile_basic(source_df, "test_file")
        assert set(result["columns"].keys()) == set(source_df.columns)

    def test_missing_value_counted(self, source_df):
        # 'country' has one None (row 5)
        result = profile_basic(source_df, "test_file")
        assert result["missing_values"]["country"]["null_count"] == 1
        assert result["missing_values"]["country"]["null_pct"] == pytest.approx(100 / 6, abs=0.01)

    def test_no_missing_for_complete_column(self, source_df):
        result = profile_basic(source_df, "test_file")
        assert result["missing_values"]["entity_id"]["null_count"] == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame({"entity_id": [], "name": []})
        result = profile_basic(df, "empty")
        assert result["row_count"] == 0

    def test_file_label_propagated(self, source_df):
        result = profile_basic(source_df, "my_label")
        assert result["file"] == "my_label"


# ---------------------------------------------------------------------------
# profile_entity_id
# ---------------------------------------------------------------------------

class TestProfileEntityId:
    def test_unique_count(self, source_df):
        result = profile_entity_id(source_df, "test")
        # E1,E2,E3,E4,E2,E5 -> 5 unique
        assert result["unique_entity_ids"] == 5

    def test_duplicate_count(self, source_df):
        result = profile_entity_id(source_df, "test")
        # 5 total rows, 4 unique -> 1 duplicate
        assert result["duplicate_entity_ids"] == 1

    def test_no_entity_id_column(self):
        df = pd.DataFrame({"name": ["A", "B"]})
        result = profile_entity_id(df, "test")
        assert result["entity_id_col_present"] is False

    def test_all_unique(self):
        df = pd.DataFrame({"entity_id": ["X", "Y", "Z"]})
        result = profile_entity_id(df, "test")
        assert result["duplicate_entity_ids"] == 0


# ---------------------------------------------------------------------------
# profile_duplicates
# ---------------------------------------------------------------------------

class TestProfileDuplicates:
    def test_duplicate_rows(self, source_df):
        result = profile_duplicates(source_df, "test")
        # Row 4 (E2/Beta LLC/2 Oak Ave/US) is an exact copy of row 1
        assert result["duplicate_rows"] == 1

    def test_duplicate_name_address_country(self, source_df):
        result = profile_duplicates(source_df, "test")
        # (Beta LLC, 2 Oak Ave, US) appears in rows 1 and 4
        assert result["duplicate_name_address_country"] == 1

    def test_combo_cols_subset_when_missing(self):
        df = pd.DataFrame({"entity_id": ["E1", "E2"], "name": ["A", "B"]})
        result = profile_duplicates(df, "test")
        # Only 'name' available in combo cols
        assert "name" in result["combo_key_columns_used"]
        assert "address" not in result["combo_key_columns_used"]

    def test_no_duplicates(self):
        df = pd.DataFrame(
            {
                "entity_id": ["E1", "E2"],
                "name": ["A", "B"],
                "address": ["Addr1", "Addr2"],
                "country": ["US", "FR"],
            }
        )
        result = profile_duplicates(df, "test")
        assert result["duplicate_rows"] == 0
        assert result["duplicate_name_address_country"] == 0


# ---------------------------------------------------------------------------
# profile_text_fields
# ---------------------------------------------------------------------------

class TestProfileTextFields:
    def test_unique_name_count(self, source_df):
        result = profile_text_fields(source_df, "test")
        # Acme Corp, Beta LLC (x2), Gamma SA, Delta GmbH, Epsilon -> 5 unique
        assert result["name"]["unique_count"] == 5

    def test_length_distribution_keys(self, source_df):
        result = profile_text_fields(source_df, "test")
        expected_keys = {"min", "max", "mean", "median", "p25", "p75", "p95"}
        assert expected_keys == set(result["name"]["length_distribution"].keys())

    def test_absent_column_flagged(self):
        df = pd.DataFrame({"entity_id": ["E1"]})
        result = profile_text_fields(df, "test")
        assert result["name"]["present"] is False
        assert result["address"]["present"] is False

    def test_min_le_max(self, source_df):
        result = profile_text_fields(source_df, "test")
        ld = result["name"]["length_distribution"]
        assert ld["min"] <= ld["max"]


# ---------------------------------------------------------------------------
# profile_country
# ---------------------------------------------------------------------------

class TestProfileCountry:
    def test_total_unique_countries(self, source_df):
        result = profile_country(source_df, "test")
        # US, FR, DE, __MISSING__ = 4 unique values
        assert result["total_unique_countries"] == 4

    def test_missing_counted_as_sentinel(self, source_df):
        result = profile_country(source_df, "test")
        assert "__MISSING__" in result["country_distribution"]

    def test_absent_column_flagged(self):
        df = pd.DataFrame({"name": ["A"]})
        result = profile_country(df, "test")
        assert result["country_col_present"] is False

    def test_open_set_all_countries_counted(self):
        """Country distribution must NOT filter to a known list."""
        df = pd.DataFrame(
            {"country": ["US", "IN", "FR", "DE", "JP", "BR", "AU", "ZA", "MX", "KR"]}
        )
        result = profile_country(df, "test")
        assert result["total_unique_countries"] == 10
        for c in ["US", "IN", "FR", "DE", "JP", "BR", "AU", "ZA", "MX", "KR"]:
            assert c in result["country_distribution"]


# ---------------------------------------------------------------------------
# profile_ground_truth
# ---------------------------------------------------------------------------

class TestProfileGroundTruth:
    def test_total_pairs(self, gt_df, source_dfs):
        result = profile_ground_truth(gt_df, source_dfs)
        assert result["total_pairs"] == 3

    def test_matched_pairs(self, gt_df, source_dfs):
        result = profile_ground_truth(gt_df, source_dfs)
        assert result["matched_pairs"] == 2

    def test_non_matched_pairs(self, gt_df, source_dfs):
        result = profile_ground_truth(gt_df, source_dfs)
        assert result["non_matched_pairs"] == 1

    def test_label_column_detected(self, gt_df, source_dfs):
        result = profile_ground_truth(gt_df, source_dfs)
        assert result["label_column_present"] is True

    def test_cross_source_distribution_keys(self, gt_df, source_dfs):
        result = profile_ground_truth(gt_df, source_dfs)
        dist = result["cross_source_pair_distribution"]
        # All values should be ints >= 0
        for v in dist.values():
            assert isinstance(v, int)
            assert v >= 0

    def test_no_label_column(self, gt_df, source_dfs):
        gt_no_label = gt_df.drop(columns=["label"])
        result = profile_ground_truth(gt_no_label, source_dfs)
        assert result["label_column_present"] is False
        assert result["matched_pairs"] is None
        assert result["non_matched_pairs"] is None

    def test_empty_ground_truth(self, source_dfs):
        gt_empty = pd.DataFrame({"entity_id_1": [], "entity_id_2": [], "label": []})
        result = profile_ground_truth(gt_empty, source_dfs)
        assert result["total_pairs"] == 0
        assert result["matched_pairs"] == 0
