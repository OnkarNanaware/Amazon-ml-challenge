"""
tests/test_features.py
======================
Unit tests for src/entity_resolution/features.py

Test coverage
-------------
NAME features:
  - Identical names → name_exact=1.0, high similarity scores
  - Completely different names → name_exact=0.0, low similarity
  - One empty name → all name features = 0.0

ADDRESS features:
  - address_missing on either side → all address features = -1.0 sentinel
  - Identical addresses → address_exact=1.0
  - Different addresses → address_exact=0.0
  - Numeric token overlap tested independently

COUNTRY features:
  - Both match → country_exact=1.0, country_normalized=1.0
  - Both missing → country_exact=1.0, country_normalized=0.0, country_unseen=1.0
  - One missing → country_normalized=-1.0, country_unseen=1.0

CANDIDATE METADATA features:
  - blocking_method strings correctly bitmask into method indicator features
  - candidate_rank and candidate_margin passed through correctly

BATCH computation:
  - compute_features_batch produces no NaN values
  - Output shape matches input length
  - candidate_rank is correctly ordered per entity
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.entity_resolution.features import (
    _FEATURE_NAMES,
    _jaccard,
    _ngram_set,
    _token_set,
    compute_features,
    compute_features_batch,
    compute_feature_stats_report,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _left(
    name: str = "acme corporation",
    address: str = "123 main street new york",
    country: str = "US",
    address_missing: bool = False,
) -> Dict[str, Any]:
    return {
        "normalized_name":    name,
        "normalized_address": address,
        "normalized_country": country,
        "address_missing":    address_missing,
    }


def _right(**kwargs) -> Dict[str, Any]:
    return _left(**kwargs)


# ---------------------------------------------------------------------------
# NAME feature tests
# ---------------------------------------------------------------------------

class TestNameFeatures:

    def test_identical_names(self):
        f = compute_features(_left(name="acme corp"), _right(name="acme corp"))
        assert f["name_exact"] == 1.0
        assert f["name_levenshtein"] == pytest.approx(1.0)
        assert f["name_token_jaccard"] == pytest.approx(1.0)
        assert f["name_token_sort_ratio"] == pytest.approx(1.0)
        assert f["name_length_difference"] == pytest.approx(0.0)

    def test_completely_different_names(self):
        f = compute_features(_left(name="acme corp"), _right(name="zzzzz qqqq"))
        assert f["name_exact"] == 0.0
        assert f["name_levenshtein"] < 0.5
        assert f["name_token_jaccard"] == pytest.approx(0.0)

    def test_empty_left_name(self):
        f = compute_features(_left(name=""), _right(name="acme corp"))
        for feat in ("name_exact", "name_levenshtein", "name_jaro_winkler",
                     "name_token_jaccard", "name_token_sort_ratio",
                     "name_ngram_cosine", "name_length_difference"):
            assert f[feat] == 0.0, f"Expected 0.0 for {feat} when left name empty"

    def test_empty_right_name(self):
        f = compute_features(_left("acme corp"), _right(name=""))
        for feat in ("name_exact", "name_levenshtein"):
            assert f[feat] == 0.0

    def test_token_sort_handles_reordering(self):
        """Token sort ratio should be high for reordered tokens."""
        f = compute_features(_left(name="new york acme"), _right(name="acme new york"))
        assert f["name_token_sort_ratio"] == pytest.approx(1.0)

    def test_length_difference_scaled(self):
        """Length difference should be normalised 0-1."""
        f = compute_features(_left(name="ab"), _right(name="abcdefghij"))
        assert 0.0 <= f["name_length_difference"] <= 1.0

    def test_name_ngram_cosine_range(self):
        f = compute_features(_left(name="hello world"), _right(name="hello earth"))
        assert 0.0 <= f["name_ngram_cosine"] <= 1.0


# ---------------------------------------------------------------------------
# ADDRESS feature tests
# ---------------------------------------------------------------------------

class TestAddressFeatures:

    def test_address_missing_left_sentinel(self):
        """address_missing=True on LEFT → all address features = -1.0."""
        f = compute_features(_left(address_missing=True), _right())
        for feat in ("address_exact", "address_jaccard", "address_ngram_similarity",
                     "address_edit_distance", "numeric_token_similarity",
                     "postal_similarity", "address_length_difference"):
            assert f[feat] == pytest.approx(-1.0), f"Expected -1.0 for {feat} (left missing)"

    def test_address_missing_right_sentinel(self):
        """address_missing=True on RIGHT → all address features = -1.0."""
        f = compute_features(_left(), _right(address_missing=True))
        for feat in ("address_exact", "address_jaccard"):
            assert f[feat] == pytest.approx(-1.0), f"Expected -1.0 for {feat} (right missing)"

    def test_address_missing_either_flag(self):
        f_left_miss  = compute_features(_left(address_missing=True), _right())
        f_right_miss = compute_features(_left(), _right(address_missing=True))
        f_both       = compute_features(_left(address_missing=True), _right(address_missing=True))
        assert f_left_miss["address_missing_either"]  == 1.0
        assert f_right_miss["address_missing_either"] == 1.0
        assert f_both["address_missing_either"]       == 1.0

    def test_address_missing_left_right_flags(self):
        f = compute_features(_left(address_missing=True), _right())
        assert f["address_missing_left"]  == 1.0
        assert f["address_missing_right"] == 0.0

    def test_identical_addresses(self):
        f = compute_features(_left(address="123 main street"), _right(address="123 main street"))
        assert f["address_exact"] == 1.0
        assert f["address_jaccard"] == pytest.approx(1.0)
        assert f["address_length_difference"] == pytest.approx(0.0)

    def test_different_addresses(self):
        f = compute_features(
            _left(address="123 main street new york"),
            _right(address="456 oak avenue chicago"),
        )
        assert f["address_exact"] == 0.0
        assert f["address_jaccard"] < 1.0

    def test_numeric_token_similarity(self):
        """Shared house number → numeric_token_similarity > 0."""
        f = compute_features(
            _left(address="123 main street"),
            _right(address="123 oak avenue"),
        )
        assert f["numeric_token_similarity"] > 0.0

    def test_numeric_token_no_overlap(self):
        f = compute_features(
            _left(address="100 main street"),
            _right(address="999 oak avenue"),
        )
        assert f["numeric_token_similarity"] == pytest.approx(0.0)

    def test_postal_similarity_same_last_token(self):
        """Same last token → postal_similarity=1."""
        f = compute_features(
            _left(address="123 main street 10001"),
            _right(address="456 oak avenue 10001"),
        )
        assert f["postal_similarity"] == 1.0

    def test_postal_similarity_different(self):
        f = compute_features(
            _left(address="123 main street 10001"),
            _right(address="123 main street 90210"),
        )
        assert f["postal_similarity"] == 0.0


# ---------------------------------------------------------------------------
# COUNTRY feature tests
# ---------------------------------------------------------------------------

class TestCountryFeatures:

    def test_both_countries_match(self):
        f = compute_features(_left(country="US"), _right(country="US"))
        assert f["country_exact"]      == 1.0
        assert f["country_normalized"] == 1.0
        assert f["country_unseen"]     == 0.0

    def test_countries_differ(self):
        f = compute_features(_left(country="US"), _right(country="India"))
        assert f["country_exact"]  == 0.0
        assert f["country_unseen"] == 0.0

    def test_both_countries_missing(self):
        f = compute_features(_left(country=""), _right(country=""))
        assert f["country_exact"]      == 1.0  # both missing → treat as same
        assert f["country_normalized"] == 0.0  # but can't confirm match
        assert f["country_unseen"]     == 1.0

    def test_one_country_missing(self):
        f = compute_features(_left(country="US"), _right(country=""))
        assert f["country_normalized"] == pytest.approx(-1.0)  # partial-missing sentinel
        assert f["country_unseen"]     == 1.0

    def test_country_case_insensitive(self):
        """country features use lower-cased comparison."""
        f = compute_features(_left(country="us"), _right(country="US"))
        assert f["country_exact"] == 1.0


# ---------------------------------------------------------------------------
# CANDIDATE METADATA feature tests
# ---------------------------------------------------------------------------

class TestMetadataFeatures:

    def test_blocking_score_passthrough(self):
        f = compute_features(_left(), _right(), blocking_score=0.87)
        assert f["blocking_score"] == pytest.approx(0.87)

    def test_blocking_method_tfidf(self):
        f = compute_features(_left(), _right(), blocking_method="tfidf+token_sort")
        assert f["blocking_method_tfidf"]       == 1.0
        assert f["blocking_method_token_sort"]  == 1.0
        assert f["blocking_method_addr_prefix"] == 0.0

    def test_blocking_method_address_prefix(self):
        f = compute_features(_left(), _right(), blocking_method="address_prefix")
        assert f["blocking_method_tfidf"]       == 0.0
        assert f["blocking_method_addr_prefix"] == 1.0

    def test_candidate_rank_and_margin(self):
        f = compute_features(_left(), _right(), candidate_rank=3, candidate_margin=0.15)
        assert f["candidate_rank"]   == pytest.approx(3.0)
        assert f["candidate_margin"] == pytest.approx(0.15)

    def test_candidate_source_s2(self):
        f = compute_features(_left(), _right(), candidate_source="s2")
        assert f["candidate_source_s2"] == 1.0
        assert f["candidate_source_s3"] == 0.0

    def test_candidate_source_s3(self):
        f = compute_features(_left(), _right(), candidate_source="s3")
        assert f["candidate_source_s2"] == 0.0
        assert f["candidate_source_s3"] == 1.0


# ---------------------------------------------------------------------------
# All features present and in expected range
# ---------------------------------------------------------------------------

class TestFeatureCompleteness:

    def test_all_features_present(self):
        """compute_features returns all expected feature keys."""
        f = compute_features(_left(), _right())
        for name in _FEATURE_NAMES:
            assert name in f, f"Missing feature: {name}"

    def test_no_nan_features(self):
        """No NaN values in any feature."""
        f = compute_features(_left(), _right())
        for k, v in f.items():
            assert v == v, f"NaN detected in feature {k}"  # NaN != NaN

    def test_features_in_range(self):
        """Most features should be in [-1.0, 1.0] (sentinel = -1.0 is ok)."""
        f = compute_features(_left(), _right())
        for k, v in f.items():
            if k == "candidate_rank":
                continue  # can be any non-negative integer
            assert -1.001 <= v <= 1.001, f"Feature {k}={v} out of range [-1, 1]"


# ---------------------------------------------------------------------------
# Batch computation tests
# ---------------------------------------------------------------------------

class TestComputeFeaturesBatch:

    @pytest.fixture
    def norm_s1(self):
        return pd.DataFrame({
            "entity_id":          ["S1-100", "S1-101"],
            "normalized_name":    ["acme corp", "global tech"],
            "normalized_address": ["123 main street", "456 oak avenue"],
            "normalized_country": ["us", "india"],
            "address_missing":    [False, False],
            "source":             ["s1", "s1"],
        })

    @pytest.fixture
    def norm_s2(self):
        return pd.DataFrame({
            "entity_id":          ["S2-200", "S2-201", "S2-999"],
            "normalized_name":    ["acme corp", "global technologies", "zzz bbb"],
            "normalized_address": ["123 main street", None, "999 elm road"],
            "normalized_country": ["us", "india", "uk"],
            "address_missing":    [False, True, False],
            "source":             ["s2", "s2", "s2"],
        })

    @pytest.fixture
    def labeled_pairs(self):
        return pd.DataFrame({
            "source1_entity_id":   ["S1-100", "S1-100", "S1-101"],
            "candidate_entity_id": ["S2-200", "S2-999", "S2-201"],
            "candidate_source":    ["s2",     "s2",     "s2"],
            "blocking_score":      [0.95,     0.10,     0.88],
            "blocking_method":     ["tfidf",  "token_sort", "tfidf"],
            "label":               [1,        0,          0],
            "negative_type":       ["positive", "easy_negative", "hard_negative"],
        })

    def test_output_length_matches_input(self, labeled_pairs, norm_s1, norm_s2):
        result = compute_features_batch(labeled_pairs, norm_s1, norm_s2)
        assert len(result) == len(labeled_pairs)

    def test_no_nan_features_in_batch(self, labeled_pairs, norm_s1, norm_s2):
        result = compute_features_batch(labeled_pairs, norm_s1, norm_s2)
        for feat in _FEATURE_NAMES:
            if feat in result.columns:
                n_null = result[feat].isnull().sum()
                assert n_null == 0, f"NaN in feature {feat}"

    def test_candidate_rank_ordered_per_entity(self, labeled_pairs, norm_s1, norm_s2):
        """candidate_rank should be 0 for the highest-scored candidate per entity."""
        result = compute_features_batch(labeled_pairs, norm_s1, norm_s2)
        for entity, group in result.groupby("source1_entity_id"):
            # sort by the original blocking_score from the labeled_pairs side
            sorted_group = group.sort_values("blocking_score", ascending=False)
            ranks = sorted_group["candidate_rank"].values
            assert ranks[0] == 0, f"Entity {entity}: rank[0]={ranks[0]}, expected 0"

    def test_address_missing_sentinel_propagated(self, labeled_pairs, norm_s1, norm_s2):
        """S2-201 has address_missing=True → address features should be -1.0 for that pair."""
        result = compute_features_batch(labeled_pairs, norm_s1, norm_s2)
        row = result[result["candidate_entity_id"] == "S2-201"]
        assert len(row) == 1
        assert float(row["address_exact"].iloc[0]) == pytest.approx(-1.0)
        assert float(row["address_jaccard"].iloc[0]) == pytest.approx(-1.0)

    def test_label_column_preserved(self, labeled_pairs, norm_s1, norm_s2):
        result = compute_features_batch(labeled_pairs, norm_s1, norm_s2)
        assert "label" in result.columns
        assert list(result["label"]) == list(labeled_pairs["label"])


# ---------------------------------------------------------------------------
# Stats report tests
# ---------------------------------------------------------------------------

class TestFeatureStatsReport:

    @pytest.fixture
    def minimal_feat_df(self):
        """A minimal feature DataFrame with all feature columns."""
        n = 20
        data = {}
        for feat in _FEATURE_NAMES:
            if feat == "candidate_rank":
                data[feat] = list(range(n))
            else:
                data[feat] = [float(i % 2) for i in range(n)]
        return pd.DataFrame(data)

    def test_report_has_all_features(self, minimal_feat_df):
        report = compute_feature_stats_report(minimal_feat_df, minimal_feat_df)
        for feat in _FEATURE_NAMES:
            assert feat in report["feature_stats"], f"Missing feature stats: {feat}"

    def test_flags_keys_present(self, minimal_feat_df):
        report = compute_feature_stats_report(minimal_feat_df, minimal_feat_df)
        assert "zero_variance" in report["flags"]
        assert "always_null"   in report["flags"]
        assert "out_of_range"  in report["flags"]

    def test_zero_variance_detected(self, minimal_feat_df):
        """A column with all-same values should be flagged as zero-variance."""
        df = minimal_feat_df.copy()
        df["name_exact"] = 0.0   # all zeros → zero variance
        report = compute_feature_stats_report(df, df)
        assert "name_exact" in report["flags"]["zero_variance"]


# ---------------------------------------------------------------------------
# Low-level helper tests
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_token_set(self):
        assert _token_set("hello world") == {"hello", "world"}
        assert _token_set("") == set()

    def test_ngram_set(self):
        assert _ngram_set("ab", 2) == {"ab"}
        assert _ngram_set("abc", 2) == {"ab", "bc"}
        assert _ngram_set("a", 2) == set()  # too short

    def test_jaccard_identical(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == pytest.approx(0.0)

    def test_jaccard_empty_sets(self):
        assert _jaccard(set(), set()) == pytest.approx(1.0)

    def test_jaccard_partial(self):
        result = _jaccard({"a", "b"}, {"b", "c"})
        assert result == pytest.approx(1/3)
