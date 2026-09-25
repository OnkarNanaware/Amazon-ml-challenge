"""
tests/test_normalization.py
===========================
Unit tests for src/entity_resolution/normalization.py.

All tests use synthetic DataFrames — no S3 / AWS credentials required.
Run with:
    pytest tests/test_normalization.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.entity_resolution.normalization import (
    normalize_address_series,
    normalize_batch,
    normalize_country_series,
    normalize_name_series,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_df(names: list[str | None], addresses: list[str | None] = None) -> pd.DataFrame:
    """Build a minimal source DataFrame for testing."""
    n = len(names)
    addresses = addresses or ["123 Main St"] * n
    return pd.DataFrame({
        "entity_id":        [f"E{i}" for i in range(n)],
        "business_name":    names,
        "business_address": addresses,
        "country":          ["US"] * n,
    })


# ---------------------------------------------------------------------------
# normalize_name_series
# ---------------------------------------------------------------------------

class TestNormalizeName:

    def test_basic_lowercase(self):
        s = pd.Series(["ACME CORP"])
        norm, _ = normalize_name_series(s)
        assert norm.iloc[0] == "acme"          # "corp" is a legal suffix → stripped

    def test_legal_suffix_stripped_inc(self):
        s = pd.Series(["Widgets Inc", "Widgets Inc.", "Widgets, Inc."])
        norm, _ = normalize_name_series(s)
        assert all(v == "widgets" for v in norm), norm.tolist()

    def test_legal_suffix_stripped_llc(self):
        norm, _ = normalize_name_series(pd.Series(["Delta LLC"]))
        assert norm.iloc[0] == "delta"

    def test_legal_suffix_stripped_ltd(self):
        norm, _ = normalize_name_series(pd.Series(["Gamma Ltd"]))
        assert norm.iloc[0] == "gamma"

    def test_legal_suffix_stripped_incorporated(self):
        norm, _ = normalize_name_series(pd.Series(["Beta Incorporated"]))
        assert norm.iloc[0] == "beta"

    def test_legal_suffix_gmbh(self):
        norm, _ = normalize_name_series(pd.Series(["Schreiber GmbH"]))
        assert norm.iloc[0] == "schreiber"

    def test_legal_suffix_sa(self):
        norm, _ = normalize_name_series(pd.Series(["Société SA"]))
        # NFKC: é stays as é, then normalized; "sa" suffix stripped
        result = norm.iloc[0]
        assert "sa" not in result.split(), f"'sa' still in: {result!r}"

    def test_ampersand_to_and(self):
        norm, _ = normalize_name_series(pd.Series(["Smith & Jones"]))
        assert "and" in norm.iloc[0]
        assert "&" not in norm.iloc[0]

    def test_ampersand_preserves_multiword(self):
        # Both sides of & should survive
        norm, _ = normalize_name_series(pd.Series(["Smith & Jones Services"]))
        assert "smith" in norm.iloc[0]
        assert "jones" in norm.iloc[0]
        assert "and" in norm.iloc[0]

    def test_nfkc_accented_characters(self):
        # "Café" → "cafe" after NFKC + lowercase (é → e via NFKC doesn't decompose,
        # but the string is valid UTF-8 and lowercased correctly)
        norm, _ = normalize_name_series(pd.Series(["Café Boulanger"]))
        result = norm.iloc[0]
        assert "caf" in result       # at minimum the root survives
        assert result == result.lower()

    def test_nfkc_full_width_chars(self):
        # Full-width Latin letters → ASCII via NFKC
        norm, _ = normalize_name_series(pd.Series(["Ａｃｍｅ"]))   # U+FF21 etc.
        assert norm.iloc[0] == "acme"

    def test_societe_accented(self):
        norm, _ = normalize_name_series(pd.Series(["Société Générale"]))
        result = norm.iloc[0]
        assert result == result.lower()
        assert len(result) > 0

    def test_whitespace_collapse(self):
        norm, _ = normalize_name_series(pd.Series(["  Acme   Corp  "]))
        assert not norm.iloc[0].startswith(" ")
        assert "  " not in norm.iloc[0]

    def test_null_input_returns_none(self):
        norm, tsorted = normalize_name_series(pd.Series([None]))
        assert norm.iloc[0] is None
        assert tsorted.iloc[0] is None

    def test_empty_string_returns_none(self):
        norm, _ = normalize_name_series(pd.Series(["   "]))
        assert norm.iloc[0] is None

    def test_token_sorted_single_word(self):
        _, tsorted = normalize_name_series(pd.Series(["Alpha"]))
        # Single word token-sorted == the word itself (after suffix strip if any)
        assert tsorted.iloc[0] is not None

    def test_token_sorted_multiword_is_sorted(self):
        _, tsorted = normalize_name_series(pd.Series(["Zebra Alpha Beta"]))
        tokens = tsorted.iloc[0].split()
        assert tokens == sorted(tokens)

    def test_token_sorted_differs_from_norm_for_reordered(self):
        # "Apple Pie" and "Pie Apple" should produce same token_sorted
        _, t1 = normalize_name_series(pd.Series(["Apple Pie"]))
        _, t2 = normalize_name_series(pd.Series(["Pie Apple"]))
        assert t1.iloc[0] == t2.iloc[0]

    def test_abbreviation_intl(self):
        norm, _ = normalize_name_series(pd.Series(["Acme Intl"]))
        assert "international" in norm.iloc[0]

    def test_abbreviation_tech(self):
        norm, _ = normalize_name_series(pd.Series(["Acme Tech"]))
        assert "technology" in norm.iloc[0]

    def test_punctuation_dots_in_abbreviation(self):
        # "A.B.C. Inc" → "abc" after dot removal and suffix strip
        norm, _ = normalize_name_series(pd.Series(["A.B.C. Inc"]))
        assert "abc" in norm.iloc[0]

    def test_suffix_not_stripped_from_middle(self):
        # "Costco" — "co" should NOT be stripped from the middle
        norm, _ = normalize_name_series(pd.Series(["Costco"]))
        assert "costco" in norm.iloc[0]

    def test_mixed_country_name_france(self):
        # French company name with accents
        norm, _ = normalize_name_series(pd.Series(["BNP Paribas SA"]))
        result = norm.iloc[0]
        assert "bnp" in result
        assert "paribas" in result
        assert "sa" not in result.split()   # stripped as legal suffix


# ---------------------------------------------------------------------------
# normalize_address_series
# ---------------------------------------------------------------------------

class TestNormalizeAddress:

    def test_basic_lowercase(self):
        s = normalize_address_series(pd.Series(["123 MAIN STREET"]))
        assert s.iloc[0] == s.iloc[0].lower()

    def test_street_abbreviation_st(self):
        s = normalize_address_series(pd.Series(["123 Main St"]))
        assert "street" in s.iloc[0]

    def test_avenue_abbreviation(self):
        s = normalize_address_series(pd.Series(["456 Oak Ave"]))
        assert "avenue" in s.iloc[0]

    def test_boulevard_abbreviation(self):
        s = normalize_address_series(pd.Series(["789 Sunset Blvd"]))
        assert "boulevard" in s.iloc[0]

    def test_suite_abbreviation(self):
        s = normalize_address_series(pd.Series(["100 Main St Ste 200"]))
        assert "suite" in s.iloc[0]

    def test_direction_north(self):
        s = normalize_address_series(pd.Series(["100 N Main St"]))
        assert "north" in s.iloc[0]

    def test_nfkc_accented_address(self):
        # French address with accented characters
        s = normalize_address_series(pd.Series(["10 Rue de la Paix"]))
        result = s.iloc[0]
        assert result == result.lower()
        assert len(result) > 0

    def test_commas_removed(self):
        s = normalize_address_series(pd.Series(["123 Main St, Suite 4, New York, NY 10001"]))
        assert "," not in s.iloc[0]

    def test_null_returns_none(self):
        s = normalize_address_series(pd.Series([None]))
        assert s.iloc[0] is None

    def test_empty_string_returns_none(self):
        s = normalize_address_series(pd.Series([""]))
        assert s.iloc[0] is None

    def test_whitespace_collapsed(self):
        s = normalize_address_series(pd.Series(["  123   Main  St  "]))
        assert "  " not in s.iloc[0]
        assert not s.iloc[0].startswith(" ")

    def test_numeric_tokens_preserved(self):
        # Street number and zip code must survive
        s = normalize_address_series(pd.Series(["1234 Oak Ave 90210"]))
        result = s.iloc[0]
        assert "1234" in result
        assert "90210" in result

    def test_india_address_format(self):
        # Indian address — no standard abbreviations expected, but must not crash
        s = normalize_address_series(pd.Series(["42 MG Road, Bangalore 560001"]))
        result = s.iloc[0]
        assert result is not None
        assert "560001" in result


# ---------------------------------------------------------------------------
# normalize_country_series
# ---------------------------------------------------------------------------

class TestNormalizeCountry:

    def test_us_title_case(self):
        s = normalize_country_series(pd.Series(["US"]))
        assert s.iloc[0] == "Us"      # title-case single word

    def test_india_title_case(self):
        s = normalize_country_series(pd.Series(["india"]))
        assert s.iloc[0] == "India"

    def test_france_title_case(self):
        s = normalize_country_series(pd.Series(["FRANCE"]))
        assert s.iloc[0] == "France"

    def test_whitespace_trimmed(self):
        s = normalize_country_series(pd.Series(["  United States  "]))
        assert s.iloc[0] == "United States"

    def test_null_returns_none(self):
        s = normalize_country_series(pd.Series([None]))
        assert s.iloc[0] is None

    def test_no_hardcoded_filtering(self):
        # Arbitrary country values must survive unchanged (except casing)
        countries = ["Germany", "Japan", "Brazil", "South Africa", "Mexico"]
        s = normalize_country_series(pd.Series(countries))
        assert list(s) == [c.title() for c in countries]


# ---------------------------------------------------------------------------
# normalize_batch  (end-to-end integration tests)
# ---------------------------------------------------------------------------

class TestNormalizeBatch:

    def test_output_columns_present(self):
        df = _name_df(["Acme Corp"])
        out = normalize_batch(df)
        for col in [
            "original_name", "original_address", "original_country",
            "normalized_name", "normalized_address", "normalized_country",
            "token_sorted_name", "address_missing",
        ]:
            assert col in out.columns, f"Missing column: {col}"

    def test_original_columns_unchanged(self):
        df = _name_df(["Acme Corp"], ["123 Main St"])
        out = normalize_batch(df)
        assert out["original_name"].iloc[0] == "Acme Corp"
        assert out["original_address"].iloc[0] == "123 Main St"
        # Also verify raw columns still exist
        assert out["business_name"].iloc[0] == "Acme Corp"
        assert out["business_address"].iloc[0] == "123 Main St"

    def test_address_missing_flag_true_when_null(self):
        df = _name_df(["Acme Corp"], [None])
        out = normalize_batch(df)
        assert out["address_missing"].iloc[0] is True or out["address_missing"].iloc[0] == True

    def test_address_missing_flag_false_when_present(self):
        df = _name_df(["Acme Corp"], ["123 Main St"])
        out = normalize_batch(df)
        assert not out["address_missing"].iloc[0]

    def test_normalized_address_none_when_missing(self):
        df = _name_df(["Acme Corp"], [None])
        out = normalize_batch(df)
        assert pd.isna(out["normalized_address"].iloc[0])

    def test_legal_suffix_removed_in_batch(self):
        df = _name_df(["Alpha Inc", "Beta LLC", "Gamma Ltd"])
        out = normalize_batch(df)
        for norm in out["normalized_name"]:
            words = norm.split()
            assert "inc" not in words
            assert "llc" not in words
            assert "ltd" not in words

    def test_ampersand_normalized_in_batch(self):
        df = _name_df(["Smith & Jones"])
        out = normalize_batch(df)
        assert "&" not in out["normalized_name"].iloc[0]
        assert "and" in out["normalized_name"].iloc[0]

    def test_non_ascii_accented_batch(self):
        # Both French and generic accented names must not crash
        df = _name_df(["Café de Paris", "Société Générale SA"])
        out = normalize_batch(df)
        assert out["normalized_name"].notna().all()
        for v in out["normalized_name"]:
            assert v == v.lower()

    def test_mixed_address_null_and_present(self):
        df = _name_df(
            ["Acme Corp", "Beta Inc"],
            [None, "456 Oak Ave Suite 10"],
        )
        out = normalize_batch(df)
        assert out["address_missing"].iloc[0] is True or out["address_missing"].iloc[0] == True
        assert not out["address_missing"].iloc[1]
        assert pd.isna(out["normalized_address"].iloc[0])
        assert "avenue" in out["normalized_address"].iloc[1]

    def test_row_count_preserved(self):
        df = _name_df(["A", "B", "C", "D", "E"])
        out = normalize_batch(df)
        assert len(out) == 5

    def test_entity_id_preserved(self):
        df = _name_df(["Acme"])
        out = normalize_batch(df)
        assert out["entity_id"].iloc[0] == "E0"

    def test_empty_dataframe(self):
        df = pd.DataFrame({
            "entity_id": [], "business_name": [],
            "business_address": [], "country": [],
        })
        out = normalize_batch(df)
        assert len(out) == 0

    def test_india_address_no_crash(self):
        df = _name_df(
            ["Tata Consultancy Services Ltd"],
            ["22 Raheja Tower, MG Road, Bangalore 560001"],
        )
        out = normalize_batch(df)
        assert out["normalized_name"].notna().iloc[0]
        assert out["normalized_address"].notna().iloc[0]
        assert "560001" in out["normalized_address"].iloc[0]

    def test_country_open_set_france(self):
        df = pd.DataFrame({
            "entity_id": ["F1"],
            "business_name": ["BNP Paribas SA"],
            "business_address": ["16 Boulevard des Italiens, Paris"],
            "country": ["France"],
        })
        out = normalize_batch(df)
        assert out["normalized_country"].iloc[0] == "France"

    def test_country_open_set_germany(self):
        df = pd.DataFrame({
            "entity_id": ["G1"],
            "business_name": ["Volkswagen AG"],
            "business_address": ["Berliner Ring 2, 38440 Wolfsburg"],
            "country": ["Germany"],
        })
        out = normalize_batch(df)
        assert out["normalized_country"].iloc[0] == "Germany"
        # AG suffix stripped from name
        assert "ag" not in out["normalized_name"].iloc[0].split()

    # ── new: source column tests ──────────────────────────────────────────

    def test_no_source_arg_no_source_column(self):
        """Calling normalize_batch without source must NOT add a source column."""
        df  = _name_df(["Acme Corp"])
        out = normalize_batch(df)
        assert "source" not in out.columns

    def test_source_arg_adds_column(self):
        """normalize_batch(df, source='s1') must add source column with correct value."""
        df  = _name_df(["Acme Corp", "Beta LLC"])
        out = normalize_batch(df, source="s1")
        assert "source" in out.columns
        assert (out["source"] == "s1").all()

    def test_source_arg_s2(self):
        """source='s2' produces the correct tag for every row."""
        df  = _name_df(["Gamma Ltd"])
        out = normalize_batch(df, source="s2")
        assert out["source"].iloc[0] == "s2"

    def test_source_does_not_overwrite_entity_id(self):
        """Adding source column must not disturb entity_id."""
        df  = _name_df(["Acme Corp"])
        out = normalize_batch(df, source="s3")
        assert out["entity_id"].iloc[0] == "E0"

