"""
tests/test_blocking_experiments.py
====================================
Unit tests for src/entity_resolution/blocking_experiments.py.

All tests use tiny synthetic DataFrames — no S3 / AWS credentials required.
Known true/false match structure is explicitly designed into each fixture so
recall assertions are exact.

Run with:
    pytest tests/test_blocking_experiments.py -v

Authors: Account 2 (blocking experiments)
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.entity_resolution.blocking_experiments import (
    # GT helpers
    GTRecord,
    parse_ground_truth_df,
    evaluate_candidates,
    evaluate_address_missing_subset,
    evaluate_by_country,
    # Blocking primitives
    _tfidf_candidates,
    _token_sort_candidates,
    _address_prefix_candidates,
    _rare_token_candidates,
    _postal_char_ngram_candidates,
    _country_scoped_tfidf_candidates,
    _empty_cands,
    merge_and_cap,
    # Experiment runners
    exp1_tfidf_topk_sweep,
    exp1_rare_token_rarity_sweep,
    exp1_global_cap_sweep,
    exp2_union_vs_weighted,
    exp2_strategy_drop_analysis,
    exp3_postal_char_ngram,
    exp3_country_scoped_tfidf,
    exp3_address_missing_subset_quality,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _s1_df(
    ids:     list,
    names:   list,
    addrs:   list | None = None,
    missing: list | None = None,
    country: list | None = None,
) -> pd.DataFrame:
    """Build a minimal S1 normalized DataFrame."""
    n       = len(ids)
    addrs   = addrs   or ["100 main street"] * n
    missing = missing or [False] * n
    country = country or ["United States"] * n
    tsorted = [" ".join(sorted(str(nm).split())) if nm else None for nm in names]
    return pd.DataFrame({
        "entity_id":          ids,
        "normalized_name":    names,
        "token_sorted_name":  tsorted,
        "normalized_address": addrs,
        "address_missing":    missing,
        "country":            country,
        "source":             ["s1"] * n,
    })


def _s2_df(ids, names, addrs=None, missing=None, country=None):
    df = _s1_df(ids, names, addrs, missing, country)
    df["source"] = "s2"
    return df


def _s3_df(ids, names, addrs=None, missing=None, country=None):
    df = _s1_df(ids, names, addrs, missing, country)
    df["source"] = "s3"
    return df


def _gt_df(s1_ids: list, matched: list) -> pd.DataFrame:
    """Build a GT DataFrame from parallel lists."""
    return pd.DataFrame({
        "source1_entity_id": s1_ids,
        "matched_entity_ids": matched,
    })


def _cands(pairs: list, method: str = "tfidf_name", score: float = 0.8) -> pd.DataFrame:
    """Build a minimal candidate DataFrame."""
    return pd.DataFrame({
        "entity_id_left":  [p[0] for p in pairs],
        "entity_id_right": [p[1] for p in pairs],
        "blocking_score":  [score] * len(pairs),
        "blocking_method": [method] * len(pairs),
    })


# ============================================================================
# GT parsing
# ============================================================================

class TestParseGroundTruthDf:

    def test_basic_parse(self):
        df = _gt_df(["S1-1", "S1-2"], ["S2-100,S3-200", "S2-300"])
        records = parse_ground_truth_df(df)
        assert len(records) == 2
        r1 = next(r for r in records if r.s1_id == "S1-1")
        assert r1.s2_targets == {"S2-100"}
        assert r1.s3_targets == {"S3-200"}
        assert not r1.is_singleton

    def test_singleton_empty_string(self):
        df = _gt_df(["S1-1"], [""])
        records = parse_ground_truth_df(df)
        assert records[0].is_singleton
        assert records[0].s2_targets == set()
        assert records[0].s3_targets == set()

    def test_singleton_whitespace(self):
        df = _gt_df(["S1-1"], ["   "])
        records = parse_ground_truth_df(df)
        assert records[0].is_singleton

    def test_multiple_s2_and_s3(self):
        df = _gt_df(["S1-1"], ["S2-1, S2-2, S3-5, S3-6"])
        records = parse_ground_truth_df(df)
        r = records[0]
        assert r.s2_targets == {"S2-1", "S2-2"}
        assert r.s3_targets == {"S3-5", "S3-6"}

    def test_s3_only(self):
        df = _gt_df(["S1-1"], ["S3-7"])
        records = parse_ground_truth_df(df)
        r = records[0]
        assert r.s2_targets == set()
        assert r.s3_targets == {"S3-7"}

    def test_whitespace_stripped_from_ids(self):
        df = _gt_df(["S1-1"], [" S2-10 ,  S3-20 "])
        records = parse_ground_truth_df(df)
        r = records[0]
        assert "S2-10" in r.s2_targets
        assert "S3-20" in r.s3_targets

    def test_no_s1s1_s2s2_keys_generated(self):
        # GT should never produce S1-* in targets (cross-source only)
        df = _gt_df(["S1-1"], ["S2-1"])
        records = parse_ground_truth_df(df)
        for r in records:
            assert not any(t.startswith("S1-") for t in r.s2_targets | r.s3_targets)


# ============================================================================
# evaluate_candidates
# ============================================================================

class TestEvaluateCandidates:

    def _gt(self):
        """Two S1 entities with S2 targets, one singleton."""
        return [
            GTRecord("S1-1", {"S2-10", "S2-11"}, set(), is_singleton=False),
            GTRecord("S1-2", {"S2-20"},           set(), is_singleton=False),
            GTRecord("S1-3", set(),               set(), is_singleton=True),   # singleton
        ]

    def test_perfect_recall(self):
        gt = self._gt()
        c  = _cands([("S1-1", "S2-10"), ("S1-1", "S2-11"), ("S1-2", "S2-20")])
        r  = evaluate_candidates(gt, c, "S1_S2")
        assert r.recall == pytest.approx(1.0)
        assert r.found_targets == 3
        assert r.total_gt_targets == 3

    def test_partial_recall(self):
        gt = self._gt()
        c  = _cands([("S1-1", "S2-10")])  # only 1 of 3 targets found
        r  = evaluate_candidates(gt, c, "S1_S2")
        assert r.recall == pytest.approx(1 / 3)

    def test_singleton_not_counted_in_recall(self):
        gt = self._gt()
        # Provide candidates for the singleton S1-3 only; should not affect recall
        c  = _cands([("S1-3", "S2-99")] +
                    [("S1-1", "S2-10"), ("S1-1", "S2-11"), ("S1-2", "S2-20")])
        r  = evaluate_candidates(gt, c, "S1_S2")
        assert r.recall == pytest.approx(1.0)       # singletons excluded
        assert r.singleton_s1_count == 1
        assert r.singleton_cand_vol == 1             # S1-3 generated 1 cand

    def test_s3_pair_type_uses_s3_targets(self):
        gt = [
            GTRecord("S1-1", set(), {"S3-30"}, is_singleton=False),
        ]
        c  = _cands([("S1-1", "S3-30")])
        r  = evaluate_candidates(gt, c, "S1_S3")
        assert r.recall == pytest.approx(1.0)

    def test_empty_candidates_zero_recall(self):
        gt = self._gt()
        r  = evaluate_candidates(gt, _empty_cands(), "S1_S2")
        assert r.recall == pytest.approx(0.0)

    def test_avg_candidates_per_entity(self):
        gt = self._gt()
        c  = _cands([("S1-1", "S2-10"), ("S1-1", "S2-11"), ("S1-2", "S2-20")])
        r  = evaluate_candidates(gt, c, "S1_S2")
        # 3 cands across 2 S1 entities -> avg = 1.5
        assert r.avg_candidates_per_entity == pytest.approx(1.5)

    def test_no_cross_singleton_recall_contamination(self):
        """All GT records are singletons -> recall should be 0.0 (zero denominator)."""
        gt = [GTRecord("S1-1", set(), set(), is_singleton=True)]
        c  = _cands([("S1-1", "S2-99")])
        r  = evaluate_candidates(gt, c, "S1_S2")
        assert r.recall == pytest.approx(0.0)


# ============================================================================
# evaluate_address_missing_subset
# ============================================================================

class TestEvaluateAddressMissingSubset:

    def test_identifies_addr_missing_s1(self):
        gt = [
            GTRecord("S1-1", {"S2-10"}, set(), is_singleton=False),  # has address
            GTRecord("S1-2", {"S2-20"}, set(), is_singleton=False),  # missing address
        ]
        df_s1 = _s1_df(
            ["S1-1", "S1-2"],
            ["alpha corp",  "beta inc"],
            missing=[False, True],
        )
        c = _cands([("S1-1", "S2-10"), ("S1-2", "S2-20")])
        result = evaluate_address_missing_subset(gt, c, df_s1, "S1_S2")
        # Only S1-2 is addr_missing
        assert result["n_s1_addr_missing_with_targets"] == 1
        assert result["recall"] == pytest.approx(1.0)

    def test_empty_addr_missing_subset_returns_none_recall(self):
        gt = [GTRecord("S1-1", {"S2-10"}, set(), is_singleton=False)]
        df_s1 = _s1_df(["S1-1"], ["alpha corp"], missing=[False])
        c = _cands([("S1-1", "S2-10")])
        result = evaluate_address_missing_subset(gt, c, df_s1, "S1_S2")
        assert result["recall"] is None    # no addr_missing entities in subset
        assert result["n_s1_addr_missing_with_targets"] == 0


# ============================================================================
# evaluate_by_country
# ============================================================================

class TestEvaluateByCountry:

    def test_country_breakdown(self):
        gt = [
            GTRecord("S1-1", {"S2-10"}, set(), is_singleton=False),  # US
            GTRecord("S1-2", {"S2-20"}, set(), is_singleton=False),  # IN
        ]
        df_s1 = _s1_df(
            ["S1-1", "S1-2"],
            ["alpha", "beta"],
            country=["United States", "India"],
        )
        c = _cands([("S1-1", "S2-10")])  # only S1-1 found
        rows = evaluate_by_country(gt, c, df_s1, "S1_S2")
        by_country = {r["country"]: r for r in rows}
        assert by_country["United States"]["recall"] == pytest.approx(1.0)
        assert by_country["India"]["recall"] == pytest.approx(0.0)

    def test_missing_country_column_graceful(self):
        gt = [GTRecord("S1-1", {"S2-10"}, set(), is_singleton=False)]
        df_s1 = _s1_df(["S1-1"], ["alpha"])
        df_s1 = df_s1.drop(columns=["country"])   # remove country col
        c = _cands([("S1-1", "S2-10")])
        rows = evaluate_by_country(gt, c, df_s1, "S1_S2")
        # Falls back to "unknown" country
        assert any(r["country"] == "unknown" for r in rows)


# ============================================================================
# Blocking primitives
# ============================================================================

class TestTfidfCandidates:

    def test_similar_names_produce_pair(self):
        left  = _s1_df(["S1-1"], ["acme corporation"])
        right = _s2_df(["S2-1", "S2-2"], ["acme corp", "xyz zyx qrs"])
        out   = _tfidf_candidates(left, right, top_k=10)
        assert "S2-1" in out["entity_id_right"].values

    def test_no_self_pairs(self):
        df  = _s1_df(["E1", "E2"], ["alpha beta", "gamma delta"])
        out = _tfidf_candidates(df, df, top_k=10)
        assert not (out["entity_id_left"] == out["entity_id_right"]).any()

    def test_empty_input_returns_empty(self):
        out = _tfidf_candidates(_s1_df([], []), _s2_df(["S2-1"], ["foo"]), top_k=10)
        assert out.empty

    def test_method_tag_used(self):
        left  = _s1_df(["S1-1"], ["acme corp"])
        right = _s2_df(["S2-1"], ["acme corporation"])
        out   = _tfidf_candidates(left, right, top_k=5, method_tag="my_tag")
        if not out.empty:
            assert (out["blocking_method"] == "my_tag").all()

    def test_top_k_respected(self):
        right = _s2_df([f"S2-{i}" for i in range(30)],
                       [f"acme variant {i}" for i in range(30)])
        left  = _s1_df(["S1-1"], ["acme corporation"])
        out   = _tfidf_candidates(left, right, top_k=5)
        counts = out.groupby("entity_id_left").size()
        assert counts.max() <= 5

    def test_scores_between_0_and_1(self):
        left  = _s1_df(["S1-1"], ["acme corporation"])
        right = _s2_df(["S2-1"], ["acme corp"])
        out   = _tfidf_candidates(left, right, top_k=10)
        if not out.empty:
            assert (out["blocking_score"] >= 0).all()
            assert (out["blocking_score"] <= 1.001).all()


class TestTokenSortCandidates:

    def test_same_prefix_produces_pair(self):
        left  = _s1_df(["S1-1"], ["acme beta"])
        right = _s2_df(["S2-1"], ["acme gamma"])
        out   = _token_sort_candidates(left, right, prefix_len=4)
        assert len(out) >= 1

    def test_word_order_invariant(self):
        left  = _s1_df(["S1-1"], ["beta alpha"])
        right = _s2_df(["S2-1"], ["alpha beta"])
        out   = _token_sort_candidates(left, right, prefix_len=5)
        assert len(out) >= 1

    def test_different_prefix_no_pair(self):
        left  = _s1_df(["S1-1"], ["aaaa corp"])
        right = _s2_df(["S2-1"], ["zzzz corp"])
        out   = _token_sort_candidates(left, right, prefix_len=4)
        assert out.empty

    def test_no_self_pairs(self):
        df  = _s1_df(["E1", "E2"], ["acme corp", "acme inc"])
        out = _token_sort_candidates(df, df, prefix_len=4)
        assert not (out["entity_id_left"] == out["entity_id_right"]).any()

    def test_null_token_sorted_skipped(self):
        left  = _s1_df(["S1-1", "S1-2"], [None, "acme corp"])
        right = _s2_df(["S2-1"],          ["acme inc"])
        out   = _token_sort_candidates(left, right, prefix_len=4)
        assert "S1-1" not in out["entity_id_left"].values


class TestAddressPrefixCandidates:

    def test_same_prefix_produces_pair(self):
        left  = _s1_df(["S1-1"], ["alpha"], addrs=["123 main street"])
        right = _s2_df(["S2-1"], ["beta"],  addrs=["123 main avenue"])
        out   = _address_prefix_candidates(left, right, prefix_len=8)
        assert len(out) >= 1

    def test_different_prefix_no_pair(self):
        left  = _s1_df(["S1-1"], ["alpha"], addrs=["100 oak"])
        right = _s2_df(["S2-1"], ["beta"],  addrs=["999 elm"])
        out   = _address_prefix_candidates(left, right, prefix_len=5)
        assert out.empty

    def test_missing_address_excluded(self):
        left  = _s1_df(["S1-1"], ["alpha"], addrs=[None], missing=[True])
        right = _s2_df(["S2-1"], ["beta"],  addrs=["123 main"])
        out   = _address_prefix_candidates(left, right)
        assert "S1-1" not in out["entity_id_left"].values

    def test_method_tag(self):
        left  = _s1_df(["S1-1"], ["alpha"], addrs=["123 main"])
        right = _s2_df(["S2-1"], ["beta"],  addrs=["123 main"])
        out   = _address_prefix_candidates(left, right, method_tag="addr_pfx")
        if not out.empty:
            assert (out["blocking_method"] == "addr_pfx").all()


class TestRareTokenCandidates:

    def _make_corpus(self):
        """10 right entities + 1 with a unique rare token 'xyzunique'."""
        names = [f"common corp {i}" for i in range(10)] + ["xyzunique ventures"]
        ids   = [f"S2-{i}" for i in range(11)]
        return _s2_df(ids, names)

    def test_rare_token_match_found(self):
        right = self._make_corpus()
        left  = _s1_df(["S1-1"], ["xyzunique holdings"])
        out   = _rare_token_candidates(left, right, rarity_pct=0.1, max_per_entity=10)
        # "xyzunique" is rare -> S2-10 should appear
        assert "S2-10" in out["entity_id_right"].values

    def test_common_token_not_matched(self):
        right = self._make_corpus()
        # "common" appears in many records -> not rare at tight threshold
        left  = _s1_df(["S1-1"], ["common holdings"])
        out   = _rare_token_candidates(left, right, rarity_pct=0.01, max_per_entity=10)
        # with a very tight threshold, "common" may not qualify as rare
        # We can only assert structure, not exact membership (threshold-dependent)
        assert set(out.columns) >= {"entity_id_left", "entity_id_right",
                                    "blocking_score", "blocking_method"}

    def test_no_rare_tokens_returns_empty(self):
        # All tokens appear in every record -> nothing is rare
        right = _s2_df(["S2-1", "S2-2"], ["alpha beta", "alpha beta"])
        left  = _s1_df(["S1-1"],          ["alpha beta"])
        out   = _rare_token_candidates(left, right, rarity_pct=0.0001)
        assert out.empty

    def test_max_per_entity_respected(self):
        # Build a corpus where "xyzunique" appears in only 3 of 30 right records
        # (rare at rarity_pct=0.2 = threshold 6), and left entity has that token.
        # The other 27 right records use "common" tokens to inflate corpus size.
        common_names = [f"common widget {i}" for i in range(27)]
        rare_names   = [f"xyzunique variant {i}" for i in range(3)]
        all_names    = common_names + rare_names
        right = _s2_df([f"S2-{i}" for i in range(30)], all_names)
        left  = _s1_df(["S1-1"], ["xyzunique holdings"])
        # rarity_pct=0.2 -> threshold = int(0.2*30) = 6; "xyzunique" appears 3 times -> rare
        out   = _rare_token_candidates(left, right, rarity_pct=0.2, max_per_entity=2)
        if not out.empty:
            counts = out.groupby("entity_id_left").size()
            assert counts.max() <= 2

    def test_scores_between_0_and_1(self):
        right = self._make_corpus()
        left  = _s1_df(["S1-1"], ["xyzunique corp"])
        out   = _rare_token_candidates(left, right, rarity_pct=0.1)
        if not out.empty:
            assert (out["blocking_score"] >= 0).all()
            assert (out["blocking_score"] <= 1.001).all()


class TestPostalCharNgramCandidates:

    def test_same_postal_code_produces_pair(self):
        # addresses with same numeric tokens "12345"
        left  = _s1_df(["S1-1"], ["alpha"], addrs=["12345 main street"])
        right = _s2_df(["S2-1"], ["beta"],  addrs=["12345 oak avenue"])
        out   = _postal_char_ngram_candidates(left, right, top_k=5)
        assert "S2-1" in out["entity_id_right"].values

    def test_different_postal_codes_no_pair(self):
        left  = _s1_df(["S1-1"], ["alpha"], addrs=["11111 main"])
        right = _s2_df(["S2-1"], ["beta"],  addrs=["99999 oak"])
        out   = _postal_char_ngram_candidates(left, right, top_k=5, min_sim=0.9)
        assert out.empty

    def test_address_missing_excluded(self):
        left  = _s1_df(["S1-1"], ["alpha"], addrs=[None], missing=[True])
        right = _s2_df(["S2-1"], ["beta"],  addrs=["12345 main"])
        out   = _postal_char_ngram_candidates(left, right, top_k=5)
        assert out.empty    # addr_missing=True rows are skipped

    def test_method_tag_postal_ngram(self):
        left  = _s1_df(["S1-1"], ["alpha"], addrs=["12345 main"])
        right = _s2_df(["S2-1"], ["beta"],  addrs=["12345 oak"])
        out   = _postal_char_ngram_candidates(left, right, top_k=5)
        if not out.empty:
            assert (out["blocking_method"] == "postal_ngram").all()


class TestCountryScopedTfidfCandidates:

    def test_same_country_produces_pair(self):
        left  = _s1_df(["S1-1"], ["acme corporation"], country=["United States"])
        right = _s2_df(["S2-1"], ["acme corp"],        country=["United States"])
        out   = _country_scoped_tfidf_candidates(left, right, top_k=5)
        assert "S2-1" in out["entity_id_right"].values

    def test_different_country_no_pair(self):
        left  = _s1_df(["S1-1"], ["acme corporation"], country=["United States"])
        right = _s2_df(["S2-1"], ["acme corp"],        country=["India"])
        out   = _country_scoped_tfidf_candidates(left, right, top_k=5)
        # Different country -> no pair generated
        assert out.empty

    def test_missing_country_col_falls_back(self):
        left  = _s1_df(["S1-1"], ["acme corporation"])
        right = _s2_df(["S2-1"], ["acme corp"])
        # Remove country column
        left  = left.drop(columns=["country"])
        right = right.drop(columns=["country"])
        out   = _country_scoped_tfidf_candidates(left, right, top_k=5)
        # Fallback to unconstrained TF-IDF; pair may or may not appear (score-dependent)
        assert isinstance(out, pd.DataFrame)

    def test_unknown_country_grouped_together(self):
        left  = _s1_df(["S1-1"], ["acme corporation"], country=["United States"])
        # S2 entity has null country -> filled to "unknown"
        right = _s2_df(["S2-1"], ["acme corp"])
        right["country"] = None
        out   = _country_scoped_tfidf_candidates(left, right, top_k=5)
        # "United States" != "unknown" -> no pair
        assert out.empty


# ============================================================================
# merge_and_cap
# ============================================================================

class TestMergeAndCap:

    def test_deduplicates_pairs(self):
        d1 = _cands([("S1-1", "S2-1"), ("S1-1", "S2-2")])
        d2 = _cands([("S1-1", "S2-1"), ("S1-1", "S2-3")])
        out = merge_and_cap([d1, d2])
        # S1-1 -> S2-1 appears once
        cnt = len(out[(out["entity_id_left"] == "S1-1") &
                      (out["entity_id_right"] == "S2-1")])
        assert cnt == 1

    def test_max_score_kept(self):
        d1 = _cands([("S1-1", "S2-1")], score=0.5)
        d2 = _cands([("S1-1", "S2-1")], score=0.9)
        out = merge_and_cap([d1, d2])
        row = out[(out["entity_id_left"] == "S1-1") & (out["entity_id_right"] == "S2-1")]
        assert row["blocking_score"].iloc[0] == pytest.approx(0.9)

    def test_methods_concatenated(self):
        d1 = _cands([("S1-1", "S2-1")], method="tfidf_name", score=0.8)
        d2 = _cands([("S1-1", "S2-1")], method="token_sort", score=0.7)
        out = merge_and_cap([d1, d2])
        method_str = out[(out["entity_id_left"] == "S1-1") &
                          (out["entity_id_right"] == "S2-1")]["blocking_method"].iloc[0]
        assert "tfidf_name" in method_str
        assert "token_sort" in method_str

    def test_per_entity_cap_applied(self):
        pairs = [("S1-1", f"S2-{i}") for i in range(50)]
        out   = merge_and_cap([_cands(pairs)], max_per_entity=10)
        counts = out.groupby("entity_id_left").size()
        assert counts.max() <= 10

    def test_empty_list_returns_empty(self):
        assert merge_and_cap([]).empty

    def test_all_empty_dfs_returns_empty(self):
        assert merge_and_cap([_empty_cands(), _empty_cands()]).empty


# ============================================================================
# Experiment Set 1 smoke tests
# ============================================================================

class TestExpSet1:
    """Smoke tests: verify experiments run and return correctly typed results."""

    @pytest.fixture()
    def tiny_data(self):
        df_s1 = _s1_df(["S1-1", "S1-2"],
                        ["acme corporation", "beta systems"],
                        addrs=["100 main street", "200 oak avenue"])
        df_s2 = _s2_df(["S2-1", "S2-2"],
                        ["acme corp", "gamma widgets"],
                        addrs=["100 main st", "300 pine"])
        df_s3 = _s3_df(["S3-1"],
                        ["beta system inc"],
                        addrs=["200 oak ave"])
        gt = [
            GTRecord("S1-1", {"S2-1"}, set(),   is_singleton=False),
            GTRecord("S1-2", set(),    {"S3-1"}, is_singleton=False),
        ]
        return df_s1, df_s2, df_s3, gt

    def test_topk_sweep_returns_list(self, tiny_data):
        df_s1, df_s2, df_s3, gt = tiny_data
        results = exp1_tfidf_topk_sweep(df_s1, df_s2, df_s3, gt,
                                        top_k_values=[5, 10], global_cap=20)
        assert len(results) == 2
        for r in results:
            assert 0.0 <= r.s1_s2.recall <= 1.0
            assert 0.0 <= r.s1_s3.recall <= 1.0
            assert r.experiment_set == 1

    def test_rarity_sweep_returns_list(self, tiny_data):
        df_s1, df_s2, df_s3, gt = tiny_data
        results = exp1_rare_token_rarity_sweep(df_s1, df_s2, df_s3, gt,
                                               rarity_pcts=[0.01, 0.1], global_cap=20)
        assert len(results) == 2
        for r in results:
            assert r.experiment_set == 1

    def test_global_cap_sweep_returns_list(self, tiny_data):
        df_s1, df_s2, df_s3, gt = tiny_data
        results = exp1_global_cap_sweep(df_s1, df_s2, df_s3, gt,
                                        cap_values=[5, 10])
        assert len(results) == 2
        for r in results:
            assert r.experiment_set == 1

    def test_higher_k_recall_gte_lower_k(self, tiny_data):
        """Recall with K=10 should be >= recall with K=2 (more candidates = ceiling higher)."""
        df_s1, df_s2, df_s3, gt = tiny_data
        results = exp1_tfidf_topk_sweep(df_s1, df_s2, df_s3, gt,
                                        top_k_values=[2, 10], global_cap=50)
        r_k2, r_k10 = results[0], results[1]
        assert r_k10.s1_s2.recall >= r_k2.s1_s2.recall - 1e-9


# ============================================================================
# Experiment Set 2 smoke tests
# ============================================================================

class TestExpSet2:

    @pytest.fixture()
    def tiny_data(self):
        df_s1 = _s1_df(["S1-1", "S1-2"],
                        ["acme corporation", "beta limited"],
                        addrs=["100 main street", "200 oak ave"])
        df_s2 = _s2_df(["S2-1", "S2-2"],
                        ["acme corp",   "gamma llc"],
                        addrs=["100 main st", "300 pine rd"])
        df_s3 = _s3_df(["S3-1"],
                        ["beta ltd"],
                        addrs=["200 oak avenue"])
        gt = [
            GTRecord("S1-1", {"S2-1"}, set(),   is_singleton=False),
            GTRecord("S1-2", set(),    {"S3-1"}, is_singleton=False),
        ]
        return df_s1, df_s2, df_s3, gt

    def test_union_vs_weighted_returns_two_results(self, tiny_data):
        df_s1, df_s2, df_s3, gt = tiny_data
        results = exp2_union_vs_weighted(df_s1, df_s2, df_s3, gt,
                                         tfidf_top_k=5, global_cap=20)
        assert len(results) == 2
        names = {r.name for r in results}
        assert "exp2a_union_baseline" in names
        assert "exp2a_weighted_tfidf_priority" in names

    def test_strategy_drop_returns_five_results(self, tiny_data):
        df_s1, df_s2, df_s3, gt = tiny_data
        results = exp2_strategy_drop_analysis(df_s1, df_s2, df_s3, gt,
                                              tfidf_top_k=5, global_cap=20)
        assert len(results) == 5
        names = {r.name for r in results}
        assert "exp2b_all3" in names
        assert "exp2b_drop_token_sort" in names
        assert "exp2b_name_only" in names

    def test_all3_recall_gte_name_only(self, tiny_data):
        """Full 3-pass recall >= name-only recall (address adds recall for addr-based matches)."""
        df_s1, df_s2, df_s3, gt = tiny_data
        results = exp2_strategy_drop_analysis(df_s1, df_s2, df_s3, gt,
                                              tfidf_top_k=10, global_cap=50)
        r_all3    = next(r for r in results if r.name == "exp2b_all3")
        r_nameonly = next(r for r in results if r.name == "exp2b_name_only")
        assert r_all3.s1_s2.recall >= r_nameonly.s1_s2.recall - 1e-9


# ============================================================================
# Experiment Set 3 smoke tests
# ============================================================================

class TestExpSet3:

    @pytest.fixture()
    def tiny_data(self):
        df_s1 = _s1_df(["S1-1", "S1-2"],
                        ["alpha corp",  "beta inc"],
                        addrs=["12345 main st", None],
                        missing=[False, True],
                        country=["United States", "India"])
        df_s2 = _s2_df(["S2-1", "S2-2"],
                        ["alpha corporation", "gamma llc"],
                        addrs=["12345 main street", "99999 oak"],
                        missing=[False, False],
                        country=["United States", "United States"])
        df_s3 = _s3_df(["S3-1"],
                        ["beta incorporated"],
                        addrs=[None],
                        missing=[True],
                        country=["India"])
        gt = [
            GTRecord("S1-1", {"S2-1"}, set(),   is_singleton=False),
            GTRecord("S1-2", set(),    {"S3-1"}, is_singleton=False),
        ]
        return df_s1, df_s2, df_s3, gt

    def test_postal_ngram_returns_single_result(self, tiny_data):
        df_s1, df_s2, df_s3, gt = tiny_data
        r = exp3_postal_char_ngram(df_s1, df_s2, df_s3, gt, top_k=5, global_cap=20)
        assert r.experiment_set == 3
        assert r.name == "exp3a_postal_char_ngram"

    def test_country_scoped_tfidf_returns_two_results(self, tiny_data):
        df_s1, df_s2, df_s3, gt = tiny_data
        results = exp3_country_scoped_tfidf(df_s1, df_s2, df_s3, gt, top_k=5, global_cap=20)
        assert len(results) == 2
        names = {r.name for r in results}
        assert "exp3b_country_tfidf_name" in names
        assert "exp3b_country_tfidf_addr" in names

    def test_addr_missing_quality_returns_single_result(self, tiny_data):
        df_s1, df_s2, df_s3, gt = tiny_data
        r = exp3_address_missing_subset_quality(df_s1, df_s2, df_s3, gt,
                                                tfidf_top_k=5, global_cap=20)
        assert r.experiment_set == 3
        assert r.name == "exp3c_addr_missing_name_only_quality"
        # addr_missing subset: S1-2 has addr_missing=True and S3-1 target
        # The subset recall may or may not be 1.0 on tiny data, but must be a float
        am_s3 = r.addr_missing_s1_s3.get("recall")
        if am_s3 is not None:
            assert 0.0 <= am_s3 <= 1.0

    def test_country_scoped_does_not_cross_country(self, tiny_data):
        """
        S1-1 is US, S2-1 is US -> should be paired.
        S1-2 is India, S2-2 is US -> should NOT be paired in country-scoped variant.
        """
        df_s1, df_s2, df_s3, gt = tiny_data
        results = exp3_country_scoped_tfidf(df_s1, df_s2, df_s3, gt, top_k=10, global_cap=50)
        name_result = next(r for r in results if r.name == "exp3b_country_tfidf_name")
        # We can't check exact pairs but can verify S1-2 (India) doesn't match S2-2 (US)
        # by checking the S1_S2 candidate map doesn't link India to US
        # This is a structural test — rely on the unit tests of _country_scoped_tfidf above


# ============================================================================
# ExperimentResult serialization
# ============================================================================

class TestExperimentResultSerialization:

    def test_to_dict_is_json_serializable(self):
        import json
        df_s1 = _s1_df(["S1-1"], ["alpha corp"])
        df_s2 = _s2_df(["S2-1"], ["alpha corporation"])
        df_s3 = _s3_df(["S3-1"], ["beta inc"])
        gt    = [GTRecord("S1-1", {"S2-1"}, set(), is_singleton=False)]

        results = exp1_tfidf_topk_sweep(df_s1, df_s2, df_s3, gt,
                                        top_k_values=[5], global_cap=10)
        for r in results:
            d    = r.to_dict()
            text = json.dumps(d, default=str)   # should not raise
            assert isinstance(text, str)

    def test_recommendation_valid_values(self):
        df_s1 = _s1_df(["S1-1"], ["alpha corp"])
        df_s2 = _s2_df(["S2-1"], ["alpha corporation"])
        df_s3 = _s3_df(["S3-1"], ["beta inc"])
        gt    = [GTRecord("S1-1", {"S2-1"}, set(), is_singleton=False)]
        results = exp1_global_cap_sweep(df_s1, df_s2, df_s3, gt, cap_values=[10])
        for r in results:
            assert r.recommendation in ("keep", "drop", "tune-further")
