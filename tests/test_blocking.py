"""
tests/test_blocking.py
======================
Unit tests for src/entity_resolution/blocking.py and
src/entity_resolution/candidate_recall.py.

All tests use tiny synthetic DataFrames — no S3 / AWS credentials required.
Run with:
    pytest tests/test_blocking.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.entity_resolution.blocking import (
    BlockingConfig,
    _empty_candidates,
    generate_address_prefix_candidates,
    generate_tfidf_candidates,
    generate_token_sort_candidates,
    merge_candidates,
    run_blocking_pair,
)
from src.entity_resolution.candidate_recall import (
    RecallResult,
    compute_recall,
    coverage_by_method,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_cfg() -> BlockingConfig:
    return BlockingConfig(
        tfidf_top_k=10,
        ngram_min=2,
        ngram_max=3,
        tfidf_min_sim=0.01,
        prefix_len=4,
        addr_prefix_len=6,
        max_candidates_per_entity=50,
    )


def _source_df(
    ids:    list,
    names:  list,
    addrs:  list | None = None,
    missing: list | None = None,
) -> pd.DataFrame:
    """Build a minimal normalized DataFrame."""
    n = len(ids)
    addrs   = addrs   or ["123 main street"] * n
    missing = missing or [False] * n
    tsorted = [" ".join(sorted(str(nm).split())) if nm else None for nm in names]
    return pd.DataFrame({
        "entity_id":          ids,
        "normalized_name":    names,
        "token_sorted_name":  tsorted,
        "normalized_address": addrs,
        "address_missing":    missing,
    })


# ---------------------------------------------------------------------------
# BlockingConfig
# ---------------------------------------------------------------------------

class TestBlockingConfig:
    def test_defaults(self):
        cfg = BlockingConfig()
        assert cfg.tfidf_top_k == 50
        assert cfg.ngram_min   == 2
        assert cfg.ngram_max   == 3
        assert cfg.prefix_len  == 5

    def test_from_config_dict(self):
        raw = {"blocking": {"tfidf_top_k": 20, "ngram_size": 4, "max_candidates_per_entity": 200}}
        cfg = BlockingConfig.from_config_dict(raw)
        assert cfg.tfidf_top_k == 20
        assert cfg.ngram_max   == 4
        assert cfg.max_candidates_per_entity == 200

    def test_from_empty_dict_uses_defaults(self):
        cfg = BlockingConfig.from_config_dict({})
        assert cfg.tfidf_top_k == 50


# ---------------------------------------------------------------------------
# TF-IDF blocking
# ---------------------------------------------------------------------------

class TestTfidfCandidates:

    def test_returns_dataframe_with_correct_columns(self, default_cfg):
        left  = _source_df(["A1", "A2"], ["acme corporation", "beta widgets"])
        right = _source_df(["B1", "B2"], ["acme corp", "gamma services"])
        out   = generate_tfidf_candidates(left, right, default_cfg)
        assert set(out.columns) >= {"entity_id_left", "entity_id_right", "blocking_score", "blocking_method"}

    def test_similar_names_produce_candidates(self, default_cfg):
        left  = _source_df(["A1"], ["acme corporation"])
        right = _source_df(["B1", "B2"], ["acme corp", "xyz zyx abc"])
        out   = generate_tfidf_candidates(left, right, default_cfg)
        # "acme corp" is more similar to "acme corporation" — should appear
        found_rids = set(out["entity_id_right"].tolist())
        assert "B1" in found_rids

    def test_method_label_is_tfidf(self, default_cfg):
        left  = _source_df(["A1"], ["acme corporation"])
        right = _source_df(["B1"], ["acme corp"])
        out   = generate_tfidf_candidates(left, right, default_cfg)
        if not out.empty:
            assert (out["blocking_method"] == "tfidf").all()

    def test_no_self_pairs(self, default_cfg):
        """Same entity appearing in both left and right should not self-pair."""
        df = _source_df(["E1", "E2"], ["acme corp", "beta inc"])
        out = generate_tfidf_candidates(df, df, default_cfg)
        assert not ((out["entity_id_left"] == out["entity_id_right"])).any()

    def test_empty_left_returns_empty(self, default_cfg):
        left  = _source_df([], [])
        right = _source_df(["B1"], ["acme corp"])
        out   = generate_tfidf_candidates(left, right, default_cfg)
        assert out.empty

    def test_all_null_names_returns_empty(self, default_cfg):
        left  = _source_df(["A1"], [None])
        right = _source_df(["B1"], [None])
        out   = generate_tfidf_candidates(left, right, default_cfg)
        assert out.empty

    def test_scores_between_0_and_1(self, default_cfg):
        left  = _source_df(["A1", "A2"], ["acme corporation", "beta systems"])
        right = _source_df(["B1", "B2"], ["acme corp",        "gamma widgets"])
        out   = generate_tfidf_candidates(left, right, default_cfg)
        if not out.empty:
            assert (out["blocking_score"] >= 0).all()
            assert (out["blocking_score"] <= 1.0001).all()   # float tolerance

    def test_top_k_respected(self, default_cfg):
        """Each left entity should have at most top_k candidates."""
        names_right = [f"acme variant {i}" for i in range(20)]
        left  = _source_df(["A1"], ["acme corporation"])
        right = _source_df([f"B{i}" for i in range(20)], names_right)
        out   = generate_tfidf_candidates(left, right, default_cfg)
        assert len(out) <= default_cfg.tfidf_top_k


# ---------------------------------------------------------------------------
# Token-sort blocking
# ---------------------------------------------------------------------------

class TestTokenSortCandidates:

    def test_same_prefix_produces_pair(self, default_cfg):
        # "acme beta" and "acme gamma" share prefix "acme" (4 chars)
        left  = _source_df(["A1"], ["acme beta"])
        right = _source_df(["B1"], ["acme gamma"])
        out   = generate_token_sort_candidates(left, right, default_cfg)
        assert len(out) >= 1

    def test_different_prefix_no_pair(self, default_cfg):
        left  = _source_df(["A1"], ["acme corp"])
        right = _source_df(["B1"], ["zzzz corp"])
        out   = generate_token_sort_candidates(left, right, default_cfg)
        assert out.empty

    def test_method_label(self, default_cfg):
        left  = _source_df(["A1"], ["acme corp"])
        right = _source_df(["B1"], ["acme inc"])
        out   = generate_token_sort_candidates(left, right, default_cfg)
        if not out.empty:
            assert (out["blocking_method"] == "token_sort").all()

    def test_no_self_pairs(self, default_cfg):
        df = _source_df(["E1", "E2"], ["acme corp", "acme inc"])
        out = generate_token_sort_candidates(df, df, default_cfg)
        assert not (out["entity_id_left"] == out["entity_id_right"]).any()

    def test_token_sort_handles_word_order(self, default_cfg):
        # "beta alpha" and "alpha beta" → same sorted form → same shard
        left  = _source_df(["A1"], ["beta alpha"])
        right = _source_df(["B1"], ["alpha beta"])
        # Both have token_sorted = "alpha beta" — should match on shard
        out   = generate_token_sort_candidates(left, right, default_cfg)
        assert len(out) >= 1

    def test_empty_input(self, default_cfg):
        out = generate_token_sort_candidates(_source_df([], []), _source_df([], []), default_cfg)
        assert out.empty

    def test_null_token_sorted_skipped(self, default_cfg):
        left  = _source_df(["A1", "A2"], [None, "acme corp"])
        right = _source_df(["B1"],       ["acme inc"])
        out   = generate_token_sort_candidates(left, right, default_cfg)
        # A1 (null token_sorted) should not appear in candidates
        assert "A1" not in out["entity_id_left"].values


# ---------------------------------------------------------------------------
# Address-prefix blocking
# ---------------------------------------------------------------------------

class TestAddressPrefixCandidates:

    def test_same_address_prefix_produces_pair(self, default_cfg):
        left  = _source_df(["A1"], ["acme"], ["123 ma"])
        right = _source_df(["B1"], ["beta"], ["123 ma"])
        out   = generate_address_prefix_candidates(left, right, default_cfg)
        assert len(out) >= 1

    def test_different_prefix_no_pair(self, default_cfg):
        left  = _source_df(["A1"], ["acme"], ["100 oak"])
        right = _source_df(["B1"], ["beta"], ["999 elm"])
        out   = generate_address_prefix_candidates(left, right, default_cfg)
        assert out.empty

    def test_missing_address_rows_excluded(self, default_cfg):
        left  = _source_df(["A1"], ["acme"], [None],     [True])
        right = _source_df(["B1"], ["beta"], ["123 main"], [False])
        out   = generate_address_prefix_candidates(left, right, default_cfg)
        # A1 has address_missing=True → must not appear
        assert "A1" not in out["entity_id_left"].values

    def test_method_label(self, default_cfg):
        left  = _source_df(["A1"], ["acme"], ["123 ma"])
        right = _source_df(["B1"], ["beta"], ["123 ma"])
        out   = generate_address_prefix_candidates(left, right, default_cfg)
        if not out.empty:
            assert (out["blocking_method"] == "address_prefix").all()


# ---------------------------------------------------------------------------
# merge_candidates
# ---------------------------------------------------------------------------

class TestMergeCandidates:

    def _make_cands(self, pairs: list, method: str = "tfidf", score: float = 0.8) -> pd.DataFrame:
        return pd.DataFrame({
            "entity_id_left":  [p[0] for p in pairs],
            "entity_id_right": [p[1] for p in pairs],
            "blocking_score":  [score] * len(pairs),
            "blocking_method": [method] * len(pairs),
        })

    def test_basic_merge_deduplicates(self):
        d1 = self._make_cands([("A", "B"), ("C", "D")])
        d2 = self._make_cands([("A", "B"), ("E", "F")])
        out = merge_candidates([d1, d2])
        # (A,B) should appear exactly once
        ab_rows = out[
            (out["entity_id_left"] == "A") & (out["entity_id_right"] == "B")
        ]
        assert len(ab_rows) == 1

    def test_direction_canonicalization(self):
        # (A,B) and (B,A) should collapse to one row
        d1 = self._make_cands([("A", "B")])
        d2 = self._make_cands([("B", "A")])
        out = merge_candidates([d1, d2])
        assert len(out) == 1

    def test_max_score_kept(self):
        d1 = self._make_cands([("A", "B")], score=0.5)
        d2 = self._make_cands([("A", "B")], score=0.9)
        out = merge_candidates([d1, d2])
        assert out["blocking_score"].iloc[0] == pytest.approx(0.9)

    def test_methods_combined(self):
        d1 = self._make_cands([("A", "B")], method="tfidf",      score=0.8)
        d2 = self._make_cands([("A", "B")], method="token_sort", score=0.7)
        out = merge_candidates([d1, d2])
        methods = out["blocking_method"].iloc[0]
        assert "tfidf" in methods
        assert "token_sort" in methods

    def test_per_entity_cap(self):
        pairs = [("A", f"X{i}") for i in range(200)]
        d1    = self._make_cands(pairs)
        out   = merge_candidates([d1], max_per_entity=10)
        counts = out.groupby("entity_id_left").size()
        assert counts.max() <= 10

    def test_empty_list_returns_empty(self):
        out = merge_candidates([])
        assert out.empty

    def test_all_empty_dfs_returns_empty(self):
        out = merge_candidates([_empty_candidates(), _empty_candidates()])
        assert out.empty


# ---------------------------------------------------------------------------
# run_blocking_pair (integration)
# ---------------------------------------------------------------------------

class TestRunBlockingPair:

    def test_returns_dataframe(self, default_cfg):
        df = _source_df(
            ["A1", "A2", "A3"],
            ["acme corporation", "beta systems", "gamma tech"],
        )
        out = run_blocking_pair(df, df, default_cfg, same_source=True)
        assert isinstance(out, pd.DataFrame)
        assert set(out.columns) >= {"entity_id_left", "entity_id_right", "blocking_score", "blocking_method"}

    def test_no_self_pairs_same_source(self, default_cfg):
        df = _source_df(["E1", "E2", "E3"], ["acme corp", "beta inc", "gamma llc"])
        out = run_blocking_pair(df, df, default_cfg, same_source=True)
        assert not (out["entity_id_left"] == out["entity_id_right"]).any()

    def test_empty_df_returns_empty(self, default_cfg):
        df  = _source_df([], [])
        out = run_blocking_pair(df, df, default_cfg)
        assert out.empty


# ---------------------------------------------------------------------------
# compute_recall
# ---------------------------------------------------------------------------

class TestComputeRecall:

    def _cands(self, pairs: list) -> pd.DataFrame:
        return pd.DataFrame({
            "entity_id_left":  [p[0] for p in pairs],
            "entity_id_right": [p[1] for p in pairs],
            "blocking_score":  [1.0] * len(pairs),
            "blocking_method": ["tfidf"] * len(pairs),
        })

    def _gt(self, pairs: list) -> pd.DataFrame:
        return pd.DataFrame({
            "entity_id_1": [p[0] for p in pairs],
            "entity_id_2": [p[1] for p in pairs],
        })

    def test_perfect_recall(self):
        gt    = self._gt([("A", "B"), ("C", "D")])
        cands = self._cands([("A", "B"), ("C", "D")])
        result = compute_recall(cands, gt)
        assert result.recall == pytest.approx(1.0)
        assert result.found_pairs == 2
        assert result.missing_pairs == 0

    def test_zero_recall(self):
        gt    = self._gt([("A", "B")])
        cands = self._cands([("X", "Y")])
        result = compute_recall(cands, gt)
        assert result.recall == pytest.approx(0.0)
        assert result.missing_pairs == 1

    def test_partial_recall(self):
        gt    = self._gt([("A", "B"), ("C", "D")])
        cands = self._cands([("A", "B"), ("X", "Y")])
        result = compute_recall(cands, gt)
        assert result.recall == pytest.approx(0.5)

    def test_direction_agnostic(self):
        # GT has (A, B); candidates have (B, A) — should count as found
        gt    = self._gt([("A", "B")])
        cands = self._cands([("B", "A")])
        result = compute_recall(cands, gt)
        assert result.recall == pytest.approx(1.0)

    def test_empty_candidates_zero_recall(self):
        gt     = self._gt([("A", "B")])
        result = compute_recall(_empty_candidates(), gt)
        assert result.recall == pytest.approx(0.0)

    def test_empty_gt_zero_over_zero(self):
        cands  = self._cands([("A", "B")])
        gt     = self._gt([])
        result = compute_recall(cands, gt)
        assert result.recall == pytest.approx(0.0)

    def test_missing_df_contains_uncovered_pairs(self):
        gt    = self._gt([("A", "B"), ("C", "D")])
        cands = self._cands([("A", "B")])
        result = compute_recall(cands, gt)
        assert len(result.missing_df) == 1

    def test_precision_formula(self):
        gt    = self._gt([("A", "B")])
        cands = self._cands([("A", "B"), ("X", "Y"), ("P", "Q")])
        result = compute_recall(cands, gt)
        # precision = found(1) / total_cands(3)
        assert result.precision == pytest.approx(1 / 3)

    def test_avg_candidates_per_entity(self):
        cands = self._cands([("A", "B"), ("A", "C")])
        gt    = self._gt([("A", "B")])
        result = compute_recall(cands, gt)
        assert result.avg_candidates_per_entity == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# coverage_by_method
# ---------------------------------------------------------------------------

class TestCoverageByMethod:

    def test_single_method_coverage(self):
        cands = pd.DataFrame({
            "entity_id_left":  ["A", "X"],
            "entity_id_right": ["B", "Y"],
            "blocking_score":  [1.0, 1.0],
            "blocking_method": ["tfidf", "tfidf"],
        })
        gt = pd.DataFrame({"entity_id_1": ["A"], "entity_id_2": ["B"]})
        df = coverage_by_method(cands, gt)
        assert "tfidf" in df["method"].values
        row = df[df["method"] == "tfidf"].iloc[0]
        assert row["gt_pairs_found"] == 1

    def test_multiple_methods_reported(self):
        cands = pd.DataFrame({
            "entity_id_left":  ["A", "C"],
            "entity_id_right": ["B", "D"],
            "blocking_score":  [1.0, 1.0],
            "blocking_method": ["tfidf", "token_sort"],
        })
        gt = pd.DataFrame({
            "entity_id_1": ["A", "C"],
            "entity_id_2": ["B", "D"],
        })
        df = coverage_by_method(cands, gt)
        methods = set(df["method"].tolist())
        assert "tfidf" in methods
        assert "token_sort" in methods
