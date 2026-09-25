"""
features.py
===========
Task B — Feature engineering for entity-resolution candidate pairs.

Design contract
---------------
* Accepts labeled pair DataFrames (output of labeling.py) and normalized
  entity DataFrames (normalized_s{1,2,3}_enriched.parquet).
* Joins normalized text fields onto the labeled pairs before computing features.
* Computes ~28 features per pair across Name, Address, Country, and Candidate
  metadata categories.
* address_missing=True on EITHER side → all address-based features = -1.0
  (sentinel; model distinguishes "address was absent" from "addresses differ").
  Never computes string similarity on empty/None addresses.
* Pure, reusable compute_features(left_record, right_record) function —
  no hardcoded paths, runs unchanged inside a SageMaker Processing Job.
* Outputs:
    {FEATURES}features_labeled_S1_S2.parquet
    {FEATURES}features_labeled_S1_S3.parquet
  and a report:
    {REPORTS}feature_stats_report.json

Feature groups
--------------
  NAME (7):
    name_exact               — 1.0 if normalized_name is identical else 0.0
    name_levenshtein         — Levenshtein similarity  (1 - normalized_distance)
    name_jaro_winkler        — Jaro-Winkler similarity
    name_token_jaccard       — Jaccard similarity of token sets
    name_token_sort_ratio    — RapidFuzz token_sort_ratio / 100 on normalized names
    name_ngram_cosine        — character 2-gram Jaccard (fast proxy for cosine)
    name_length_difference   — abs(len(left_name) - len(right_name)) / max_len

  ADDRESS (7):
    address_exact            — 1.0 if identical normalized_address else 0.0
    address_jaccard          — Jaccard on address token sets
    address_ngram_similarity — character 2-gram Jaccard on addresses
    address_edit_distance    — Levenshtein similarity on addresses
    numeric_token_similarity — Jaccard of numeric tokens only (house numbers etc.)
    postal_similarity        — similarity of the last token (often postal code)
    address_length_difference — abs difference / max_len

  COUNTRY (3):
    country_exact            — 1.0 if normalized_country identical (case-insensitive)
    country_normalized       — 1.0 if both present (non-null), 0.0 if both missing,
                               -1.0 if one is missing
    country_unseen           — 1.0if either side's country was null/empty

  CANDIDATE METADATA (11):
    blocking_score           — raw score from blocking stage
    blocking_method_tfidf    — 1.0 if 'tfidf' in blocking_method else 0.0
    blocking_method_token_sort — 1.0 if 'token_sort' in blocking_method else 0.0
    blocking_method_addr_prefix — 1.0 if 'address_prefix' in blocking_method else 0.0
    candidate_rank           — position within source1_entity_id's candidate list
                               (0-indexed, sorted by blocking_score desc)
    candidate_margin         — score gap to next-best candidate for same entity
    candidate_source_s2      — 1.0 if candidate_source == 's2' else 0.0
    candidate_source_s3      — 1.0 if candidate_source == 's3' else 0.0
    address_missing_left     — 1.0 if left entity's address_missing is True
    address_missing_right    — 1.0 if right entity's address_missing is True
    address_missing_either   — 1.0 if either side has address_missing

Authors: Account 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import distance as rfdist
from rapidfuzz import fuzz as rffuzz

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entity_resolution.config import (
    add_account_prefix_arg,
    resolve_config_from_args,
)

logger = logging.getLogger(__name__)

_ADDR_SENTINEL = -1.0   # sentinel for address features when address_missing
_COUNTRY_PARTIAL_MISSING = -1.0  # sentinel when exactly one country is missing


# ---------------------------------------------------------------------------
# Low-level string helpers
# ---------------------------------------------------------------------------

def _token_set(s: str) -> set:
    """Split string into a set of lower-case tokens."""
    return set(s.lower().split()) if s else set()


def _ngram_set(s: str, n: int = 2) -> set:
    """Character n-gram set of a string."""
    s = s.lower()
    return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity: |A∩B| / |A∪B|."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def _numeric_tokens(s: str) -> set:
    """Extract numeric sub-tokens from an address string."""
    return set(re.findall(r"\b\d+\b", s)) if s else set()


def _last_token(s: str) -> str:
    """Return the last whitespace token (often postal code)."""
    tokens = s.split()
    return tokens[-1] if tokens else ""


def _safe_len(s: Optional[str]) -> int:
    return len(s) if s else 0


# ---------------------------------------------------------------------------
# Per-record-pair feature computation (pure function — no I/O)
# ---------------------------------------------------------------------------

def compute_features(
    left:  Dict[str, Any],
    right: Dict[str, Any],
    blocking_score:  float = 0.0,
    blocking_method: str   = "",
    candidate_rank:  int   = 0,
    candidate_margin: float = 0.0,
    candidate_source: str  = "",
) -> Dict[str, float]:
    """
    Compute all features for a single (left, right) entity pair.

    Parameters
    ----------
    left / right : dicts with keys:
        normalized_name, normalized_address, normalized_country, address_missing
    blocking_score, blocking_method, candidate_rank, candidate_margin,
    candidate_source : metadata from the blocking/candidate table.

    Returns
    -------
    Dict[str, float] — one entry per feature (all floats, never NaN).
    """
    feats: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Unpack fields (safe defaults for missing values)                    #
    # ------------------------------------------------------------------ #
    l_name   = (left.get("normalized_name")    or "").strip()
    r_name   = (right.get("normalized_name")   or "").strip()
    l_addr   = (left.get("normalized_address") or "").strip()
    r_addr   = (right.get("normalized_address") or "").strip()
    l_cntry  = (left.get("normalized_country") or "").strip().lower()
    r_cntry  = (right.get("normalized_country") or "").strip().lower()
    l_addr_missing = bool(left.get("address_missing",  False))
    r_addr_missing = bool(right.get("address_missing", False))

    # ------------------------------------------------------------------ #
    # NAME features (7)                                                   #
    # ------------------------------------------------------------------ #
    if l_name and r_name:
        feats["name_exact"] = 1.0 if l_name == r_name else 0.0

        # Levenshtein similarity via RapidFuzz
        lev = rfdist.Levenshtein.normalized_similarity(l_name, r_name)
        feats["name_levenshtein"] = float(lev)

        # Jaro-Winkler
        feats["name_jaro_winkler"] = float(rffuzz.WRatio(l_name, r_name) / 100.0)

        # Token Jaccard
        l_tok = _token_set(l_name)
        r_tok = _token_set(r_name)
        feats["name_token_jaccard"] = _jaccard(l_tok, r_tok)

        # Token sort ratio (RapidFuzz)
        feats["name_token_sort_ratio"] = float(rffuzz.token_sort_ratio(l_name, r_name) / 100.0)

        # Character 2-gram Jaccard (fast cosine proxy)
        feats["name_ngram_cosine"] = _jaccard(_ngram_set(l_name, 2), _ngram_set(r_name, 2))

        # Length difference (normalised 0–1, higher = more different)
        max_len = max(_safe_len(l_name), _safe_len(r_name), 1)
        feats["name_length_difference"] = abs(_safe_len(l_name) - _safe_len(r_name)) / max_len

    else:
        # At least one name is empty — use 0.0 for all name features
        for k in ("name_exact", "name_levenshtein", "name_jaro_winkler",
                  "name_token_jaccard", "name_token_sort_ratio",
                  "name_ngram_cosine", "name_length_difference"):
            feats[k] = 0.0

    # ------------------------------------------------------------------ #
    # ADDRESS features (7) — sentinel -1.0 when either side is missing   #
    # ------------------------------------------------------------------ #
    addr_either_missing = l_addr_missing or r_addr_missing

    if addr_either_missing:
        for k in ("address_exact", "address_jaccard", "address_ngram_similarity",
                  "address_edit_distance", "numeric_token_similarity",
                  "postal_similarity", "address_length_difference"):
            feats[k] = _ADDR_SENTINEL
    elif l_addr and r_addr:
        feats["address_exact"] = 1.0 if l_addr == r_addr else 0.0

        l_atk = _token_set(l_addr)
        r_atk = _token_set(r_addr)
        feats["address_jaccard"] = _jaccard(l_atk, r_atk)

        feats["address_ngram_similarity"] = _jaccard(
            _ngram_set(l_addr, 2), _ngram_set(r_addr, 2)
        )

        feats["address_edit_distance"] = float(
            rfdist.Levenshtein.normalized_similarity(l_addr, r_addr)
        )

        feats["numeric_token_similarity"] = _jaccard(
            _numeric_tokens(l_addr), _numeric_tokens(r_addr)
        )

        l_last = _last_token(l_addr)
        r_last = _last_token(r_addr)
        feats["postal_similarity"] = 1.0 if l_last == r_last and l_last else 0.0

        max_alen = max(_safe_len(l_addr), _safe_len(r_addr), 1)
        feats["address_length_difference"] = (
            abs(_safe_len(l_addr) - _safe_len(r_addr)) / max_alen
        )
    else:
        # Both present but normalised to empty — unlikely but safe fallback
        for k in ("address_exact", "address_jaccard", "address_ngram_similarity",
                  "address_edit_distance", "numeric_token_similarity",
                  "postal_similarity", "address_length_difference"):
            feats[k] = 0.0

    # ------------------------------------------------------------------ #
    # COUNTRY features (3)                                                #
    # ------------------------------------------------------------------ #
    l_has_cntry = bool(l_cntry)
    r_has_cntry = bool(r_cntry)

    feats["country_unseen"] = 1.0 if (not l_has_cntry or not r_has_cntry) else 0.0

    if l_has_cntry and r_has_cntry:
        feats["country_exact"]      = 1.0 if l_cntry == r_cntry else 0.0
        feats["country_normalized"] = 1.0 if l_cntry == r_cntry else 0.0
    elif not l_has_cntry and not r_has_cntry:
        feats["country_exact"]      = 1.0   # both missing → treat as "same"
        feats["country_normalized"] = 0.0   # but can't confirm match
    else:
        feats["country_exact"]      = 0.0
        feats["country_normalized"] = _COUNTRY_PARTIAL_MISSING  # one missing

    # ------------------------------------------------------------------ #
    # CANDIDATE METADATA features (11)                                    #
    # ------------------------------------------------------------------ #
    feats["blocking_score"]             = float(blocking_score)
    feats["blocking_method_tfidf"]      = 1.0 if "tfidf"          in blocking_method else 0.0
    feats["blocking_method_token_sort"] = 1.0 if "token_sort"     in blocking_method else 0.0
    feats["blocking_method_addr_prefix"]= 1.0 if "address_prefix" in blocking_method else 0.0
    feats["candidate_rank"]             = float(candidate_rank)
    feats["candidate_margin"]           = float(candidate_margin)
    feats["candidate_source_s2"]        = 1.0 if candidate_source == "s2" else 0.0
    feats["candidate_source_s3"]        = 1.0 if candidate_source == "s3" else 0.0
    feats["address_missing_left"]       = 1.0 if l_addr_missing else 0.0
    feats["address_missing_right"]      = 1.0 if r_addr_missing else 0.0
    feats["address_missing_either"]     = 1.0 if addr_either_missing else 0.0

    return feats


# ---------------------------------------------------------------------------
# Batch feature computation
# ---------------------------------------------------------------------------

def _build_entity_lookup(norm_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build {entity_id -> record_dict} lookup from a normalized DataFrame.
    Fields: normalized_name, normalized_address, normalized_country, address_missing.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in norm_df[["entity_id", "normalized_name", "normalized_address",
                         "normalized_country", "address_missing"]].itertuples(index=False):
        lookup[row.entity_id] = {
            "normalized_name":    row.normalized_name,
            "normalized_address": row.normalized_address,
            "normalized_country": row.normalized_country,
            "address_missing":    row.address_missing,
        }
    return lookup


def _compute_rank_and_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candidate_rank (0-indexed, descending by blocking_score within each
    source1_entity_id) and candidate_margin (score gap to next-best candidate).
    """
    df = df.copy()
    df = df.sort_values(
        ["source1_entity_id", "blocking_score"], ascending=[True, False]
    ).reset_index(drop=True)

    # Rank within entity
    df["candidate_rank"] = df.groupby("source1_entity_id").cumcount()

    # Margin: score - next_score, 0.0 for the lowest-ranked candidate
    def _margin(group: pd.DataFrame) -> pd.Series:  # group excludes grouping col — use reset_index
        scores = group["blocking_score"].values
        margins = np.zeros(len(scores))
        margins[:-1] = scores[:-1] - scores[1:]
        return pd.Series(margins, index=group.index)

    df["candidate_margin"] = df.groupby("source1_entity_id", group_keys=False).apply(
        _margin, include_groups=False
    )
    return df


def compute_features_batch(
    labeled_pairs: pd.DataFrame,
    norm_left: pd.DataFrame,    # normalized_s1_*.parquet
    norm_right: pd.DataFrame,   # normalized_s2_*.parquet or normalized_s3_*.parquet
) -> pd.DataFrame:
    """
    Compute features for all pairs in labeled_pairs.

    Parameters
    ----------
    labeled_pairs : output of labeling.py — must have columns:
        source1_entity_id, candidate_entity_id, candidate_source,
        blocking_score, blocking_method, label, negative_type
    norm_left     : normalized DataFrame for the left (S1) entities
    norm_right    : normalized DataFrame for the right (S2 or S3) entities

    Returns
    -------
    DataFrame with all original columns + computed feature columns + 'label'.
    """
    logger.info("Building entity lookup tables ...")
    left_lookup  = _build_entity_lookup(norm_left)
    right_lookup = _build_entity_lookup(norm_right)

    logger.info("Computing rank and margin columns ...")
    df = _compute_rank_and_margin(labeled_pairs)

    logger.info("Computing %d feature rows ...", len(df))
    feature_rows: List[Dict[str, float]] = []

    for row in df.itertuples(index=False):
        left_rec  = left_lookup.get(row.source1_entity_id,   {})
        right_rec = right_lookup.get(row.candidate_entity_id, {})

        feats = compute_features(
            left            = left_rec,
            right           = right_rec,
            blocking_score  = float(row.blocking_score),
            blocking_method = str(row.blocking_method),
            candidate_rank  = int(row.candidate_rank),
            candidate_margin = float(row.candidate_margin),
            candidate_source = str(row.candidate_source),
        )
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)

    # blocking_score already exists in df (from labeled_pairs) — drop the duplicate
    # from feat_df to avoid column collision when concatenating.
    cols_in_df   = set(df.columns)
    cols_to_drop = [c for c in feat_df.columns if c in cols_in_df]
    if cols_to_drop:
        feat_df = feat_df.drop(columns=cols_to_drop)

    result = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    # Verify no NaN features
    null_counts = feat_df.isnull().sum()
    nulls_found = null_counts[null_counts > 0]
    if not nulls_found.empty:
        logger.warning("Null features found (should be zero):\n%s", nulls_found)

    logger.info("Feature computation complete: %d rows x %d feature cols",
                len(result), len(feat_df.columns))
    return result


# ---------------------------------------------------------------------------
# Feature stats report
# ---------------------------------------------------------------------------

_FEATURE_NAMES = [
    "name_exact", "name_levenshtein", "name_jaro_winkler", "name_token_jaccard",
    "name_token_sort_ratio", "name_ngram_cosine", "name_length_difference",
    "address_exact", "address_jaccard", "address_ngram_similarity",
    "address_edit_distance", "numeric_token_similarity", "postal_similarity",
    "address_length_difference",
    "country_exact", "country_normalized", "country_unseen",
    "blocking_score", "blocking_method_tfidf", "blocking_method_token_sort",
    "blocking_method_addr_prefix", "candidate_rank", "candidate_margin",
    "candidate_source_s2", "candidate_source_s3",
    "address_missing_left", "address_missing_right", "address_missing_either",
]


def compute_feature_stats_report(
    feat_s1_s2: pd.DataFrame,
    feat_s1_s3: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute min/max/mean/null-rate per feature across both pair types.
    Flag zero-variance, always-null, and out-of-range features.
    """
    report: Dict[str, Any] = {
        "feature_stats": {},
        "flags": {
            "zero_variance": [],
            "always_null":   [],
            "out_of_range":  [],
        }
    }

    combined = pd.concat([
        feat_s1_s2[_FEATURE_NAMES].assign(_split="S1_S2"),
        feat_s1_s3[_FEATURE_NAMES].assign(_split="S1_S3"),
    ], ignore_index=True)

    for feat in _FEATURE_NAMES:
        col = combined[feat]
        null_rate = float(col.isnull().mean())
        if null_rate == 1.0:
            report["flags"]["always_null"].append(feat)
            report["feature_stats"][feat] = {"null_rate": null_rate}
            continue

        non_null = col.dropna()
        fmin  = float(non_null.min())
        fmax  = float(non_null.max())
        fmean = float(non_null.mean())
        fstd  = float(non_null.std())

        stats = {
            "min":       round(fmin,  6),
            "max":       round(fmax,  6),
            "mean":      round(fmean, 6),
            "std":       round(fstd,  6),
            "null_rate": round(null_rate, 6),
        }
        report["feature_stats"][feat] = stats

        # Flag zero-variance
        if fstd == 0.0:
            report["flags"]["zero_variance"].append(feat)

        # Flag out-of-range (features should be in [-1, 1] range).
        # Exempt ordinal / unbounded metadata features whose scale is not [-1, 1].
        _UNBOUNDED_FEATURES = {"candidate_rank", "candidate_margin"}
        if feat not in _UNBOUNDED_FEATURES and (fmin < -1.001 or fmax > 1.001):
            report["flags"]["out_of_range"].append({
                "feature": feat,
                "min": round(fmin, 6),
                "max": round(fmax, 6),
            })

    return report


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    from src.entity_resolution.utils.io import read_parquet_s3, write_parquet_s3, write_json_s3

    parser = argparse.ArgumentParser(description="Task B — feature engineering for labeled pairs.")
    add_account_prefix_arg(parser)
    parser.add_argument("--aws-profile",  default="amazon-ml-account3")
    parser.add_argument("--labeled-s1-s2",
                        default="s3://amzn-s3-ml-c/account3/features/labeled_pairs_S1_S2.parquet")
    parser.add_argument("--labeled-s1-s3",
                        default="s3://amzn-s3-ml-c/account3/features/labeled_pairs_S1_S3.parquet")
    parser.add_argument("--norm-s1",
                        default="s3://amzn-s3-ml-c/account3/processed/normalized_s1_enriched.parquet")
    parser.add_argument("--norm-s2",
                        default="s3://amzn-s3-ml-c/account3/processed/normalized_s2_enriched.parquet")
    parser.add_argument("--norm-s3",
                        default="s3://amzn-s3-ml-c/account3/processed/normalized_s3_enriched.parquet")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cfg = resolve_config_from_args(args)
    paths = cfg.paths
    profile = cfg.aws_profile

    # Load normalized lookup tables
    logger.info("Loading normalized S1/S2/S3 enriched parquets ...")
    norm_s1 = read_parquet_s3(args.norm_s1, profile=profile)
    norm_s2 = read_parquet_s3(args.norm_s2, profile=profile)
    norm_s3 = read_parquet_s3(args.norm_s3, profile=profile)

    # Load labeled pairs
    logger.info("Loading labeled pairs ...")
    labeled_s1_s2 = read_parquet_s3(args.labeled_s1_s2, profile=profile)
    labeled_s1_s3 = read_parquet_s3(args.labeled_s1_s3, profile=profile)

    # Compute features
    logger.info("Computing S1×S2 features ...")
    feat_s1_s2 = compute_features_batch(labeled_s1_s2, norm_s1, norm_s2)

    logger.info("Computing S1×S3 features ...")
    feat_s1_s3 = compute_features_batch(labeled_s1_s3, norm_s1, norm_s3)

    # Upload
    out_s1_s2 = f"{paths.FEATURES}features_labeled_S1_S2.parquet"
    out_s1_s3 = f"{paths.FEATURES}features_labeled_S1_S3.parquet"
    write_parquet_s3(feat_s1_s2, out_s1_s2, profile=profile)
    write_parquet_s3(feat_s1_s3, out_s1_s3, profile=profile)
    logger.info("Feature files uploaded: %s | %s", out_s1_s2, out_s1_s3)

    # Save local copies
    feat_s1_s2.to_parquet("/tmp/features_labeled_S1_S2.parquet", index=False)
    feat_s1_s3.to_parquet("/tmp/features_labeled_S1_S3.parquet", index=False)

    # Stats report
    logger.info("Computing feature stats report ...")
    stats_report = compute_feature_stats_report(feat_s1_s2, feat_s1_s3)
    report_uri = f"{paths.REPORTS}feature_stats_report.json"
    write_json_s3(stats_report, report_uri, profile=profile)
    logger.info("Feature stats report uploaded to %s", report_uri)

    print("\n" + "=" * 70)
    print("TASK B — FEATURE STATS REPORT")
    print("=" * 70)
    print(json.dumps(stats_report, indent=2))


if __name__ == "__main__":
    main()
