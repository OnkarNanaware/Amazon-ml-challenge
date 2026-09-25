"""
blocking.py
===========
Step 3 of the Amazon-ML-Challenge entity-resolution pipeline.

Generates candidate pairs from normalized entity records.
Uses a two-pass strategy for maximum recall:

  Pass 1 — TF-IDF cosine blocking on character n-grams of normalized_name.
            Top-K most similar pairs per entity (configurable via config.yaml).
            Works well for name variations, abbreviations, and transpositions.

  Pass 2 — Token-sort prefix blocking on token_sorted_name.
            Groups entities whose sorted-token representation shares a common
            prefix shard.  Cheap and high-recall for near-identical names.

  Pass 3 — Address-prefix blocking (optional, for entities WITH addresses).
            Groups entities whose normalized_address shares a numeric-token
            prefix (street number + first address token).

Results are merged, deduplicated, and written to
  {CANDIDATES}candidates_<tag>.parquet

Null-safety:
  Entities with address_missing=True are routed through name-only passes
  (Pass 1 + Pass 2 only), never into Pass 3.

Public API (pure functions, no S3 deps):
  generate_tfidf_candidates(df_left, df_right, cfg) -> pd.DataFrame
  generate_token_sort_candidates(df_left, df_right, cfg) -> pd.DataFrame
  generate_address_prefix_candidates(df_left, df_right, cfg) -> pd.DataFrame
  merge_candidates(list_of_dfs) -> pd.DataFrame

SageMaker handoff:
  The pure candidate-generation functions can be imported unchanged into a
  SageMaker Processing Job.  Only the CLI wiring (S3 I/O) changes.

Authors: Account 1 (canonical / production pipeline)
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants (must match normalization.py output)
# ---------------------------------------------------------------------------
_ENTITY_ID     = "entity_id"
_NORM_NAME     = "normalized_name"
_TOKEN_SORTED  = "token_sorted_name"
_NORM_ADDR     = "normalized_address"
_ADDR_MISSING  = "address_missing"

# Output columns for the candidate table
_LEFT_ID   = "entity_id_left"
_RIGHT_ID  = "entity_id_right"
_SCORE     = "blocking_score"
_METHOD    = "blocking_method"


# ---------------------------------------------------------------------------
# Config dataclass for blocking hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class BlockingConfig:
    """
    Hyper-parameters for the blocking stage.
    Populated from config.yaml → blocking section.
    """
    # TF-IDF pass
    tfidf_top_k:    int   = 50     # candidates returned per entity from TF-IDF
    ngram_min:      int   = 2      # character n-gram min size
    ngram_max:      int   = 3      # character n-gram max size
    tfidf_min_sim:  float = 0.05   # minimum cosine similarity to keep a pair

    # Token-sort prefix pass
    prefix_len:     int   = 5      # prefix shard length in characters

    # Address-prefix pass
    addr_prefix_len: int  = 8      # address prefix shard length

    # Global cap
    max_candidates_per_entity: int = 100

    @classmethod
    def from_config_dict(cls, cfg_dict: Dict[str, Any]) -> "BlockingConfig":
        """Build from a parsed config.yaml blocking sub-dict."""
        blocking = cfg_dict.get("blocking", {})
        return cls(
            tfidf_top_k                = blocking.get("tfidf_top_k", 50),
            ngram_min                  = blocking.get("ngram_min", 2),
            ngram_max                  = blocking.get("ngram_size", 3),
            tfidf_min_sim              = blocking.get("tfidf_min_sim", 0.05),
            prefix_len                 = blocking.get("prefix_len", 5),
            addr_prefix_len            = blocking.get("addr_prefix_len", 8),
            max_candidates_per_entity  = blocking.get("max_candidates_per_entity", 100),
        )


# ---------------------------------------------------------------------------
# Pure blocking functions (independently unit-testable — no S3)
# ---------------------------------------------------------------------------

def generate_tfidf_candidates(
    df_left:  pd.DataFrame,
    df_right: pd.DataFrame,
    cfg:      BlockingConfig,
    *,
    left_source:  str = "left",
    right_source: str = "right",
) -> pd.DataFrame:
    """
    TF-IDF character n-gram cosine-similarity blocking.

    For each entity in df_left, returns the top-K most similar entities
    from df_right by cosine similarity on normalized_name n-grams.

    Parameters
    ----------
    df_left / df_right : DataFrames with columns [entity_id, normalized_name].
                         Rows where normalized_name is null are skipped.
    cfg                : BlockingConfig
    left_source        : Label for the left source (for logging)
    right_source       : Label for the right source (for logging)

    Returns
    -------
    pd.DataFrame with columns [entity_id_left, entity_id_right,
                                blocking_score, blocking_method]
    """
    # Drop null names
    left  = df_left.dropna(subset=[_NORM_NAME]).reset_index(drop=True)
    right = df_right.dropna(subset=[_NORM_NAME]).reset_index(drop=True)

    if left.empty or right.empty:
        logger.warning("TF-IDF blocking: empty input after null-name drop.")
        return _empty_candidates()

    logger.info(
        "TF-IDF blocking: %d left x %d right  (top_k=%d, ngrams=(%d,%d))",
        len(left), len(right), cfg.tfidf_top_k, cfg.ngram_min, cfg.ngram_max,
    )

    # Fit vectorizer on the union of left + right names for a shared vocabulary
    all_names = pd.concat([left[_NORM_NAME], right[_NORM_NAME]], ignore_index=True)
    vectorizer = TfidfVectorizer(
        analyzer     = "char_wb",
        ngram_range  = (cfg.ngram_min, cfg.ngram_max),
        min_df       = 1,
        sublinear_tf = True,
    )
    vectorizer.fit(all_names)

    mat_left  = vectorizer.transform(left[_NORM_NAME])
    mat_right = vectorizer.transform(right[_NORM_NAME])

    # Compute cosine similarity in batches to control peak memory
    rows: List[Dict] = []
    batch_size = 1000
    n_left = mat_left.shape[0]

    for start in range(0, n_left, batch_size):
        end    = min(start + batch_size, n_left)
        batch  = mat_left[start:end]
        sims   = linear_kernel(batch, mat_right)   # shape: (batch, n_right)

        for local_i, sim_row in enumerate(sims):
            global_i = start + local_i
            # Get top-K indices (unsorted, then sort)
            top_k = min(cfg.tfidf_top_k, len(sim_row))
            top_idx = np.argpartition(sim_row, -top_k)[-top_k:]
            top_idx = top_idx[sim_row[top_idx] >= cfg.tfidf_min_sim]
            top_idx = top_idx[np.argsort(sim_row[top_idx])[::-1]]

            lid = left[_ENTITY_ID].iloc[global_i]
            for j in top_idx:
                rid = right[_ENTITY_ID].iloc[j]
                if lid == rid:
                    continue   # skip self-matches in same-source blocking
                rows.append({
                    _LEFT_ID:  lid,
                    _RIGHT_ID: rid,
                    _SCORE:    float(sim_row[j]),
                    _METHOD:   "tfidf",
                })

    df_out = pd.DataFrame(rows, columns=[_LEFT_ID, _RIGHT_ID, _SCORE, _METHOD])
    logger.info("  -> %d TF-IDF candidate pairs", len(df_out))
    return df_out


def generate_token_sort_candidates(
    df_left:  pd.DataFrame,
    df_right: pd.DataFrame,
    cfg:      BlockingConfig,
) -> pd.DataFrame:
    """
    Token-sort prefix blocking.

    Groups entities by the first `cfg.prefix_len` characters of their
    token_sorted_name.  All pairs within the same shard are emitted as
    candidates with score=1.0 (binary — in shard or not).

    Handles the case where df_left and df_right have overlapping entity_ids
    (same-source blocking) by deduplicating pairs (left_id < right_id).

    Parameters
    ----------
    df_left / df_right : DataFrames with [entity_id, token_sorted_name].

    Returns
    -------
    pd.DataFrame [entity_id_left, entity_id_right, blocking_score, blocking_method]
    """
    left  = df_left.dropna(subset=[_TOKEN_SORTED]).copy()
    right = df_right.dropna(subset=[_TOKEN_SORTED]).copy()

    if left.empty or right.empty:
        return _empty_candidates()

    left["_shard"]  = left[_TOKEN_SORTED].str[:cfg.prefix_len]
    right["_shard"] = right[_TOKEN_SORTED].str[:cfg.prefix_len]

    # Join on shard
    merged = pd.merge(
        left[[_ENTITY_ID, "_shard"]].rename(columns={_ENTITY_ID: _LEFT_ID}),
        right[[_ENTITY_ID, "_shard"]].rename(columns={_ENTITY_ID: _RIGHT_ID}),
        on="_shard",
    )
    merged = merged[merged[_LEFT_ID] != merged[_RIGHT_ID]]   # no self-pairs
    merged[_SCORE]  = 1.0
    merged[_METHOD] = "token_sort"
    df_out = merged[[_LEFT_ID, _RIGHT_ID, _SCORE, _METHOD]].reset_index(drop=True)

    logger.info("  -> %d token-sort candidate pairs", len(df_out))
    return df_out


def generate_address_prefix_candidates(
    df_left:  pd.DataFrame,
    df_right: pd.DataFrame,
    cfg:      BlockingConfig,
) -> pd.DataFrame:
    """
    Address numeric-prefix blocking.

    Extracts the leading numeric token (street number) from normalized_address
    and groups entities that share the same street number prefix.
    Only applied to rows where address_missing=False.

    Returns
    -------
    pd.DataFrame [entity_id_left, entity_id_right, blocking_score, blocking_method]
    """
    def _addr_key(row: pd.Series) -> Optional[str]:
        if row.get(_ADDR_MISSING, True):
            return None
        addr = row.get(_NORM_ADDR)
        if pd.isna(addr) or not str(addr).strip():
            return None
        # Take first cfg.addr_prefix_len chars of the normalized address
        return str(addr)[:cfg.addr_prefix_len]

    left  = df_left.copy()
    right = df_right.copy()

    left["_akey"]  = left.apply(_addr_key, axis=1)
    right["_akey"] = right.apply(_addr_key, axis=1)

    left  = left.dropna(subset=["_akey"])
    right = right.dropna(subset=["_akey"])

    if left.empty or right.empty:
        return _empty_candidates()

    merged = pd.merge(
        left[[_ENTITY_ID, "_akey"]].rename(columns={_ENTITY_ID: _LEFT_ID}),
        right[[_ENTITY_ID, "_akey"]].rename(columns={_ENTITY_ID: _RIGHT_ID}),
        on="_akey",
    )
    merged = merged[merged[_LEFT_ID] != merged[_RIGHT_ID]]
    merged[_SCORE]  = 1.0
    merged[_METHOD] = "address_prefix"
    df_out = merged[[_LEFT_ID, _RIGHT_ID, _SCORE, _METHOD]].reset_index(drop=True)

    logger.info("  -> %d address-prefix candidate pairs", len(df_out))
    return df_out


def merge_candidates(
    dfs: List[pd.DataFrame],
    max_per_entity: int = 100,
) -> pd.DataFrame:
    """
    Merge multiple candidate DataFrames, deduplicate pairs, and cap per entity.

    Deduplication: (left_id, right_id) and (right_id, left_id) are treated as
    the same pair — we canonicalize by sorting so left_id <= right_id (string).

    Scoring: when a pair appears via multiple methods, keep the MAX score.
    Methods: concatenated as a '+'-joined string (e.g. 'tfidf+token_sort').

    Parameters
    ----------
    dfs            : list of candidate DataFrames from the blocking functions
    max_per_entity : cap on candidates per entity_id (applied after dedup)

    Returns
    -------
    pd.DataFrame [entity_id_left, entity_id_right, blocking_score, blocking_method]
    """
    if not dfs:
        return _empty_candidates()

    non_empty = [d for d in dfs if not d.empty]
    if not non_empty:
        return _empty_candidates()

    combined = pd.concat(non_empty, ignore_index=True)
    if combined.empty:
        return _empty_candidates()

    # Canonicalize pair direction so left_id <= right_id
    swap = combined[_LEFT_ID] > combined[_RIGHT_ID]
    combined.loc[swap, [_LEFT_ID, _RIGHT_ID]] = (
        combined.loc[swap, [_RIGHT_ID, _LEFT_ID]].values
    )

    # Aggregate per canonical pair: max score, union of methods
    agg = (
        combined
        .groupby([_LEFT_ID, _RIGHT_ID], sort=False)
        .agg(
            blocking_score  = (_SCORE,  "max"),
            blocking_method = (_METHOD, lambda m: "+".join(sorted(set(m)))),
        )
        .reset_index()
    )

    # Cap per entity — keep highest-scoring candidates
    agg = (
        agg
        .sort_values(_SCORE, ascending=False)
        .groupby(_LEFT_ID, sort=False)
        .head(max_per_entity)
        .reset_index(drop=True)
    )

    logger.info("Merged candidates: %d unique pairs", len(agg))
    return agg


# ---------------------------------------------------------------------------
# Orchestrator — run blocking for one source pair
# ---------------------------------------------------------------------------

def run_blocking_pair(
    df_left:  pd.DataFrame,
    df_right: pd.DataFrame,
    cfg:      BlockingConfig,
    left_tag:  str = "left",
    right_tag: str = "right",
    same_source: bool = False,
) -> pd.DataFrame:
    """
    Run all three blocking passes for a (left, right) source pair and return
    the merged, deduplicated, capped candidate table.

    Parameters
    ----------
    df_left / df_right : Normalized DataFrames (output of normalize_batch).
    cfg                : BlockingConfig
    left_tag / right_tag : Source labels for logging (e.g. "S1", "S2").
    same_source        : If True, both DFs come from the same source file.
                         Self-pairs (same entity_id) are skipped automatically.
    """
    logger.info("Blocking: %s x %s", left_tag, right_tag)

    candidates: List[pd.DataFrame] = []

    # Pass 1 — TF-IDF on normalized_name
    candidates.append(generate_tfidf_candidates(
        df_left, df_right, cfg,
        left_source=left_tag, right_source=right_tag,
    ))

    # Pass 2 — token-sort prefix
    candidates.append(generate_token_sort_candidates(df_left, df_right, cfg))

    # Pass 3 — address prefix (only for rows with addresses)
    candidates.append(generate_address_prefix_candidates(df_left, df_right, cfg))

    merged = merge_candidates(candidates, max_per_entity=cfg.max_candidates_per_entity)
    logger.info(
        "Final candidate pairs %s x %s: %d", left_tag, right_tag, len(merged)
    )
    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=[_LEFT_ID, _RIGHT_ID, _SCORE, _METHOD])


def _get_s3_client(aws_profile: str, region: str):
    session = boto3.Session(profile_name=aws_profile, region_name=region)
    return session.client("s3")


def _read_parquet_from_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    logger.info("Reading parquet s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df  = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    logger.info("  -> %d rows x %d cols", len(df), len(df.columns))
    return df


def _write_parquet_to_s3(df: pd.DataFrame, s3, bucket: str, key: str) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf   = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info("Candidates written -> s3://%s/%s  (%d pairs)", bucket, key, len(df))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Generate blocking candidates for a normalized source file against itself
    or against another normalized file, and upload the candidate table to S3.

    Usage examples:
        # S1 vs S1 (same-source dedup / within-source)
        python src/entity_resolution/blocking.py --left s1 --right s1

        # S1 vs S2 (cross-source matching)
        python src/entity_resolution/blocking.py --left s1 --right s2
    """
    import argparse

    from src.entity_resolution.config import (
        add_account_prefix_arg,
        resolve_config_from_args,
    )

    parser = argparse.ArgumentParser(description="Generate blocking candidates from normalized data.")
    add_account_prefix_arg(parser)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--aws-profile", default=None, dest="aws_profile")
    parser.add_argument(
        "--left",  default="s1",
        choices=["s1", "s2", "s3"],
        help="Left source (default: s1)",
    )
    parser.add_argument(
        "--right", default="s2",
        choices=["s1", "s2", "s3"],
        help="Right source (default: s2)",
    )
    parser.add_argument(
        "--input-key", default=None,
        help="Override S3 key for normalized parquet (default: <PROCESSED>normalized_sample.parquet)",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix on output filename, e.g. '_s1_s2'",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cfg    = resolve_config_from_args(args, config_path=args.config)
    paths  = cfg.paths
    bucket = paths.bucket
    bcfg   = BlockingConfig.from_config_dict(cfg._raw)

    s3 = _get_s3_client(cfg.aws_profile, cfg.aws_region)

    # Load normalized sample (uses the single normalized_sample.parquet for demo)
    input_key = args.input_key or (paths.key(paths.PROCESSED) + "normalized_sample.parquet")
    df_norm   = _read_parquet_from_s3(s3, bucket, input_key)

    # For a real multi-source run each source would be a separate parquet;
    # here we demonstrate using the single sample as both left and right.
    left_tag  = args.left.upper()
    right_tag = args.right.upper()
    same      = args.left == args.right

    candidates = run_blocking_pair(
        df_norm, df_norm, bcfg,
        left_tag=left_tag, right_tag=right_tag,
        same_source=same,
    )

    pair_tag  = f"{left_tag}_{right_tag}{args.output_suffix}"
    out_key   = paths.key(paths.CANDIDATES) + f"candidates_{pair_tag}.parquet"
    _write_parquet_to_s3(candidates, s3, bucket, out_key)

    print(f"\nBlocking complete — {len(candidates):,} candidate pairs")
    print(f"Output: s3://{bucket}/{out_key}")
    print(candidates["blocking_method"].value_counts().to_string())
    print()


if __name__ == "__main__":
    main()
