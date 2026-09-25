"""
candidate_recall.py
===================
Step 4 of the Amazon-ML-Challenge entity-resolution pipeline.

Measures the recall of the blocking stage against the ground-truth
matched pairs.  This is the **primary quality gate** for blocking:
a pair missed here can never be recovered by the classifier.

Key metric: Recall@K
  = (GT matched pairs that appear in candidate table) / (total GT matched pairs)

Also reports:
  - Precision@K (candidate pairs that are in GT)
  - Pairs-per-entity (efficiency of blocking)
  - Method breakdown (which blocking pass contributed each found GT pair)
  - Missing pairs: GT pairs NOT found in candidates (for error analysis)

Public API (pure functions, no S3 deps):
  compute_recall(candidates, ground_truth) -> RecallResult
  coverage_by_method(candidates, ground_truth) -> pd.DataFrame

Authors: Account 1 (canonical / production pipeline)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
_LEFT_ID   = "entity_id_left"
_RIGHT_ID  = "entity_id_right"
_SCORE     = "blocking_score"
_METHOD    = "blocking_method"
_GT_LEFT   = "entity_id_1"    # ground-truth column names
_GT_RIGHT  = "entity_id_2"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RecallResult:
    """
    Outcome of a single recall evaluation.

    Attributes
    ----------
    total_gt_pairs     : Total GT matched pairs evaluated
    found_pairs        : GT pairs that appear in the candidate table
    missing_pairs      : GT pairs NOT in the candidate table
    total_candidates   : Total candidate pairs generated
    recall             : found / total_gt_pairs
    precision          : found / total_candidates (upper bound on precision)
    avg_candidates_per_entity : mean number of candidate pairs per left entity
    missing_df         : DataFrame of GT pairs not recovered (for error analysis)
    """
    total_gt_pairs:    int
    found_pairs:       int
    missing_pairs:     int
    total_candidates:  int
    recall:            float
    precision:         float
    avg_candidates_per_entity: float
    missing_df:        pd.DataFrame


# ---------------------------------------------------------------------------
# Core helper: canonicalize pair direction
# ---------------------------------------------------------------------------

def _canonicalize(df: pd.DataFrame, left_col: str, right_col: str) -> "set[frozenset]":
    """
    Convert a DataFrame of pairs into a set of frozensets for O(1) lookup.
    Direction-agnostic: (A, B) and (B, A) are the same pair.
    """
    pairs: set[frozenset] = set()
    for lid, rid in zip(df[left_col], df[right_col]):
        pairs.add(frozenset([lid, rid]))
    return pairs


# ---------------------------------------------------------------------------
# Public pure functions
# ---------------------------------------------------------------------------

def compute_recall(
    candidates:   pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    gt_left_col:  str = _GT_LEFT,
    gt_right_col: str = _GT_RIGHT,
) -> RecallResult:
    """
    Compute blocking recall against ground-truth matched pairs.

    Parameters
    ----------
    candidates   : Output of blocking.merge_candidates()
                   Expected columns: entity_id_left, entity_id_right
    ground_truth : GT DataFrame with matched pairs only.
                   Expected columns: entity_id_1, entity_id_2
    gt_left_col  : Column name for left entity_id in ground_truth
    gt_right_col : Column name for right entity_id in ground_truth

    Returns
    -------
    RecallResult
    """
    cand_pairs = _canonicalize(candidates, _LEFT_ID, _RIGHT_ID)
    gt_pairs_list = list(zip(ground_truth[gt_left_col], ground_truth[gt_right_col]))
    gt_pairs_set  = {frozenset(p) for p in gt_pairs_list}

    total_gt   = len(gt_pairs_set)
    total_cand = len(cand_pairs)

    found_pairs   = cand_pairs & gt_pairs_set
    missing_pairs = gt_pairs_set - cand_pairs

    n_found   = len(found_pairs)
    n_missing = len(missing_pairs)

    recall    = n_found / total_gt    if total_gt   > 0 else 0.0
    precision = n_found / total_cand  if total_cand > 0 else 0.0

    avg_per_entity = (
        candidates.groupby(_LEFT_ID).size().mean()
        if not candidates.empty else 0.0
    )

    # Build missing-pairs DataFrame for error analysis
    missing_rows = [
        {gt_left_col: min(p), gt_right_col: max(p)}
        for p in missing_pairs
    ]
    missing_df = pd.DataFrame(missing_rows, columns=[gt_left_col, gt_right_col])

    return RecallResult(
        total_gt_pairs             = total_gt,
        found_pairs                = n_found,
        missing_pairs              = n_missing,
        total_candidates           = total_cand,
        recall                     = recall,
        precision                  = precision,
        avg_candidates_per_entity  = float(avg_per_entity),
        missing_df                 = missing_df,
    )


def coverage_by_method(
    candidates:   pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    gt_left_col:  str = _GT_LEFT,
    gt_right_col: str = _GT_RIGHT,
) -> pd.DataFrame:
    """
    Break down which blocking methods recovered which GT pairs.

    For each unique blocking_method value in the candidate table, reports
    how many GT pairs were found exclusively by that method vs jointly.

    Returns
    -------
    pd.DataFrame with columns:
        method, gt_pairs_found, pct_of_total_gt, pct_of_found_gt
    """
    gt_pairs_set = _canonicalize(ground_truth, gt_left_col, gt_right_col)
    total_gt     = len(gt_pairs_set)
    total_found  = len(_canonicalize(candidates, _LEFT_ID, _RIGHT_ID) & gt_pairs_set)

    rows = []
    for method in candidates[_METHOD].str.split("+").explode().unique():
        mask = candidates[_METHOD].str.contains(method, regex=False)
        sub  = candidates[mask]
        sub_pairs = _canonicalize(sub, _LEFT_ID, _RIGHT_ID)
        found = len(sub_pairs & gt_pairs_set)
        rows.append({
            "method":           method,
            "candidates":       len(sub),
            "gt_pairs_found":   found,
            "pct_of_total_gt":  round(found / total_gt    * 100, 2) if total_gt    > 0 else 0.0,
            "pct_of_found_gt":  round(found / total_found * 100, 2) if total_found > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("gt_pairs_found", ascending=False).reset_index(drop=True)


def print_recall_report(result: RecallResult, label: str = "") -> None:
    """Print a human-readable recall report to stdout."""
    sep  = "=" * 60
    head = f"  BLOCKING RECALL REPORT{f'  [{label}]' if label else ''}"
    print(f"\n{sep}")
    print(head)
    print(sep)
    print(f"  GT matched pairs evaluated : {result.total_gt_pairs:>10,}")
    print(f"  Found in candidates        : {result.found_pairs:>10,}")
    print(f"  Missing (recall gap)       : {result.missing_pairs:>10,}")
    print(f"  Total candidate pairs      : {result.total_candidates:>10,}")
    print(f"  Recall                     : {result.recall:>10.4f}  ({result.recall*100:.2f}%)")
    print(f"  Upper-bound precision      : {result.precision:>10.4f}  ({result.precision*100:.2f}%)")
    print(f"  Avg candidates / entity    : {result.avg_candidates_per_entity:>10.1f}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# S3 I/O helpers (CLI only)
# ---------------------------------------------------------------------------

def _get_s3_client(aws_profile: str, region: str):
    session = boto3.Session(profile_name=aws_profile, region_name=region)
    return session.client("s3")


def _read_parquet_from_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _read_tsv_from_s3(s3, bucket: str, key: str, nrows: Optional[int] = None) -> pd.DataFrame:
    obj  = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]
    if nrows is not None:
        lines = [body.readline()]
        for i, line in enumerate(body.iter_lines()):
            if i >= nrows: break
            lines.append(line)
        raw = b"\n".join(lines)
    else:
        raw = body.read()
    return pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Evaluate blocking recall for a candidate file against ground truth.

    Reads:
      - {CANDIDATES}candidates_<tag>.parquet
      - raw/train_ground_truth.tsv (first --gt-sample rows)

    Prints recall report to stdout.
    """
    import argparse

    from src.entity_resolution.config import (
        add_account_prefix_arg,
        resolve_config_from_args,
    )

    parser = argparse.ArgumentParser(description="Evaluate blocking recall against ground truth.")
    add_account_prefix_arg(parser)
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--aws-profile", default=None, dest="aws_profile")
    parser.add_argument(
        "--candidates-key", default=None,
        help="S3 key for candidates parquet (default: <CANDIDATES>candidates_S1_S2.parquet)",
    )
    parser.add_argument(
        "--gt-sample", type=int, default=100_000,
        help="Number of GT rows to evaluate against (default: 100000)",
    )
    parser.add_argument(
        "--label", default="",
        help="Label for the recall report header",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cfg    = resolve_config_from_args(args, config_path=args.config)
    paths  = cfg.paths
    bucket = paths.bucket

    s3 = _get_s3_client(cfg.aws_profile, cfg.aws_region)

    cand_key = args.candidates_key or (paths.key(paths.CANDIDATES) + "candidates_S1_S2.parquet")
    candidates   = _read_parquet_from_s3(s3, bucket, cand_key)
    ground_truth = _read_tsv_from_s3(
        s3, bucket,
        paths.raw_source_keys["ground_truth"],
        nrows=args.gt_sample,
    )

    result = compute_recall(candidates, ground_truth)
    print_recall_report(result, label=args.label or cand_key.split("/")[-1])

    method_df = coverage_by_method(candidates, ground_truth)
    if not method_df.empty:
        print("  Method coverage breakdown:")
        print(method_df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
