"""
profiling.py
============
Step 1 of the Amazon-ML-Challenge entity-resolution pipeline.

Profiles the raw TSV files stored in S3 and writes:
  - A structured JSON report  → {REPORTS}profiling_report{suffix}.json
  - A human-readable summary  → stdout

All S3 paths are derived from src.entity_resolution.config — no hardcoded
bucket/prefix strings in this file.

Designed to run on a configurable sample (profiling.sample_size in config.yaml)
so that it stays fast even when the full dataset exceeds 1 M rows.

IMPORTANT: Full-scale execution MUST run on SageMaker Processing Jobs.
           Never run --sample-size 0 (full file) directly in Antigravity/IDE.

Authors: Account 1 (canonical / production pipeline)
"""

from __future__ import annotations

import io
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
import yaml

from src.entity_resolution.config import (
    PipelineConfig,
    Paths,
    add_account_prefix_arg,
    load_pipeline_config,
    resolve_config_from_args,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
# We check both naming conventions so the module works with the real schema
# (business_name / business_address) and generic test fixtures (name / address).
_NAME_COLS     = ["business_name", "name"]
_ADDR_COLS     = ["business_address", "address"]
_COUNTRY_COL   = "country"
_ENTITY_ID_COL = "entity_id"

# Ground-truth columns
_GT_LEFT_COL  = "entity_id_1"
_GT_RIGHT_COL = "entity_id_2"
_GT_MATCH_COL = "label"        # may be absent (GT contains only matched pairs)

# Source-file short tags for cross-source pair distribution
_SOURCE_TAGS: Dict[str, str] = {
    "train_source1": "S1",
    "train_source2": "S2",
    "train_source3": "S3",
}


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from *candidates* that exists in *df*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _get_s3_client(aws_profile: str, region: str) -> Any:
    """Return a boto3 S3 client for the given AWS profile."""
    session = boto3.Session(profile_name=aws_profile, region_name=region)
    return session.client("s3")


def _read_tsv_from_s3(
    s3_client: Any,
    bucket: str,
    key: str,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stream a TSV from S3 into a pandas DataFrame.

    Parameters
    ----------
    s3_client : boto3 S3 client
    bucket    : S3 bucket name
    key       : S3 object key (path inside the bucket)
    nrows     : If set, read only the first *nrows* data rows (after the header).

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Reading s3://%s/%s (nrows=%s)", bucket, key, nrows)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]

    if nrows is not None:
        lines: List[bytes] = []
        header = body.readline()
        lines.append(header)
        for i, line in enumerate(body.iter_lines()):
            if i >= nrows:
                break
            lines.append(line)
        raw = b"\n".join(lines)
        df = pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False)
    else:
        raw = body.read()
        df = pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False)

    logger.info("  -> loaded %d rows x %d cols", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Core profiling functions  (independently unit-testable — no S3 required)
# ---------------------------------------------------------------------------

def profile_basic(df: pd.DataFrame, file_label: str) -> Dict[str, Any]:
    """
    Basic structural stats: row count, column dtypes, missing-value counts.
    """
    missing: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        missing[col] = {
            "null_count": null_count,
            "null_pct": round(null_count / max(len(df), 1) * 100, 4),
        }
    return {
        "file": file_label,
        "row_count": len(df),
        "columns": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": missing,
    }


def profile_entity_id(df: pd.DataFrame, file_label: str) -> Dict[str, Any]:
    """Uniqueness check on entity_id column."""
    if _ENTITY_ID_COL not in df.columns:
        return {"file": file_label, "entity_id_col_present": False}
    total  = len(df)
    unique = int(df[_ENTITY_ID_COL].nunique(dropna=True))
    return {
        "file": file_label,
        "entity_id_col_present": True,
        "total_rows": total,
        "unique_entity_ids": unique,
        "duplicate_entity_ids": total - unique,
    }


def profile_duplicates(df: pd.DataFrame, file_label: str) -> Dict[str, Any]:
    """
    Duplicate-row and duplicate-(name, address, country) combination counts.
    Supports both business_name/business_address and name/address schemas.
    """
    dup_rows = int(df.duplicated().sum())
    name_col = _pick_col(df, _NAME_COLS)
    addr_col = _pick_col(df, _ADDR_COLS)
    combo_cols = [c for c in [name_col, addr_col, _COUNTRY_COL]
                  if c is not None and c in df.columns]
    dup_combos = int(df.duplicated(subset=combo_cols).sum()) if combo_cols else None
    return {
        "file": file_label,
        "duplicate_rows": dup_rows,
        "duplicate_name_address_country": dup_combos,
        "combo_key_columns_used": combo_cols,
    }


def profile_text_fields(df: pd.DataFrame, file_label: str) -> Dict[str, Any]:
    """
    Unique value counts and length distributions for name and address columns.
    Keys in the result are always 'name' / 'address' regardless of actual column name.
    """
    result: Dict[str, Any] = {"file": file_label}
    for candidates, field_key in [(_NAME_COLS, "name"), (_ADDR_COLS, "address")]:
        col = _pick_col(df, candidates)
        if col is None:
            result[field_key] = {"present": False, "column_name": None}
            continue
        series  = df[col].dropna().astype(str)
        lengths = series.str.len()
        n = len(lengths)
        result[field_key] = {
            "present": True,
            "column_name": col,
            "unique_count": int(series.nunique()),
            "length_distribution": {
                "min":    int(lengths.min())                    if n else None,
                "max":    int(lengths.max())                    if n else None,
                "mean":   round(float(lengths.mean()), 2)      if n else None,
                "median": round(float(lengths.median()), 2)    if n else None,
                "p25":    round(float(lengths.quantile(0.25)), 2) if n else None,
                "p75":    round(float(lengths.quantile(0.75)), 2) if n else None,
                "p95":    round(float(lengths.quantile(0.95)), 2) if n else None,
            },
        }
    return result


def profile_country(df: pd.DataFrame, file_label: str) -> Dict[str, Any]:
    """
    Open-set country distribution — fully data-driven, no hard-coded list.
    Returns top-50 countries by frequency + total unique count.
    """
    if _COUNTRY_COL not in df.columns:
        return {"file": file_label, "country_col_present": False}
    vc = df[_COUNTRY_COL].fillna("__MISSING__").value_counts()
    return {
        "file": file_label,
        "country_col_present": True,
        "total_unique_countries": int(vc.shape[0]),
        "country_distribution": {str(k): int(v) for k, v in vc.head(50).items()},
    }


def profile_ground_truth(
    gt_df: pd.DataFrame,
    source_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Profile the ground-truth file.

    Computes matched/non-matched pair counts and cross-source pair distribution
    (S1-S2, S1-S3, S2-S3) for matched pairs.
    """
    entity_to_sources: Dict[Any, List[str]] = defaultdict(list)
    for tag, df in source_dfs.items():
        short_tag = _SOURCE_TAGS.get(tag, tag)
        if _ENTITY_ID_COL in df.columns:
            for eid in df[_ENTITY_ID_COL].dropna().unique():
                entity_to_sources[eid].append(short_tag)

    total_pairs = len(gt_df)
    matched_pairs = non_matched_pairs = None

    if _GT_MATCH_COL in gt_df.columns:
        matched_pairs     = int((gt_df[_GT_MATCH_COL] == 1).sum())
        non_matched_pairs = int((gt_df[_GT_MATCH_COL] == 0).sum())
        gt_matched        = gt_df[gt_df[_GT_MATCH_COL] == 1]
    else:
        gt_matched = gt_df

    pair_counts: Dict[str, int] = defaultdict(int)
    for _, row in gt_matched.iterrows():
        eid1 = row.get(_GT_LEFT_COL)
        eid2 = row.get(_GT_RIGHT_COL)
        tags1 = set(entity_to_sources.get(eid1, []))
        tags2 = set(entity_to_sources.get(eid2, []))
        all_tags = sorted(tags1 | tags2)
        if len(all_tags) >= 2:
            pair_key = f"{all_tags[0]}-{all_tags[1]}"
        elif len(all_tags) == 1:
            pair_key = f"{all_tags[0]}-{all_tags[0]}"
        else:
            pair_key = "unknown"
        pair_counts[pair_key] += 1

    return {
        "total_pairs": total_pairs,
        "matched_pairs": matched_pairs,
        "non_matched_pairs": non_matched_pairs,
        "label_column_present": _GT_MATCH_COL in gt_df.columns,
        "cross_source_pair_distribution": dict(pair_counts),
    }


# ---------------------------------------------------------------------------
# Aggregate orchestrator
# ---------------------------------------------------------------------------

def run_profiling(
    cfg: PipelineConfig,
    sample_size: Optional[int] = None,
    report_suffix: str = "",
) -> Tuple[Dict[str, Any], Any, str, str]:
    """
    Orchestrate the full profiling run.

    Parameters
    ----------
    cfg           : PipelineConfig (from load_pipeline_config / resolve_config_from_args)
    sample_size   : Rows per source file.  None = use config default.  0 = full file.
    report_suffix : Suffix appended to the report filename, e.g. "_100k"

    Returns
    -------
    (report_dict, s3_client, bucket, report_key)
    """
    paths: Paths = cfg.paths

    # Resolve sample size (CLI arg > config > hardcoded fallback)
    if sample_size is None:
        sample_size = cfg.profiling.get("sample_size", 10_000)
    if sample_size == 0:
        sample_size = None  # pandas nrows=None means "all rows"

    # Report key derived entirely from config — no hardcoded paths
    report_s3_key = paths.key(paths.REPORTS) + f"profiling_report{report_suffix}.json"

    s3 = _get_s3_client(cfg.aws_profile, cfg.aws_region)
    bucket = paths.bucket

    # Source file keys from config
    raw_keys = paths.raw_source_keys
    source_keys = {
        "train_source1": raw_keys["train_source1"],
        "train_source2": raw_keys["train_source2"],
        "train_source3": raw_keys["train_source3"],
    }
    gt_key = raw_keys["ground_truth"]

    # Load data
    source_dfs: Dict[str, pd.DataFrame] = {}
    for label, key in source_keys.items():
        source_dfs[label] = _read_tsv_from_s3(s3, bucket, key, nrows=sample_size)
    gt_df = _read_tsv_from_s3(s3, bucket, gt_key, nrows=sample_size)

    # Profile each source
    per_file_reports: List[Dict[str, Any]] = []
    for label, df in source_dfs.items():
        per_file_reports.append({
            "label":      label,
            "basic":      profile_basic(df, label),
            "entity_id":  profile_entity_id(df, label),
            "duplicates": profile_duplicates(df, label),
            "text_fields": profile_text_fields(df, label),
            "country":    profile_country(df, label),
        })

    # Profile ground truth
    gt_report = {
        "label": "train_ground_truth",
        "basic": profile_basic(gt_df, "train_ground_truth"),
        "entity_id_left": profile_entity_id(
            gt_df.rename(columns={_GT_LEFT_COL: _ENTITY_ID_COL}),
            "ground_truth_left",
        ) if _GT_LEFT_COL in gt_df.columns else {},
        "entity_id_right": profile_entity_id(
            gt_df.rename(columns={_GT_RIGHT_COL: _ENTITY_ID_COL}),
            "ground_truth_right",
        ) if _GT_RIGHT_COL in gt_df.columns else {},
        "ground_truth_stats": profile_ground_truth(gt_df, source_dfs),
    }

    report: Dict[str, Any] = {
        "generated_at":          datetime.now(tz=timezone.utc).isoformat(),
        "sample_size_per_file":  sample_size,
        "account_prefix":        paths.account_prefix,
        "bucket":                bucket,
        "report_s3_uri":         paths.uri(report_s3_key),
        "source_files":          per_file_reports,
        "ground_truth":          gt_report,
    }

    return report, s3, bucket, report_s3_key


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

def upload_report(
    report: Dict[str, Any],
    s3_client: Any,
    bucket: str,
    key: str,
) -> None:
    """Serialise the profiling report as JSON and upload it to S3."""
    body = json.dumps(report, indent=2, default=str).encode("utf-8")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("Report uploaded -> s3://%s/%s", bucket, key)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(report: Dict[str, Any]) -> None:
    """Print a human-readable profiling summary to stdout."""
    sep  = "=" * 72
    thin = "-" * 72

    sample = report["sample_size_per_file"]
    sample_str = f"{sample:,}" if sample else "FULL FILE"

    print(f"\n{sep}")
    print("  ENTITY RESOLUTION -- PROFILING REPORT")
    print(f"  Generated      : {report['generated_at']}")
    print(f"  Account prefix : {report['account_prefix']}")
    print(f"  Sample per file: {sample_str}")
    print(f"  Report S3 URI  : {report['report_s3_uri']}")
    print(sep)

    for src in report["source_files"]:
        label = src["label"]
        basic = src["basic"]
        eid   = src["entity_id"]
        dups  = src["duplicates"]
        txt   = src["text_fields"]
        ctry  = src["country"]

        print(f"\n  > {label.upper()}")
        print(f"    Rows : {basic['row_count']:,}   Columns : {list(basic['columns'].keys())}")

        missing_cols = {c: v for c, v in basic["missing_values"].items()
                        if v["null_count"] > 0}
        if missing_cols:
            print("    Missing values:")
            for c, v in missing_cols.items():
                print(f"        {c:25s}  {v['null_count']:>8,}  ({v['null_pct']:.2f}%)")
        else:
            print("    Missing values: none")

        if eid.get("entity_id_col_present"):
            print(f"    entity_id -- unique: {eid['unique_entity_ids']:,}  "
                  f"duplicates: {eid['duplicate_entity_ids']:,}")
        else:
            print("    entity_id column: NOT FOUND")

        print(f"    Duplicate rows: {dups['duplicate_rows']:,}   "
              f"Dup (name,addr,country): {dups['duplicate_name_address_country']}")

        for field_key in ["name", "address"]:
            info       = txt.get(field_key, {})
            actual_col = info.get("column_name") or field_key
            if info.get("present"):
                ld = info["length_distribution"]
                print(f"    {actual_col:22s} -- unique: {info['unique_count']:,}   "
                      f"len [min={ld['min']}, mean={ld['mean']}, "
                      f"max={ld['max']}, p95={ld['p95']}]")
            else:
                print(f"    {field_key:8s} column: NOT FOUND")

        if ctry.get("country_col_present"):
            print(f"    Countries -- {ctry['total_unique_countries']} unique")
            for country, cnt in list(ctry["country_distribution"].items())[:15]:
                print(f"        {country:35s}  {cnt:>8,}")
            if ctry["total_unique_countries"] > 15:
                print(f"        ... ({ctry['total_unique_countries']} total)")
        else:
            print("    country column: NOT FOUND")

    # Ground truth
    gt  = report["ground_truth"]
    gts = gt["ground_truth_stats"]
    print(f"\n{thin}")
    print("  GROUND TRUTH")
    print(thin)
    print(f"    Total pairs           : {gts['total_pairs']:,}")
    if gts["label_column_present"]:
        print(f"    Matched pairs         : {gts['matched_pairs']:,}")
        print(f"    Non-matched pairs     : {gts['non_matched_pairs']:,}")
    else:
        print("    (No label column — GT contains only matched pairs)")
    print("    Cross-source pair distribution (matched):")
    for pair, cnt in sorted(gts["cross_source_pair_distribution"].items()):
        print(f"        {pair:20s}  {cnt:>8,}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run profiling end-to-end:
      1. Load config (account_prefix resolved via CLI arg / env var / yaml / default)
      2. Read sampled data from S3
      3. Compute all metrics
      4. Upload JSON report to S3 (path derived from config — no hardcoded paths)
      5. Print summary to console
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile raw entity-resolution data from S3."
    )
    add_account_prefix_arg(parser)   # --account-prefix (lets Account 2/3 reuse unchanged)
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--aws-profile",
        default=None,
        dest="aws_profile",
        help="AWS CLI profile (default: amazon-ml-account1 or AWS_PROFILE env var)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Rows per source file (0 = full file, SageMaker only). "
             "Overrides config profiling.sample_size.",
    )
    parser.add_argument(
        "--report-suffix",
        default="",
        help="Suffix appended to report filename, e.g. '_100k' -> profiling_report_100k.json",
    )
    args = parser.parse_args()

    cfg = resolve_config_from_args(args, config_path=args.config)

    logger.info(
        "Starting profiling | account_prefix=%s | profile=%s | sample=%s",
        cfg.paths.account_prefix,
        cfg.aws_profile,
        args.sample_size,
    )

    report, s3, bucket, report_key = run_profiling(
        cfg,
        sample_size   = args.sample_size,
        report_suffix = args.report_suffix,
    )

    upload_report(report, s3, bucket, report_key)
    print_summary(report)

    logger.info("Profiling complete.")


if __name__ == "__main__":
    main()
