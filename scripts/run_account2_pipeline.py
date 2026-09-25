#!/usr/bin/env python3
"""
scripts/run_account2_pipeline.py
================================
Account 2 Pipeline Runner:
- Streams raw samples via aws CLI subprocess (no boto3)
- Normalizes batches via normalize_batch()
- Uploads normalized parquet files to s3://amzn-s3-ml-c/account2/processed/
- Runs all 21 blocking experiments from blocking_experiments.py
- Uploads per-experiment JSON results to s3://amzn-s3-ml-c/account2/experiments/
- Uploads consolidated report to s3://amzn-s3-ml-c/account2/reports/
- Verifies every S3 upload with `aws s3 ls` for non-zero file size

Strict constraint: boto3 is NOT imported or called anywhere in this script.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Import domain modules (pure logic, no boto3 dependencies)
from src.entity_resolution.normalization import normalize_batch
from src.entity_resolution.blocking_experiments import (
    ExperimentResult,
    GTRecord,
    parse_ground_truth_df,
    run_all_experiments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("account2_pipeline")

BUCKET = "amzn-s3-ml-c"
ACCOUNT_PREFIX = "account2"


# ---------------------------------------------------------------------------
# S3 Subprocess Helpers (No boto3)
# ---------------------------------------------------------------------------

def run_aws_cli(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run an AWS CLI command via subprocess."""
    cmd = ["aws"] + args
    logger.debug("Executing: %s", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"AWS CLI command failed (code {res.returncode}): {' '.join(cmd)}\nStderr: {res.stderr}")
    return res


def verify_s3_file(s3_uri: str, profile: str) -> int:
    """
    Verify that an S3 object exists and has non-zero size via `aws s3 ls`.
    Returns the file size in bytes.
    Raises RuntimeError if file is missing or 0 bytes.
    """
    res = run_aws_cli(["s3", "ls", s3_uri, "--profile", profile], check=False)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"Verification failed: object does not exist at {s3_uri}\nStderr: {res.stderr}")
    
    # Typical aws s3 ls output: "2026-09-25 12:00:00    123456 filename"
    line = res.stdout.strip().splitlines()[-1]
    parts = line.split()
    if len(parts) >= 3:
        try:
            size = int(parts[2])
            if size <= 0:
                raise RuntimeError(f"Verification failed: object at {s3_uri} has size 0 bytes.")
            logger.info(" Verified %s: %d bytes", s3_uri, size)
            return size
        except ValueError:
            pass
    logger.info(" Verified %s exists: %s", s3_uri, line)
    return 1


def stream_s3_tsv_sample(s3_uri: str, n_rows: int, profile: str) -> pd.DataFrame:
    """
    Stream exactly n_rows (+ 1 header row) from S3 TSV via `aws s3 cp <s3_uri> -`.
    Uses process termination to prevent downloading the rest of large files.
    """
    cmd = ["aws", "s3", "cp", s3_uri, "-", "--profile", profile]
    logger.info("Streaming %d sample rows from %s...", n_rows, s3_uri)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=65536
    )

    lines: List[str] = []
    try:
        for i, line in enumerate(proc.stdout):
            lines.append(line)
            if i >= n_rows:  # header (0) + n_rows data lines
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

    if not lines:
        raise RuntimeError(f"No data received when streaming from {s3_uri}")

    raw_tsv = "".join(lines)
    df = pd.read_csv(io.StringIO(raw_tsv), sep="\t", low_memory=False)
    logger.info("Loaded %d rows x %d cols from %s", len(df), len(df.columns), s3_uri)
    return df


def upload_file_to_s3(local_path: str, s3_uri: str, profile: str) -> int:
    """Upload a local file to S3 and verify non-zero size."""
    logger.info("Uploading %s -> %s...", local_path, s3_uri)
    run_aws_cli(["s3", "cp", local_path, s3_uri, "--profile", profile])
    size = verify_s3_file(s3_uri, profile)
    return size


def download_file_from_s3(s3_uri: str, local_path: str, profile: str) -> None:
    """Download a file from S3 to local path."""
    logger.info("Downloading %s -> %s...", s3_uri, local_path)
    run_aws_cli(["s3", "cp", s3_uri, local_path, "--profile", profile])


# ---------------------------------------------------------------------------
# Task 2: Smoke Test
# ---------------------------------------------------------------------------

def run_smoke_test(profile: str, sample_size: int = 10000) -> bool:
    """
    Smoke test on train_source1.tsv:
    1. Stream sample
    2. Normalize batch tagging source='s1'
    3. Save to local temp parquet & upload
    4. Download back from S3 & load with pandas
    5. Compare schema, row count, columns, verify non-corruption
    """
    logger.info("=" * 60)
    logger.info("STARTING TASK 2: SMOKE TEST (train_source1.tsv)")
    logger.info("=" * 60)

    s3_raw_s1 = f"s3://{BUCKET}/raw/train_source1.tsv"
    s3_proc_s1 = f"s3://{BUCKET}/{ACCOUNT_PREFIX}/processed/normalized_s1_sample.parquet"

    # Step 1: Stream
    df_raw = stream_s3_tsv_sample(s3_raw_s1, sample_size, profile)

    # Step 2: Normalize
    logger.info("Normalizing batch (source='s1')...")
    df_norm = normalize_batch(df_raw, source="s1")
    logger.info("Normalized DataFrame: %d rows x %d cols", len(df_norm), len(df_norm.columns))

    with tempfile.TemporaryDirectory() as tmpdir:
        local_out = os.path.join(tmpdir, "normalized_s1_sample.parquet")
        local_in = os.path.join(tmpdir, "downloaded_s1_sample.parquet")

        # Step 3: Write parquet and upload
        df_norm.to_parquet(local_out, compression="snappy", index=False)
        upload_file_to_s3(local_out, s3_proc_s1, profile)

        # Step 4: Download back
        download_file_from_s3(s3_proc_s1, local_in, profile)

        # Step 5: Verify round-trip
        df_roundtrip = pd.read_parquet(local_in)
        logger.info("Downloaded roundtrip parquet: %d rows x %d cols", len(df_roundtrip), len(df_roundtrip.columns))

        # Checks
        assert len(df_norm) == len(df_roundtrip), f"Row count mismatch: {len(df_norm)} vs {len(df_roundtrip)}"
        assert list(df_norm.columns) == list(df_roundtrip.columns), "Column schema mismatch"
        assert "source" in df_roundtrip.columns, "Missing 'source' column"
        assert (df_roundtrip["source"] == "s1").all(), "Incorrect source tags"
        assert "address_missing" in df_roundtrip.columns, "Missing address_missing column"

        logger.info("✅ SMOKE TEST PASSED: Round-trip verification successful!")
        logger.info("  Rows: %d", len(df_roundtrip))
        logger.info("  Columns: %s", list(df_roundtrip.columns))
        return True


# ---------------------------------------------------------------------------
# Task 3: Full Pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    profile: str,
    source_sample_size: int = 10000,
    gt_sample_size: int = 50000,
    tfidf_top_k: int = 50,
    global_cap: int = 100,
) -> Tuple[List[ExperimentResult], pd.DataFrame]:
    """
    Full Account 2 pipeline:
    1. Stream & normalize s1, s2, s3 (10,000 rows each)
    2. Upload normalized_s{1,2,3}_sample.parquet & verify
    3. Stream & parse train_ground_truth.tsv sample (50,000 rows)
    4. Run all 21 blocking experiments
    5. Upload each experiment JSON (21 files) & verify
    6. Consolidate and upload report & verify
    """
    logger.info("=" * 60)
    logger.info("STARTING TASK 3: FULL ACCOUNT 2 PIPELINE")
    logger.info("=" * 60)

    # 1. Stream, normalize, upload s1, s2, s3
    sources = [("s1", "train_source1.tsv"), ("s2", "train_source2.tsv"), ("s3", "train_source3.tsv")]
    normalized_dfs: Dict[str, pd.DataFrame] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for tag, fname in sources:
            s3_raw = f"s3://{BUCKET}/raw/{fname}"
            s3_dest = f"s3://{BUCKET}/{ACCOUNT_PREFIX}/processed/normalized_{tag}_sample.parquet"

            df_raw = stream_s3_tsv_sample(s3_raw, source_sample_size, profile)
            df_norm = normalize_batch(df_raw, source=tag)
            normalized_dfs[tag] = df_norm

            local_parquet = os.path.join(tmpdir, f"normalized_{tag}_sample.parquet")
            df_norm.to_parquet(local_parquet, compression="snappy", index=False)
            upload_file_to_s3(local_parquet, s3_dest, profile)

        # 2. Stream ground truth sample
        s3_gt = f"s3://{BUCKET}/raw/train_ground_truth.tsv"
        df_gt_raw = stream_s3_tsv_sample(s3_gt, gt_sample_size, profile)
        logger.info("Parsing ground truth into GTRecord structures...")
        gt_records: List[GTRecord] = parse_ground_truth_df(df_gt_raw)

        # 3. Run all 21 experiments
        logger.info("Executing 21 blocking experiments...")
        all_results = run_all_experiments(
            df_s1=normalized_dfs["s1"],
            df_s2=normalized_dfs["s2"],
            df_s3=normalized_dfs["s3"],
            gt_records=gt_records,
            tfidf_top_k=tfidf_top_k,
            global_cap=global_cap,
        )
        logger.info("Total experiments executed: %d", len(all_results))

        # 4. Upload each experiment result JSON
        logger.info("Uploading %d individual experiment JSON files...", len(all_results))
        for exp in all_results:
            exp_dict = exp.to_dict()
            exp_name_clean = re.sub(r"[^a-zA-Z0-9_-]", "_", exp.name)
            local_exp_path = os.path.join(tmpdir, f"exp_{exp_name_clean}.json")
            with open(local_exp_path, "w", encoding="utf-8") as f:
                json.dump(exp_dict, f, indent=2, default=str)

            s3_exp_dest = f"s3://{BUCKET}/{ACCOUNT_PREFIX}/experiments/exp_{exp_name_clean}.json"
            upload_file_to_s3(local_exp_path, s3_exp_dest, profile)

        # 5. Build consolidated report table
        logger.info("Generating consolidated recall/cost tradeoff report...")
        report_rows = []
        for exp in all_results:
            row = {
                "name": exp.name,
                "set": exp.experiment_set,
                "recall_s1_s2": round(exp.s1_s2.recall, 4),
                "recall_s1_s3": round(exp.s1_s3.recall, 4),
                "avg_cands_s1_s2": round(exp.s1_s2.avg_candidates_per_entity, 1),
                "avg_cands_s1_s3": round(exp.s1_s3.avg_candidates_per_entity, 1),
                "total_cands_s1_s2": exp.s1_s2.total_candidates,
                "total_cands_s1_s3": exp.s1_s3.total_candidates,
                "singleton_vol_s1_s2": exp.s1_s2.singleton_cand_vol,
                "singleton_vol_s1_s3": exp.s1_s3.singleton_cand_vol,
                "recommendation": exp.recommendation,
                "notes": exp.notes,
            }
            report_rows.append(row)

        df_report = pd.DataFrame(report_rows)

        full_report_data = {
            "title": "Account 2 Blocking Experiments Consolidated Report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_experiments": len(all_results),
            "tradeoff_table": report_rows,
            "experiments": [exp.to_dict() for exp in all_results],
        }

        local_report_path = os.path.join(tmpdir, "blocking_experiments_report.json")
        with open(local_report_path, "w", encoding="utf-8") as f:
            json.dump(full_report_data, f, indent=2, default=str)

        s3_report_dest = f"s3://{BUCKET}/{ACCOUNT_PREFIX}/reports/blocking_experiments_report.json"
        upload_file_to_s3(local_report_path, s3_report_dest, profile)

    return all_results, df_report


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Account 2 Blocking Pipeline Runner")
    parser.add_argument("--aws-profile", default="amazon-ml-account2", help="AWS CLI profile name")
    parser.add_argument("--smoke-test-only", action="store_true", help="Run only Task 2 smoke test")
    parser.add_argument("--source-sample-size", type=int, default=10000, help="Source sample size")
    parser.add_argument("--gt-sample-size", type=int, default=50000, help="Ground truth sample size")
    parser.add_argument("--tfidf-top-k", type=int, default=50, help="Default TF-IDF Top K")
    parser.add_argument("--global-cap", type=int, default=100, help="Default global cap")

    args = parser.parse_args()

    if args.smoke_test_only:
        run_smoke_test(args.aws_profile, sample_size=args.source_sample_size)
    else:
        # Run smoke test first as required by Task 2
        smoke_passed = run_smoke_test(args.aws_profile, sample_size=args.source_sample_size)
        if not smoke_passed:
            logger.error("Smoke test failed. Aborting full pipeline run.")
            sys.exit(1)
        
        # Run full pipeline as Task 3
        all_results, df_report = run_full_pipeline(
            profile=args.aws_profile,
            source_sample_size=args.source_sample_size,
            gt_sample_size=args.gt_sample_size,
            tfidf_top_k=args.tfidf_top_k,
            global_cap=args.global_cap,
        )

        logger.info("\n" + "=" * 80)
        logger.info("FINAL TRADE-OFF REPORT TABLE")
        logger.info("=" * 80)
        print(df_report.to_string(index=False))


if __name__ == "__main__":
    main()
