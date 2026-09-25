"""
blocking_s1_s2_s3.py  —  Task A
================================
Correct blocking script for the Amazon-ML-Challenge entity-resolution task.

Ground-truth structure (confirmed from train_ground_truth.tsv):
  - All matches are CROSS-SOURCE: source1_entity_id (S1-*) maps to S2-* and S3-* only
  - NO S1-to-S1 matches exist in the ground truth
  - A single S1 entity can match MULTIPLE S2 and/or S3 records (one-to-many)
  - Some S1 entities have EMPTY matched_entity_ids (singletons — no true match)

Therefore:
  - Blocking pair types: S1×S2 and S1×S3 ONLY
  - S1×S1, S2×S2, S3×S3, S2×S3 are NOT generated here

Strategy: same three-pass approach as blocking.py, applied per pair type:
  Pass 1 — TF-IDF character n-gram cosine on normalized_name
  Pass 2 — Token-sort prefix blocking on token_sorted_name
  Pass 3 — Address-prefix blocking (only for rows with addresses present)

Outputs to:
  {CANDIDATES}candidates_S1_S2_sample.parquet
  {CANDIDATES}candidates_S1_S3_sample.parquet

Deletes (or clearly marks stale):
  {CANDIDATES}candidates_S1_S1_sample.parquet  ← wrong, must not be used downstream

Authors: Account 1 (canonical pipeline)
"""
from __future__ import annotations

import io
import logging

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.entity_resolution.blocking import BlockingConfig, run_blocking_pair
from src.entity_resolution.config import add_account_prefix_arg, resolve_config_from_args

logger = logging.getLogger(__name__)


# ── S3 helpers ───────────────────────────────────────────────────────────────

def _s3(profile, region):
    return boto3.Session(profile_name=profile, region_name=region).client("s3")

def _read_parquet(s3c, bucket, key):
    logger.info("Reading s3://%s/%s", bucket, key)
    df = pd.read_parquet(io.BytesIO(s3c.get_object(Bucket=bucket, Key=key)["Body"].read()))
    logger.info("  -> %d rows x %d cols", len(df), len(df.columns))
    return df

def _write_parquet(df, s3c, bucket, key):
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(tbl, buf, compression="snappy")
    buf.seek(0)
    s3c.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info("Written -> s3://%s/%s  (%d rows)", bucket, key, len(df))

def _delete_object(s3c, bucket, key):
    try:
        s3c.delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted stale file -> s3://%s/%s", bucket, key)
    except Exception as e:
        logger.warning("Could not delete s3://%s/%s: %s", bucket, key, e)

def _object_exists(s3c, bucket, key):
    try:
        s3c.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate S1×S2 and S1×S3 blocking candidates.")
    add_account_prefix_arg(p)
    p.add_argument("--config",       default="configs/config.yaml")
    p.add_argument("--aws-profile",  default=None, dest="aws_profile")
    p.add_argument("--sample-size",  type=int, default=10_000,
                   help="Rows already in the normalized sample files (for logging only)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    cfg    = resolve_config_from_args(args, config_path=args.config)
    paths  = cfg.paths
    bucket = paths.bucket
    bcfg   = BlockingConfig.from_config_dict(cfg._raw)
    s3c    = _s3(cfg.aws_profile, cfg.aws_region)
    proc   = paths.key(paths.PROCESSED)
    cands  = paths.key(paths.CANDIDATES)

    # ── Step 1: Delete the incorrect S1×S1 candidates ────────────────────────
    stale_key = cands + "candidates_S1_S1_sample.parquet"
    if _object_exists(s3c, bucket, stale_key):
        logger.warning("Deleting INCORRECT S1×S1 candidates (no S1-to-S1 matches exist in GT)")
        _delete_object(s3c, bucket, stale_key)
    else:
        logger.info("Stale S1×S1 file not found (already clean)")

    # ── Step 2: Load normalized sources ──────────────────────────────────────
    df_s1 = _read_parquet(s3c, bucket, proc + "normalized_s1_sample.parquet")
    df_s2 = _read_parquet(s3c, bucket, proc + "normalized_s2_sample.parquet")
    df_s3 = _read_parquet(s3c, bucket, proc + "normalized_s3_sample.parquet")

    # ── Step 3: S1 × S2 ──────────────────────────────────────────────────────
    logger.info("="*60)
    logger.info("Generating S1×S2 candidates")
    cands_s1_s2 = run_blocking_pair(
        df_s1, df_s2, bcfg,
        left_tag="S1", right_tag="S2", same_source=False,
    )
    out_key_s1_s2 = cands + "candidates_S1_S2_sample.parquet"
    _write_parquet(cands_s1_s2, s3c, bucket, out_key_s1_s2)
    logger.info("S1×S2 method breakdown:\n%s",
                cands_s1_s2["blocking_method"].value_counts().to_string())

    # ── Step 4: S1 × S3 ──────────────────────────────────────────────────────
    logger.info("="*60)
    logger.info("Generating S1×S3 candidates")
    cands_s1_s3 = run_blocking_pair(
        df_s1, df_s3, bcfg,
        left_tag="S1", right_tag="S3", same_source=False,
    )
    out_key_s1_s3 = cands + "candidates_S1_S3_sample.parquet"
    _write_parquet(cands_s1_s3, s3c, bucket, out_key_s1_s3)
    logger.info("S1×S3 method breakdown:\n%s",
                cands_s1_s3["blocking_method"].value_counts().to_string())

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  BLOCKING COMPLETE (Task A)")
    print(f"{'='*65}")
    print(f"  S1×S2 candidates: {len(cands_s1_s2):>8,}  -> {out_key_s1_s2.split('/')[-1]}")
    print(f"  S1×S3 candidates: {len(cands_s1_s3):>8,}  -> {out_key_s1_s3.split('/')[-1]}")
    print(f"  Stale S1×S1 file: DELETED")
    print(f"  Bucket: s3://{bucket}/{cands}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
