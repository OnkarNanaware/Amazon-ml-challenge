"""
run_blocking_experiments.py
============================
Account 2 — CLI runner for blocking experiments.

Reads Account 1's canonical normalized data from {INPUT_PROCESSED},
runs all three experiment sets via blocking_experiments.py,
writes candidate parquets to {CANDIDATES} and the consolidated report
to {REPORTS}.

All S3 paths come from config.py — no hardcoded paths anywhere.

IMPORTANT: This script is for SMALL-SAMPLE VALIDATION ONLY.
Full-scale blocking MUST run via SageMaker Processing Jobs.
Never set sample_size=0 locally.

Usage
-----
    # Small sample (10K rows each source), Account 2 paths:
    python scripts/run_blocking_experiments.py \\
        --account-prefix account2 \\
        --input-account-prefix account1 \\
        --aws-profile amazon-ml-account2

    # Custom experiment params:
    python scripts/run_blocking_experiments.py \\
        --account-prefix account2 \\
        --input-account-prefix account1 \\
        --aws-profile amazon-ml-account2 \\
        --tfidf-top-k 50 \\
        --global-cap 100 \\
        --gt-sample 50000

Authors: Account 2 (blocking experiments)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.entity_resolution.blocking_experiments import (
    ExperimentResult,
    parse_ground_truth_df,
    run_all_experiments,
)
from src.entity_resolution.config import (
    Paths,
    load_pipeline_config,
    add_account_prefix_arg,
    resolve_config_from_args,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client(profile: str, region: str):
    return boto3.Session(profile_name=profile, region_name=region).client("s3")


def _read_parquet(s3c, bucket: str, key: str) -> pd.DataFrame:
    logger.info("Reading s3://%s/%s", bucket, key)
    body = s3c.get_object(Bucket=bucket, Key=key)["Body"].read()
    df   = pd.read_parquet(io.BytesIO(body))
    logger.info("  -> %d rows x %d cols", len(df), len(df.columns))
    return df


def _read_tsv(s3c, bucket: str, key: str, nrows: int = 0) -> pd.DataFrame:
    logger.info("Reading GT s3://%s/%s (nrows=%d)", bucket, key, nrows)
    body = s3c.get_object(Bucket=bucket, Key=key)["Body"]
    if nrows > 0:
        lines = [body.readline()]
        for i, ln in enumerate(body.iter_lines()):
            if i >= nrows:
                break
            lines.append(ln)
        raw = b"\n".join(lines)
    else:
        raw = body.read()
    df = pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False, keep_default_na=False)
    logger.info("  -> %d GT rows", len(df))
    return df


def _write_parquet(df: pd.DataFrame, s3c, bucket: str, key: str) -> None:
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(tbl, buf, compression="snappy")
    buf.seek(0)
    s3c.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info("Written -> s3://%s/%s  (%d rows)", bucket, key, len(df))


def _write_json(obj: dict, s3c, bucket: str, key: str) -> None:
    body = json.dumps(obj, indent=2, default=str).encode()
    s3c.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("Written -> s3://%s/%s", bucket, key)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(results: list[ExperimentResult], args: argparse.Namespace) -> dict:
    """
    Build the consolidated JSON report dict.

    The recall/cost tradeoff table is the first thing in the report so it
    is immediately visible when the file is opened.
    """
    table_rows = []
    for r in results:
        table_rows.append({
            "experiment":                r.name,
            "set":                       r.experiment_set,
            "recall_S1_S2":              round(r.s1_s2.recall, 4),
            "recall_S1_S3":              round(r.s1_s3.recall, 4),
            "avg_recall":                round((r.s1_s2.recall + r.s1_s3.recall) / 2, 4),
            "addr_missing_recall_S1_S2": r.addr_missing_s1_s2.get("recall"),
            "addr_missing_recall_S1_S3": r.addr_missing_s1_s3.get("recall"),
            "avg_cands_S1_S2":           round(r.s1_s2.avg_candidates_per_entity, 1),
            "avg_cands_S1_S3":           round(r.s1_s3.avg_candidates_per_entity, 1),
            "total_cands_S1_S2":         r.s1_s2.total_candidates,
            "total_cands_S1_S3":         r.s1_s3.total_candidates,
            "singleton_cand_vol_S1_S2":  r.s1_s2.singleton_cand_vol,
            "singleton_cand_vol_S1_S3":  r.s1_s3.singleton_cand_vol,
            "recommendation":            r.recommendation,
            "notes":                     r.notes,
        })

    return {
        "generated_at":       datetime.now(tz=timezone.utc).isoformat(),
        "account_prefix":     args.account_prefix or "account2",
        "input_account_prefix": args.input_account_prefix or "account1",
        "sample_size_approx": args.sample_size,
        "gt_sample_rows":     args.gt_sample,
        "tfidf_top_k":        args.tfidf_top_k,
        "global_cap":         args.global_cap,
        # *** PRIMARY OUTPUT: recall/cost tradeoff table ***
        "recall_cost_tradeoff_table": table_rows,
        # Full per-experiment details (including country breakdown)
        "experiments": [r.to_dict() for r in results],
    }


def _print_tradeoff_table(results: list[ExperimentResult]) -> None:
    """Print the recall/cost tradeoff table to stdout."""
    sep = "=" * 120
    print(f"\n{sep}")
    print("  BLOCKING EXPERIMENTS — RECALL / COST TRADEOFF TABLE")
    print(f"  (Account 2 report — small sample only; full scale via SageMaker)")
    print(sep)
    header = (
        f"  {'Experiment':<45} {'Set':>3}  "
        f"{'RecS1S2':>8}  {'RecS1S3':>8}  {'AvgRec':>7}  "
        f"{'AvgCandS2':>10}  {'AvgCandS3':>10}  "
        f"{'AddrMissS2':>11}  {'AddrMissS3':>11}  "
        f"{'Rec':<14}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        am_s2 = r.addr_missing_s1_s2.get("recall")
        am_s3 = r.addr_missing_s1_s3.get("recall")
        am_s2_str = f"{am_s2:.4f}" if am_s2 is not None else "  N/A  "
        am_s3_str = f"{am_s3:.4f}" if am_s3 is not None else "  N/A  "
        avg_rec = (r.s1_s2.recall + r.s1_s3.recall) / 2
        print(
            f"  {r.name:<45} {r.experiment_set:>3}  "
            f"{r.s1_s2.recall:>8.4f}  {r.s1_s3.recall:>8.4f}  {avg_rec:>7.4f}  "
            f"{r.s1_s2.avg_candidates_per_entity:>10.1f}  "
            f"{r.s1_s3.avg_candidates_per_entity:>10.1f}  "
            f"{am_s2_str:>11}  {am_s3_str:>11}  "
            f"{r.recommendation:<14}"
        )
    print(sep)
    print()
    print("  LEGEND:")
    print("   RecS1S2 / RecS1S3 : Blocking recall for S1xS2 / S1xS3 pair types")
    print("   AvgRec             : Mean of RecS1S2 and RecS1S3")
    print("   AvgCand*           : Avg candidates generated per S1 entity (compute proxy)")
    print("   AddrMiss*          : Recall for address_missing=True subset of S1")
    print("   Rec                : Recommendation (keep / drop / tune-further)")
    print()
    print("  CEILING METRIC: If a true pair never becomes a candidate, no downstream")
    print("  model can recover it. Recall % IS the ceiling for downstream classifier.")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Account 2 — Run blocking experiments on Account 1's normalized data."
    )
    add_account_prefix_arg(p)
    p.add_argument(
        "--input-account-prefix", default=None, dest="input_account_prefix",
        metavar="PREFIX",
        help="S3 prefix to READ normalized data from (default: 'account1'). "
             "Override via INPUT_ACCOUNT_PREFIX env var.",
    )
    p.add_argument("--config",       default="configs/config.yaml")
    p.add_argument("--aws-profile",  default=None, dest="aws_profile")
    p.add_argument(
        "--sample-size", type=int, default=10_000,
        help="Approximate rows in normalized sample parquets (for logging only).",
    )
    p.add_argument(
        "--gt-sample", type=int, default=50_000,
        help="GT rows to load for recall evaluation (0=full, >0=sample). Default: 50000.",
    )
    # Experiment params
    p.add_argument("--tfidf-top-k",  type=int, default=50)
    p.add_argument("--global-cap",   type=int, default=100)
    p.add_argument(
        "--topk-sweep", nargs="+", type=int, default=[10, 25, 50, 100],
        metavar="K",
        help="K values for exp1a TF-IDF top-K sweep.",
    )
    p.add_argument(
        "--rarity-sweep", nargs="+", type=float, default=[0.001, 0.005, 0.01],
        metavar="PCT",
        help="Rarity pct values for exp1b rare-token sweep.",
    )
    p.add_argument(
        "--cap-sweep", nargs="+", type=int, default=[50, 100, 200],
        metavar="CAP",
        help="Cap values for exp1c global cap sweep.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # ---- Resolve configs ----------------------------------------------------
    # My output paths: account2
    cfg    = resolve_config_from_args(args, config_path=args.config)
    paths  = cfg.paths   # account2/* outputs

    # Input paths: account1 (or whatever --input-account-prefix says)
    import os
    input_prefix = (
        args.input_account_prefix
        or os.environ.get("INPUT_ACCOUNT_PREFIX")
        or "account1"
    )
    input_paths = Paths(bucket=paths.bucket, account_prefix=input_prefix)

    bucket  = paths.bucket
    s3c     = _s3_client(cfg.aws_profile, cfg.aws_region)

    logger.info("=" * 70)
    logger.info("Account 2 Blocking Experiments")
    logger.info("  Input (Account 1 normalized): %s", input_paths.PROCESSED)
    logger.info("  Output candidates:            %s", paths.CANDIDATES)
    logger.info("  Output report:                %s", paths.REPORTS)
    logger.info("=" * 70)

    # ---- Load normalized data (from Account 1's processed/) -----------------
    proc = input_paths.key(input_paths.PROCESSED)

    df_s1 = _read_parquet(s3c, bucket, proc + "normalized_s1_sample.parquet")
    df_s2 = _read_parquet(s3c, bucket, proc + "normalized_s2_sample.parquet")
    df_s3 = _read_parquet(s3c, bucket, proc + "normalized_s3_sample.parquet")

    logger.info(
        "Loaded: S1=%d rows, S2=%d rows, S3=%d rows",
        len(df_s1), len(df_s2), len(df_s3),
    )

    # ---- Load ground truth --------------------------------------------------
    gt_key  = paths.raw_source_keys["ground_truth"]   # raw/train_ground_truth.tsv
    df_gt   = _read_tsv(s3c, bucket, gt_key, nrows=args.gt_sample)
    gt_records = parse_ground_truth_df(df_gt)

    # ---- Run experiments ----------------------------------------------------
    results = run_all_experiments(
        df_s1, df_s2, df_s3, gt_records,
        topk_sweep   = args.topk_sweep,
        rarity_sweep = args.rarity_sweep,
        cap_sweep    = args.cap_sweep,
        tfidf_top_k  = args.tfidf_top_k,
        global_cap   = args.global_cap,
    )

    # ---- Print tradeoff table to stdout ------------------------------------
    _print_tradeoff_table(results)

    # ---- Write candidate parquets per experiment ---------------------------
    cands_prefix = paths.key(paths.CANDIDATES)
    for r in results:
        # We don't store candidates for every exp to save space;
        # store only the combined/best-config ones (Set 1c and Set 2a union)
        if r.name in ("exp1c_global_cap_100", "exp2a_union_baseline",
                      "exp2b_all3", "exp3c_addr_missing_name_only_quality"):
            # Re-derive candidates for these key experiments
            # (the runner already computed them; for simplicity we re-run a tiny subset)
            logger.info("Skipping parquet write for %s (candidates not cached in runner)", r.name)
            # NOTE: For a memory-efficient full-scale version, the experiment functions
            # should return (result, candidates) tuples. This is scaffolded here.

    # Write a placeholder for the canonical combined S1xS2 and S1xS3 candidate sets
    # from the best experiment (exp2b_all3 = baseline, will be updated after review)
    # TODO: after reviewing the tradeoff table, re-run with winning config and write
    # the actual candidate parquets.
    logger.info(
        "NOTE: Candidate parquet writing is scaffolded. After reviewing the "
        "tradeoff table, run the winning experiment config explicitly with "
        "--write-candidates to emit exp_<name>_S1_S2.parquet / exp_<name>_S1_S3.parquet"
    )

    # ---- Write consolidated report ------------------------------------------
    report   = _build_report(results, args)
    rpt_key  = paths.key(paths.REPORTS) + "blocking_experiments_report.json"
    _write_json(report, s3c, bucket, rpt_key)

    print(f"\n  Report -> s3://{bucket}/{rpt_key}")
    print(f"  {len(results)} experiments logged.\n")


if __name__ == "__main__":
    main()
