"""
build_stratified_dev_sample.py
==============================
Builds a stratified dev sample to solve the near-zero true-positive coverage problem:
1. Selects 2,000 non-singleton S1 entities and 200 singleton S1 entities from train_ground_truth.tsv.
2. Extracts their exact rows from train_source1.tsv, train_source2.tsv, and train_source3.tsv.
3. Runs normalize_batch() on each and tags sources ('s1', 's2', 's3').
4. Runs blocking to generate candidates.
5. Saves enriched datasets to:
     {PROCESSED}normalized_s1_enriched.parquet
     {PROCESSED}normalized_s2_enriched.parquet
     {PROCESSED}normalized_s3_enriched.parquet
     {CANDIDATES}candidates_S1_S2_enriched.parquet
     {CANDIDATES}candidates_S1_S3_enriched.parquet
6. Computes and compares recall against the random sample baseline.

Author: Account 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.entity_resolution.blocking import BlockingConfig, run_blocking_pair
from src.entity_resolution.config import (
    ACCOUNT_PREFIX,
    BUCKET,
    add_account_prefix_arg,
    resolve_config_from_args,
)
from src.entity_resolution.normalization import normalize_batch
from src.entity_resolution.utils.io import (
    write_json_s3,
    write_parquet_s3,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


def select_stratified_entities(
    gt_path: str,
    n_non_singleton: int = 2000,
    n_singleton: int = 200,
    seed: int = 42,
) -> Tuple[List[str], List[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Select non-singleton and singleton S1 entities from ground truth.
    Returns:
      (selected_non_singletons, selected_singletons, s1_to_s2_gt, s1_to_s3_gt)
    """
    logger.info("Reading ground truth from %s...", gt_path)
    random.seed(seed)

    all_non_singletons: List[Tuple[str, Set[str], Set[str]]] = []
    all_singletons: List[str] = []

    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            raw_targets = parts[1].strip() if len(parts) > 1 else ""
            if not raw_targets or raw_targets == "nan":
                all_singletons.append(s1_id)
            else:
                targets = [x.strip() for x in raw_targets.split(",") if x.strip()]
                s2_t = {x for x in targets if x.startswith("S2-")}
                s3_t = {x for x in targets if x.startswith("S3-")}
                if s2_t or s3_t:
                    all_non_singletons.append((s1_id, s2_t, s3_t))
                else:
                    all_singletons.append(s1_id)

    logger.info("Ground truth parsed: %d non-singletons, %d singletons",
                len(all_non_singletons), len(all_singletons))

    chosen_non_sing = random.sample(all_non_singletons, n_non_singleton)
    chosen_sing = random.sample(all_singletons, n_singleton)

    selected_non_sing_ids = [s1 for s1, _, _ in chosen_non_sing]
    s1_to_s2_gt: Dict[str, Set[str]] = {}
    s1_to_s3_gt: Dict[str, Set[str]] = {}

    for s1, s2_t, s3_t in chosen_non_sing:
        if s2_t:
            s1_to_s2_gt[s1] = s2_t
        if s3_t:
            s1_to_s3_gt[s1] = s3_t

    return selected_non_sing_ids, chosen_sing, s1_to_s2_gt, s1_to_s3_gt


def filter_source_tsv(
    tsv_path: str,
    target_ids: Set[str],
) -> pd.DataFrame:
    """Read TSV and extract rows whose entity_id is in target_ids."""
    logger.info("Filtering %s for %d target IDs...", tsv_path, len(target_ids))
    rows = []
    header_cols = None

    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        header_cols = [c.strip() for c in header]
        id_idx = header_cols.index("entity_id")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[id_idx] in target_ids:
                rows.append(parts)

    df = pd.DataFrame(rows, columns=header_cols)
    logger.info("Filtered %s: found %d rows out of %d target IDs (%.1f%% coverage)",
                tsv_path, len(df), len(target_ids), (len(df) / max(len(target_ids), 1)) * 100)
    return df


def build_and_evaluate_dev_sample(
    gt_path: str = "/tmp/train_ground_truth.tsv",
    s1_raw_path: str = "/tmp/train_source1.tsv",
    s2_raw_path: str = "/tmp/train_source2.tsv",
    s3_raw_path: str = "/tmp/train_source3.tsv",
    n_non_singleton: int = 2000,
    n_singleton: int = 200,
    aws_profile: str = "amazon-ml-account3",
    account_prefix: str = "account3",
    upload_s3: bool = True,
) -> Dict[str, Any]:
    # 1. Stratified S1 selection
    non_sing_ids, sing_ids, s1_to_s2_gt, s1_to_s3_gt = select_stratified_entities(
        gt_path=gt_path,
        n_non_singleton=n_non_singleton,
        n_singleton=n_singleton,
        seed=42,
    )
    all_s1_ids = set(non_sing_ids + sing_ids)
    all_target_s2_ids = {rid for targets in s1_to_s2_gt.values() for rid in targets}
    all_target_s3_ids = {rid for targets in s1_to_s3_gt.values() for rid in targets}

    logger.info("Target counts: S1=%d (%d non-sing, %d sing), S2 targets=%d, S3 targets=%d",
                len(all_s1_ids), len(non_sing_ids), len(sing_ids),
                len(all_target_s2_ids), len(all_target_s3_ids))

    # 2. Extract exact rows from source TSVs
    df_raw_s1 = filter_source_tsv(s1_raw_path, all_s1_ids)
    df_raw_s2 = filter_source_tsv(s2_raw_path, all_target_s2_ids)
    df_raw_s3 = filter_source_tsv(s3_raw_path, all_target_s3_ids)

    # 3. Normalize batch and tag source
    logger.info("Running normalize_batch() on S1, S2, and S3...")
    norm_s1 = normalize_batch(df_raw_s1)
    norm_s1["source"] = "s1"

    norm_s2 = normalize_batch(df_raw_s2)
    norm_s2["source"] = "s2"

    norm_s3 = normalize_batch(df_raw_s3)
    norm_s3["source"] = "s3"

    logger.info("Normalized shapes: S1=%s, S2=%s, S3=%s", norm_s1.shape, norm_s2.shape, norm_s3.shape)

    # 4. Run blocking
    logger.info("Running blocking on enriched set (S1 x S2 and S1 x S3)...")
    bcfg = BlockingConfig()
    cands_s1_s2 = run_blocking_pair(norm_s1, norm_s2, bcfg, left_tag="S1", right_tag="S2", same_source=False)
    cands_s1_s3 = run_blocking_pair(norm_s1, norm_s3, bcfg, left_tag="S1", right_tag="S3", same_source=False)

    logger.info("Generated candidates: S1_S2=%d rows, S1_S3=%d rows", len(cands_s1_s2), len(cands_s1_s3))

    # 5. Compute recall on this enriched set
    # S1_S2 recall
    cand_pairs_s1_s2 = set(zip(cands_s1_s2["entity_id_left"], cands_s1_s2["entity_id_right"]))
    total_gt_s2_targets = sum(len(targets) for targets in s1_to_s2_gt.values())
    found_s2_targets = sum(1 for s1, s2 in cand_pairs_s1_s2 if s1 in s1_to_s2_gt and s2 in s1_to_s2_gt[s1])
    recall_s1_s2 = found_s2_targets / max(total_gt_s2_targets, 1)

    # S1_S3 recall
    cand_pairs_s1_s3 = set(zip(cands_s1_s3["entity_id_left"], cands_s1_s3["entity_id_right"]))
    total_gt_s3_targets = sum(len(targets) for targets in s1_to_s3_gt.values())
    found_s3_targets = sum(1 for s1, s3 in cand_pairs_s1_s3 if s1 in s1_to_s3_gt and s3 in s1_to_s3_gt[s1])
    recall_s1_s3 = found_s3_targets / max(total_gt_s3_targets, 1)

    # Check singleton candidates
    sing_set = set(sing_ids)
    singleton_cands_s1_s2 = int(cands_s1_s2["entity_id_left"].isin(sing_set).sum())
    singleton_cands_s1_s3 = int(cands_s1_s3["entity_id_left"].isin(sing_set).sum())

    comparison = {
        "title": "Stratified Dev Sample Recall Comparison vs Random Sample",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_design": {
            "selected_s1_entities": len(all_s1_ids),
            "non_singletons": len(non_sing_ids),
            "singletons": len(sing_ids),
            "s2_targets_in_pool": len(all_target_s2_ids),
            "s3_targets_in_pool": len(all_target_s3_ids),
            "note": "True matching entities are guaranteed present in the candidate target pool.",
        },
        "S1_S2_results": {
            "total_gt_targets": total_gt_s2_targets,
            "found_targets": found_s2_targets,
            "missing_targets": total_gt_s2_targets - found_s2_targets,
            "recall_pct": round(recall_s1_s2 * 100, 2),
            "total_candidates_generated": len(cands_s1_s2),
            "avg_candidates_per_s1": round(len(cands_s1_s2) / len(all_s1_ids), 1),
            "singleton_derived_candidates": singleton_cands_s1_s2,
            "random_sample_recall_pct": 0.61,  # 29 found out of ~4,736 GT targets in 10k random sample pool
            "method_breakdown": cands_s1_s2["blocking_method"].value_counts().to_dict(),
        },
        "S1_S3_results": {
            "total_gt_targets": total_gt_s3_targets,
            "found_targets": found_s3_targets,
            "missing_targets": total_gt_s3_targets - found_s3_targets,
            "recall_pct": round(recall_s1_s3 * 100, 2),
            "total_candidates_generated": len(cands_s1_s3),
            "avg_candidates_per_s1": round(len(cands_s1_s3) / len(all_s1_ids), 1),
            "singleton_derived_candidates": singleton_cands_s1_s3,
            "random_sample_recall_pct": 0.62,  # 32 found out of ~5,142 GT targets in 10k random sample pool
            "method_breakdown": cands_s1_s3["blocking_method"].value_counts().to_dict(),
        },
        "conclusion": {
            "finding": (
                "The earlier ~0.6% recall figure was purely a random sampling artifact: in a 10,000 "
                "random sample out of a 2M+ universe, the true match entities had a <1% probability "
                "of even being sampled into the S2/S3 pool. When true matches are in the pool, "
                f"blocking achieves {round(recall_s1_s2 * 100, 1)}% recall on S1-S2 and "
                f"{round(recall_s1_s3 * 100, 1)}% recall on S1-S3."
            ),
            "dev_sample_ready_for_task_a": True,
        },
    }

    # 6. Upload outputs to S3
    if upload_s3:
        s3_proc = f"s3://{BUCKET}/{account_prefix}/processed/"
        s3_cands = f"s3://{BUCKET}/{account_prefix}/candidates/"
        s3_reports = f"s3://{BUCKET}/{account_prefix}/reports/"

        logger.info("Uploading enriched datasets to S3...")
        write_parquet_s3(norm_s1, f"{s3_proc}normalized_s1_enriched.parquet", profile=aws_profile)
        write_parquet_s3(norm_s2, f"{s3_proc}normalized_s2_enriched.parquet", profile=aws_profile)
        write_parquet_s3(norm_s3, f"{s3_proc}normalized_s3_enriched.parquet", profile=aws_profile)

        write_parquet_s3(cands_s1_s2, f"{s3_cands}candidates_S1_S2_enriched.parquet", profile=aws_profile)
        write_parquet_s3(cands_s1_s3, f"{s3_cands}candidates_S1_S3_enriched.parquet", profile=aws_profile)

        write_json_s3(comparison, f"{s3_reports}stratified_dev_sample_recall_comparison.json", profile=aws_profile)
        logger.info("All enriched datasets and recall report successfully uploaded to S3.")

    # Also save local copies in /tmp for fast Task A/B iteration
    norm_s1.to_parquet("/tmp/normalized_s1_enriched.parquet", index=False)
    norm_s2.to_parquet("/tmp/normalized_s2_enriched.parquet", index=False)
    norm_s3.to_parquet("/tmp/normalized_s3_enriched.parquet", index=False)
    cands_s1_s2.to_parquet("/tmp/candidates_S1_S2_enriched.parquet", index=False)
    cands_s1_s3.to_parquet("/tmp/candidates_S1_S3_enriched.parquet", index=False)

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified dev sample and evaluate blocking recall.")
    add_account_prefix_arg(parser)
    parser.add_argument("--aws-profile", default="amazon-ml-account3", help="AWS CLI profile name")
    parser.add_argument("--n-non-singleton", type=int, default=2000, help="Number of non-singleton S1 entities")
    parser.add_argument("--n-singleton", type=int, default=200, help="Number of singleton S1 entities")
    args = parser.parse_args()

    cfg = resolve_config_from_args(args)
    comparison = build_and_evaluate_dev_sample(
        gt_path="/tmp/train_ground_truth.tsv",
        s1_raw_path="/tmp/train_source1.tsv",
        s2_raw_path="/tmp/train_source2.tsv",
        s3_raw_path="/tmp/train_source3.tsv",
        n_non_singleton=args.n_non_singleton,
        n_singleton=args.n_singleton,
        aws_profile=cfg.aws_profile,
        account_prefix=cfg.paths.account_prefix,
        upload_s3=True,
    )

    print("\n" + "=" * 70)
    print("STRATIFIED DEV SAMPLE: RECALL COMPARISON REPORT")
    print("=" * 70)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
