"""
recall_s1_s2_s3.py  —  Task B
===============================
Evaluate blocking recall for S1×S2 and S1×S3 candidate sets against the
actual ground truth structure of train_ground_truth.tsv.

GT file structure (verified):
  source1_entity_id \\t matched_entity_ids
  - source1_entity_id is always S1-*
  - matched_entity_ids is a comma-separated list of S2-* and S3-* ids (mixed)
  - Some S1 entities have empty matched_entity_ids (singletons)

Recall computation:
  - Parsed separately into S2 targets and S3 targets per S1 entity
  - Recall = # GT targets found in candidates / # GT targets that exist
  - Singleton S1 entities (empty matched_entity_ids) are tracked SEPARATELY
    and do NOT affect the recall metric — they contribute to false-positive risk

Output: {REPORTS}blocking_recall_report_v2.json
Authors: Account 1
"""
from __future__ import annotations

import io
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Set

import boto3
import pandas as pd

from src.entity_resolution.config import add_account_prefix_arg, resolve_config_from_args

logger = logging.getLogger(__name__)


# ── GT parser ────────────────────────────────────────────────────────────────

@dataclass
class GTRecord:
    s1_id:       str
    s2_targets:  Set[str]   # S2-* ids that truly match this S1
    s3_targets:  Set[str]   # S3-* ids that truly match this S1
    is_singleton: bool       # True if matched_entity_ids was empty


def parse_ground_truth(s3c, bucket: str, gt_key: str, nrows: int = 0) -> List[GTRecord]:
    """
    Parse train_ground_truth.tsv into structured GTRecord objects.

    Parameters
    ----------
    nrows : int
        0 = full file; >0 = sample (for testing).
    """
    body = s3c.get_object(Bucket=bucket, Key=gt_key)["Body"]
    if nrows > 0:
        lines = [body.readline()]
        for i, ln in enumerate(body.iter_lines()):
            if i >= nrows: break
            lines.append(ln)
        raw = b"\n".join(lines)
    else:
        raw = body.read()

    df = pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False, keep_default_na=False)
    # Columns: source1_entity_id, matched_entity_ids
    df.columns = df.columns.str.strip()

    records = []
    for _, row in df.iterrows():
        s1_id   = str(row["source1_entity_id"]).strip()
        raw_ids = str(row.get("matched_entity_ids", "")).strip()
        if not raw_ids:
            records.append(GTRecord(s1_id=s1_id, s2_targets=set(),
                                    s3_targets=set(), is_singleton=True))
            continue
        all_ids   = [x.strip() for x in raw_ids.split(",") if x.strip()]
        s2_targets = {x for x in all_ids if x.startswith("S2-")}
        s3_targets = {x for x in all_ids if x.startswith("S3-")}
        records.append(GTRecord(s1_id=s1_id, s2_targets=s2_targets,
                                s3_targets=s3_targets, is_singleton=False))

    n_sing = sum(1 for r in records if r.is_singleton)
    logger.info("GT: %d S1 entities  (%d singletons, %d with matches)",
                len(records), n_sing, len(records)-n_sing)
    return records


# ── Recall computation ───────────────────────────────────────────────────────

def _build_cand_set(df_cands: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build {s1_entity_id: {s2_or_s3_candidate_ids}} from a candidate DataFrame.
    Direction-aware: entity_id_left is always S1, entity_id_right is S2 or S3.
    """
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for lid, rid in zip(df_cands["entity_id_left"], df_cands["entity_id_right"]):
        mapping[lid].add(rid)
    return dict(mapping)


@dataclass
class RecallStats:
    pair_type:          str    # 'S1_S2' or 'S1_S3'
    s1_entities_in_gt:  int    # S1 entities with >= 1 target of this type
    total_gt_targets:   int    # total individual targets to be found
    found_targets:      int    # targets that appear in candidate set
    missing_targets:    int    # targets NOT in candidate set
    recall:             float
    total_candidates:   int
    singleton_s1_count: int    # S1 entities with no targets of this type
    singleton_cand_vol: int    # total candidates generated for singletons


def compute_recall_from_gt(
    gt_records:  List[GTRecord],
    df_cands:    pd.DataFrame,
    pair_type:   str,           # 'S1_S2' or 'S1_S3'
) -> RecallStats:
    """
    Compute recall for one pair type (S1×S2 or S1×S3).
    Singletons (no targets of this type) are excluded from recall but tracked.
    """
    cand_map   = _build_cand_set(df_cands)
    total_cand = sum(len(v) for v in cand_map.values())

    target_attr = "s2_targets" if pair_type == "S1_S2" else "s3_targets"

    n_entities    = 0
    total_targets = 0
    found         = 0
    missing       = 0
    sing_count    = 0
    sing_vol      = 0

    for rec in gt_records:
        targets = getattr(rec, target_attr)
        if not targets:
            # Singleton for this pair type
            sing_count += 1
            sing_vol   += len(cand_map.get(rec.s1_id, set()))
            continue
        n_entities    += 1
        total_targets += len(targets)
        cands_for_s1   = cand_map.get(rec.s1_id, set())
        n_found        = len(targets & cands_for_s1)
        found   += n_found
        missing += len(targets) - n_found

    recall = found / total_targets if total_targets > 0 else 0.0
    return RecallStats(
        pair_type         = pair_type,
        s1_entities_in_gt = n_entities,
        total_gt_targets  = total_targets,
        found_targets     = found,
        missing_targets   = missing,
        recall            = recall,
        total_candidates  = total_cand,
        singleton_s1_count = sing_count,
        singleton_cand_vol = sing_vol,
    )


def compute_method_breakdown(
    gt_records:  List[GTRecord],
    df_cands:    pd.DataFrame,
    pair_type:   str,
) -> List[Dict]:
    """Per-blocking-method recall contribution for one pair type."""
    target_attr = "s2_targets" if pair_type == "S1_S2" else "s3_targets"
    all_targets  = {}
    for rec in gt_records:
        tgts = getattr(rec, target_attr)
        if tgts:
            all_targets[rec.s1_id] = tgts

    total_targets = sum(len(v) for v in all_targets.values())
    methods = df_cands["blocking_method"].str.split("+").explode().unique()

    rows = []
    for method in methods:
        sub      = df_cands[df_cands["blocking_method"].str.contains(method, regex=False)]
        sub_map  = _build_cand_set(sub)
        found    = sum(
            len(all_targets.get(lid, set()) & rids)
            for lid, rids in sub_map.items()
        )
        rows.append({
            "method":          method,
            "candidates":      len(sub),
            "gt_found":        found,
            "pct_of_total_gt": round(found / total_targets * 100, 2) if total_targets else 0,
        })
    return sorted(rows, key=lambda x: x["gt_found"], reverse=True)


# ── S3 helpers ───────────────────────────────────────────────────────────────

def _s3(profile, region):
    return boto3.Session(profile_name=profile, region_name=region).client("s3")

def _parquet(s3c, bucket, key):
    logger.info("Loading s3://%s/%s", bucket, key)
    df = pd.read_parquet(io.BytesIO(s3c.get_object(Bucket=bucket, Key=key)["Body"].read()))
    logger.info("  -> %d rows", len(df))
    return df


# ── Print ────────────────────────────────────────────────────────────────────

def _print_stats(stats: RecallStats, method_df: List[Dict]):
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  BLOCKING RECALL  —  {stats.pair_type}")
    print(sep)
    print(f"  S1 entities with {stats.pair_type} targets: {stats.s1_entities_in_gt:>6,}")
    print(f"  Total GT targets (individual):             {stats.total_gt_targets:>6,}")
    print(f"  Found in candidates:                       {stats.found_targets:>6,}")
    print(f"  Missing (recall gap):                      {stats.missing_targets:>6,}")
    print(f"  RECALL:                                    {stats.recall:>9.4f}  ({stats.recall*100:.2f}%)")
    print(f"  Total candidate pairs:                     {stats.total_candidates:>6,}")
    print(f"  Singleton S1 (no {stats.pair_type} target):      {stats.singleton_s1_count:>6,}")
    print(f"  Candidates generated for singletons:       {stats.singleton_cand_vol:>6,}")
    print(f"\n  Method breakdown:")
    for row in method_df:
        print(f"    {row['method']:30s}  cands={row['candidates']:>7,}  "
              f"gt_found={row['gt_found']:>5,}  ({row['pct_of_total_gt']:.1f}% of GT)")
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate S1×S2 and S1×S3 blocking recall.")
    add_account_prefix_arg(p)
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--aws-profile", default=None, dest="aws_profile")
    p.add_argument("--gt-sample",   type=int, default=0,
                   help="GT rows to load (0=full, >0=sample for testing)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    cfg    = resolve_config_from_args(args, config_path=args.config)
    paths  = cfg.paths
    bucket = paths.bucket
    s3c    = _s3(cfg.aws_profile, cfg.aws_region)
    cands  = paths.key(paths.CANDIDATES)
    gt_key = paths.raw_source_keys["ground_truth"]

    # Load GT
    logger.info("Parsing ground truth…")
    gt_records = parse_ground_truth(s3c, bucket, gt_key, nrows=args.gt_sample)

    # Load candidate tables
    df_s1_s2 = _parquet(s3c, bucket, cands + "candidates_S1_S2_sample.parquet")
    df_s1_s3 = _parquet(s3c, bucket, cands + "candidates_S1_S3_sample.parquet")

    # Compute recall
    stats_s1_s2 = compute_recall_from_gt(gt_records, df_s1_s2, "S1_S2")
    stats_s1_s3 = compute_recall_from_gt(gt_records, df_s1_s3, "S1_S3")

    methods_s1_s2 = compute_method_breakdown(gt_records, df_s1_s2, "S1_S2")
    methods_s1_s3 = compute_method_breakdown(gt_records, df_s1_s3, "S1_S3")

    _print_stats(stats_s1_s2, methods_s1_s2)
    _print_stats(stats_s1_s3, methods_s1_s3)

    # Build and upload report
    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "gt_total_records": len(gt_records),
        "gt_singletons": sum(1 for r in gt_records if r.is_singleton),
        "S1_S2": {**asdict(stats_s1_s2), "method_breakdown": methods_s1_s2},
        "S1_S3": {**asdict(stats_s1_s3), "method_breakdown": methods_s1_s3},
    }
    rk   = paths.key(paths.REPORTS) + "blocking_recall_report_v2.json"
    body = json.dumps(report, indent=2, default=str).encode()
    s3c.put_object(Bucket=bucket, Key=rk, Body=body, ContentType="application/json")
    logger.info("Report -> s3://%s/%s", bucket, rk)


if __name__ == "__main__":
    main()
