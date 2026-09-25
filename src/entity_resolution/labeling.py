"""
labeling.py
===========
Task A — Label generation for entity-resolution candidate pairs.

Design contract
---------------
* Loads candidate pairs (entity_id_left, entity_id_right, blocking_score,
  blocking_method) and parsed ground-truth, then assigns label=1/0 per pair.
* Standardises column names:
    entity_id_left  -> source1_entity_id
    entity_id_right -> candidate_entity_id
    candidate_source inferred from ID prefix (S2-* → 's2', S3-* → 's3')
* All singleton S1 entities (absent from GT or with empty matched_entity_ids)
  have every candidate labelled 0 automatically.
* Hard-negative sampling — per source1_entity_id, samples a configurable
  mix of:
    hard negatives : high blocking_score, label=0 (what model must learn to reject)
    easy negatives : low blocking_score, label=0
  Default ratio: 1 positive : 2 hard negatives : 2 easy negatives.
  Configurable via NegativeSamplingConfig.
* Does NOT compute train/val splits — that responsibility belongs to train.py
  (entity-level split by source1_entity_id).
* Pure functions; no hardcoded S3 paths. All I/O wiring is in the CLI.

Outputs (separate per pair type — positive-pool sizes differ significantly)
-------
  {FEATURES}labeled_pairs_S1_S2.parquet
  {FEATURES}labeled_pairs_S1_S3.parquet

Schema of output parquets
--------------------------
  source1_entity_id    : str   — left entity (always S1)
  candidate_entity_id  : str   — right entity (S2 or S3)
  candidate_source     : str   — 's2' or 's3'
  blocking_score       : float
  blocking_method      : str
  label                : int   — 1=match, 0=non-match
  negative_type        : str   — 'positive', 'hard_negative', 'easy_negative', 'singleton_negative'

Authors: Account 3
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entity_resolution.config import (
    add_account_prefix_arg,
    resolve_config_from_args,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NegativeSamplingConfig:
    """
    Controls hard/easy negative sampling ratios.

    For each source1_entity_id that has at least one positive pair,
    we attempt to sample:
      hard_neg_per_pos * n_positives  hard negatives  (confusable with the TP)
      easy_neg_per_pos * n_positives  easy negatives  (low score)

    If the entity does not have enough candidates of the requested type,
    we take all available (no oversampling).

    For singleton entities (no GT match), ALL their candidates are kept as
    'singleton_negative' without any sub-sampling — they provide pure-negative
    coverage and are intentionally not down-sampled.

    hard_neg_delta  : float — a hard negative must have
                        blocking_score >= (entity's max TP score - hard_neg_delta)
                      i.e. it must be within `delta` of the entity's best true-match
                      score.  Default 0.10.

                      This replaces the old "≥70th percentile of entity's negative
                      distribution" approach, which was found to be too loose: both
                      TPs and the sampled hard negatives frequently hit blocking_score
                      1.0 (exact token-sort match), making the percentile cut
                      meaningless.  Entity-relative delta ensures hard negatives are
                      genuinely confusable with the true match.

    easy_score_pct  : float — a candidate is 'easy' if its blocking_score is at or
                      below this percentile of the entity's *full* negative distribution.
                      Default 0.30 (bottom 30%).
    """
    hard_neg_per_pos: int   = 2     # hard negatives to sample per positive
    easy_neg_per_pos: int   = 2     # easy negatives to sample per positive
    hard_neg_delta:   float = 0.10  # hard neg must be within delta of entity's TP-max score
    easy_score_pct:   float = 0.30  # score ≤ this percentile → easy negative
    seed:             int   = 42


# ---------------------------------------------------------------------------
# Ground-truth parsing
# ---------------------------------------------------------------------------

def parse_ground_truth(gt_path: str) -> Tuple[
    Dict[str, Set[str]],   # s1_to_s2: {s1_id -> {s2_id, ...}}
    Dict[str, Set[str]],   # s1_to_s3: {s1_id -> {s3_id, ...}}
    Set[str],              # singleton_s1_ids (no matches in either source)
]:
    """
    Parse train_ground_truth.tsv into GT lookup dicts.

    TSV structure:
        source1_entity_id  <TAB>  matched_entity_ids (comma-separated, may be empty)

    Matched IDs are split by prefix:
        S2-* → s1_to_s2
        S3-* → s1_to_s3

    Singletons: rows where matched_entity_ids is empty / NaN / whitespace.
    There is NO S1-S1, S2-S2, S3-S3, or S2-S3 matching.
    """
    logger.info("Parsing ground truth from %s ...", gt_path)
    s1_to_s2: Dict[str, Set[str]] = {}
    s1_to_s3: Dict[str, Set[str]] = {}
    singleton_ids: Set[str] = set()

    with open(gt_path, "r", encoding="utf-8") as fh:
        _header = fh.readline()  # consume header
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            raw_targets = parts[1].strip() if len(parts) > 1 else ""

            if not raw_targets or raw_targets.lower() in ("nan", ""):
                singleton_ids.add(s1_id)
                continue

            targets = [t.strip() for t in raw_targets.split(",") if t.strip()]
            s2_targets = {t for t in targets if t.startswith("S2-")}
            s3_targets = {t for t in targets if t.startswith("S3-")}

            if s2_targets:
                s1_to_s2[s1_id] = s2_targets
            if s3_targets:
                s1_to_s3[s1_id] = s3_targets
            if not s2_targets and not s3_targets:
                # All targets have unknown prefix — treat as singleton
                singleton_ids.add(s1_id)

    n_with_s2 = len(s1_to_s2)
    n_with_s3 = len(s1_to_s3)
    n_sing = len(singleton_ids)
    logger.info(
        "GT parsed: %d S1→S2 entities, %d S1→S3 entities, %d singletons",
        n_with_s2, n_with_s3, n_sing,
    )
    return s1_to_s2, s1_to_s3, singleton_ids


# ---------------------------------------------------------------------------
# Core labelling logic
# ---------------------------------------------------------------------------

def _infer_candidate_source(entity_id: str) -> str:
    """Infer 's2' or 's3' from entity_id prefix (S2-* or S3-*)."""
    if entity_id.startswith("S2-"):
        return "s2"
    if entity_id.startswith("S3-"):
        return "s3"
    raise ValueError(f"Cannot infer source from entity_id: {entity_id!r}")


def assign_labels(
    candidates: pd.DataFrame,
    s1_to_targets: Dict[str, Set[str]],
    singleton_s1_ids: Set[str],
) -> pd.DataFrame:
    """
    Assign raw labels (1/0) to every candidate pair, WITHOUT any sampling.

    Parameters
    ----------
    candidates       : DataFrame with columns [entity_id_left, entity_id_right,
                        blocking_score, blocking_method]
    s1_to_targets    : {s1_id -> set of true match IDs} for the relevant source
                        (either s1_to_s2 or s1_to_s3, not both combined)
    singleton_s1_ids : set of S1 IDs that have no GT matches at all

    Returns
    -------
    DataFrame with added columns:
        source1_entity_id, candidate_entity_id, candidate_source,
        label (int), negative_type (str)
    """
    df = candidates.copy()

    # Rename and derive standard columns
    df = df.rename(columns={
        "entity_id_left":  "source1_entity_id",
        "entity_id_right": "candidate_entity_id",
    })
    df["candidate_source"] = df["candidate_entity_id"].map(_infer_candidate_source)

    def _label_row(row: pd.Series) -> Tuple[int, str]:
        s1 = row["source1_entity_id"]
        cand = row["candidate_entity_id"]

        # Singleton S1 entities: every candidate is a negative
        if s1 in singleton_s1_ids:
            return 0, "singleton_negative"

        # Check against GT
        gt_targets = s1_to_targets.get(s1, set())
        if cand in gt_targets:
            return 1, "positive"

        # Non-singleton S1 with a non-matching candidate → negative (type assigned later)
        return 0, "unsampled_negative"

    logger.info("Assigning raw labels to %d candidates ...", len(df))
    labels_types = df.apply(_label_row, axis=1, result_type="expand")
    df["label"] = labels_types[0].astype(int)
    df["negative_type"] = labels_types[1]

    return df


def sample_negatives(
    df_labeled: pd.DataFrame,
    cfg: NegativeSamplingConfig,
) -> pd.DataFrame:
    """
    Apply hard/easy negative sampling to an already-labeled DataFrame.

    For entities WITH positives:
      - hard negatives: candidates with blocking_score ≥ hard_score_pct percentile
        of that entity's negative scores. Sample up to hard_neg_per_pos × n_positives.
      - easy negatives: candidates with blocking_score ≤ easy_score_pct percentile.
        Sample up to easy_neg_per_pos × n_positives.
      - All remaining unsampled_negatives are dropped.

    For singleton entities:
      - ALL 'singleton_negative' rows are retained as-is (no sampling).

    Parameters
    ----------
    df_labeled : output of assign_labels() containing 'label', 'negative_type',
                 'blocking_score', 'source1_entity_id' columns.

    Returns
    -------
    DataFrame with negative_type updated to 'hard_negative' or 'easy_negative'
    for selected negatives, and unselected negatives dropped.
    """
    rng = random.Random(cfg.seed)

    positives = df_labeled[df_labeled["label"] == 1].copy()
    singletons = df_labeled[df_labeled["negative_type"] == "singleton_negative"].copy()
    negatives = df_labeled[
        (df_labeled["label"] == 0) & (df_labeled["negative_type"] == "unsampled_negative")
    ].copy()

    logger.info(
        "Sampling: %d positives, %d singleton_negatives, %d unsampled_negatives to process",
        len(positives), len(singletons), len(negatives),
    )

    selected_hard: List[pd.DataFrame] = []
    selected_easy: List[pd.DataFrame] = []
    n_no_hard_pool = 0   # entities where no neg is within delta of TP-max

    # Get all S1 entities with at least one positive
    s1_with_positives: Set[str] = set(positives["source1_entity_id"].unique())
    # Pre-compute per-entity TP-max blocking score for the delta threshold
    tp_max_per_entity: Dict[str, float] = (
        positives.groupby("source1_entity_id")["blocking_score"].max().to_dict()
    )

    for s1_id in s1_with_positives:
        n_pos = int((positives["source1_entity_id"] == s1_id).sum())
        n_hard_target = cfg.hard_neg_per_pos * n_pos
        n_easy_target = cfg.easy_neg_per_pos * n_pos

        ent_negs = negatives[negatives["source1_entity_id"] == s1_id]
        if ent_negs.empty:
            continue

        # --- Hard negatives: entity-relative delta from TP-max score -----------
        # A hard negative must be within `hard_neg_delta` of the entity's best
        # true-match blocking_score.  This ensures it is genuinely confusable
        # with the true match at the blocking stage, not merely high-ranked
        # within a pool that may itself be far from the TP.
        tp_max = tp_max_per_entity.get(s1_id, 1.0)
        hard_threshold = max(0.0, tp_max - cfg.hard_neg_delta)
        hard_pool = ent_negs[ent_negs["blocking_score"] >= hard_threshold]

        # Fallback: if the delta produces an empty pool (entity's negatives are
        # all far below its TP), widen to 2× delta once before giving up.
        if hard_pool.empty:
            hard_pool = ent_negs[ent_negs["blocking_score"] >= max(0.0, tp_max - cfg.hard_neg_delta * 2)]
        if hard_pool.empty:
            n_no_hard_pool += 1

        # --- Easy negatives: bottom percentile of entity's negative distribution
        scores = ent_negs["blocking_score"]
        easy_thresh = scores.quantile(cfg.easy_score_pct)
        easy_pool = ent_negs[ent_negs["blocking_score"] <= easy_thresh]

        # Sample (no replacement; take all if pool < target)
        hard_idx = hard_pool.index.tolist()
        easy_idx = easy_pool.index.tolist()
        rng.shuffle(hard_idx)
        rng.shuffle(easy_idx)

        hard_chosen = hard_idx[:n_hard_target]
        easy_chosen = easy_idx[:n_easy_target]

        if hard_chosen:
            h = ent_negs.loc[hard_chosen].copy()
            h["negative_type"] = "hard_negative"
            selected_hard.append(h)
        if easy_chosen:
            e = ent_negs.loc[easy_chosen].copy()
            e["negative_type"] = "easy_negative"
            selected_easy.append(e)

    if n_no_hard_pool > 0:
        logger.warning(
            "%d entities had no negatives within %.2f of their TP-max score "
            "(even after 2× delta fallback). Those entities will have no hard negatives.",
            n_no_hard_pool, cfg.hard_neg_delta,
        )

    hard_df = pd.concat(selected_hard, ignore_index=True) if selected_hard else pd.DataFrame(columns=df_labeled.columns)
    easy_df = pd.concat(selected_easy, ignore_index=True) if selected_easy else pd.DataFrame(columns=df_labeled.columns)

    result = pd.concat([positives, hard_df, easy_df, singletons], ignore_index=True)

    logger.info(
        "After sampling: %d positives | %d hard_negatives (delta=%.2f) | "
        "%d easy_negatives | %d singleton_negatives → %d total",
        len(positives), len(hard_df), cfg.hard_neg_delta,
        len(easy_df), len(singletons), len(result),
    )
    return result


def build_labeled_pairs(
    candidates: pd.DataFrame,
    s1_to_targets: Dict[str, Set[str]],
    singleton_s1_ids: Set[str],
    sampling_cfg: NegativeSamplingConfig,
) -> pd.DataFrame:
    """
    End-to-end label assignment + negative sampling for one candidate DataFrame.

    Parameters
    ----------
    candidates       : raw blocking output (entity_id_left, entity_id_right,
                        blocking_score, blocking_method)
    s1_to_targets    : GT lookup for the relevant right-side source (s2 or s3)
    singleton_s1_ids : S1 IDs with no GT matches
    sampling_cfg     : NegativeSamplingConfig

    Returns
    -------
    Labeled + sampled DataFrame ready for feature engineering.
    """
    df_raw_labels = assign_labels(candidates, s1_to_targets, singleton_s1_ids)
    df_sampled = sample_negatives(df_raw_labels, sampling_cfg)
    return df_sampled


def compute_label_statistics(df: pd.DataFrame, pair_type: str) -> Dict[str, Any]:
    """
    Compute class-imbalance statistics for a labeled DataFrame.

    Returns a dict with counts, ratios, and a summary string.
    """
    total = len(df)
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    n_hard = int((df["negative_type"] == "hard_negative").sum())
    n_easy = int((df["negative_type"] == "easy_negative").sum())
    n_singleton = int((df["negative_type"] == "singleton_negative").sum())

    imbalance_ratio = round(n_neg / max(n_pos, 1), 2)

    stats = {
        "pair_type":          pair_type,
        "total_pairs":        total,
        "positives":          n_pos,
        "negatives":          n_neg,
        "hard_negatives":     n_hard,
        "easy_negatives":     n_easy,
        "singleton_negatives": n_singleton,
        "imbalance_ratio_neg_per_pos": imbalance_ratio,
        "positive_rate_pct":  round(n_pos / max(total, 1) * 100, 2),
    }
    return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    CLI: load enriched candidates + GT, assign labels, sample negatives,
    upload labeled_pairs_S1_{S2,S3}.parquet to S3 FEATURES prefix.
    """
    from src.entity_resolution.utils.io import read_parquet_s3, write_parquet_s3, write_json_s3

    parser = argparse.ArgumentParser(description="Task A — label generation for candidate pairs.")
    add_account_prefix_arg(parser)
    parser.add_argument("--aws-profile",       default="amazon-ml-account3")
    parser.add_argument("--gt-path",           default="/tmp/train_ground_truth.tsv",
                        help="Local path to train_ground_truth.tsv")
    # Enriched candidate S3 URIs (default to account3 enriched)
    parser.add_argument("--candidates-s1-s2",
                        default="s3://amzn-s3-ml-c/account3/candidates/candidates_S1_S2_enriched.parquet")
    parser.add_argument("--candidates-s1-s3",
                        default="s3://amzn-s3-ml-c/account3/candidates/candidates_S1_S3_enriched.parquet")
    parser.add_argument("--hard-neg-per-pos",  type=int,   default=2)
    parser.add_argument("--easy-neg-per-pos",  type=int,   default=2)
    parser.add_argument("--hard-neg-delta",    type=float, default=0.10,
                        help="Hard negatives must be within this delta of the entity's "
                             "max TP blocking_score (default 0.10)")
    parser.add_argument("--easy-score-pct",    type=float, default=0.30)
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cfg = resolve_config_from_args(args)
    paths = cfg.paths
    profile = cfg.aws_profile

    sampling_cfg = NegativeSamplingConfig(
        hard_neg_per_pos=args.hard_neg_per_pos,
        easy_neg_per_pos=args.easy_neg_per_pos,
        hard_neg_delta=args.hard_neg_delta,
        easy_score_pct=args.easy_score_pct,
        seed=args.seed,
    )

    # 1. Parse ground truth (local file — already downloaded during enriched sample build)
    s1_to_s2, s1_to_s3, singleton_ids = parse_ground_truth(args.gt_path)

    # 2. Process S1×S2
    logger.info("Loading S1×S2 candidates from %s ...", args.candidates_s1_s2)
    cands_s1_s2 = read_parquet_s3(args.candidates_s1_s2, profile=profile)
    labeled_s1_s2 = build_labeled_pairs(cands_s1_s2, s1_to_s2, singleton_ids, sampling_cfg)
    stats_s1_s2 = compute_label_statistics(labeled_s1_s2, "S1_S2")

    # 3. Process S1×S3
    logger.info("Loading S1×S3 candidates from %s ...", args.candidates_s1_s3)
    cands_s1_s3 = read_parquet_s3(args.candidates_s1_s3, profile=profile)
    labeled_s1_s3 = build_labeled_pairs(cands_s1_s3, s1_to_s3, singleton_ids, sampling_cfg)
    stats_s1_s3 = compute_label_statistics(labeled_s1_s3, "S1_S3")

    # 4. Upload to S3
    out_s1_s2 = f"{paths.FEATURES}labeled_pairs_S1_S2.parquet"
    out_s1_s3 = f"{paths.FEATURES}labeled_pairs_S1_S3.parquet"
    write_parquet_s3(labeled_s1_s2, out_s1_s2, profile=profile)
    write_parquet_s3(labeled_s1_s3, out_s1_s3, profile=profile)
    logger.info("Labeled pairs uploaded: %s | %s", out_s1_s2, out_s1_s3)

    # 5. Save stats report
    report = {"S1_S2": stats_s1_s2, "S1_S3": stats_s1_s3}
    report_uri = f"{paths.REPORTS}labeling_stats.json"
    write_json_s3(report, report_uri, profile=profile)

    # Also save local copies
    labeled_s1_s2.to_parquet("/tmp/labeled_pairs_S1_S2.parquet", index=False)
    labeled_s1_s3.to_parquet("/tmp/labeled_pairs_S1_S3.parquet", index=False)

    # 6. Print comparison
    import json
    print("\n" + "=" * 70)
    print("TASK A — LABELING STATISTICS")
    print("=" * 70)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
