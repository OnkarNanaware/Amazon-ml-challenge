"""
blocking_experiments.py
=======================
Account 2 — Blocking Experimentation Module.

This module implements three experiment sets that systematically improve on
Account 1's baseline blocking (3-pass: TF-IDF name + token-sort + address prefix).

EXPERIMENT PHILOSOPHY
---------------------
Account 1 shipped a working baseline.  Our job is NOT to reimplement it — it is
to find where the recall/cost curve can be pushed further before handing a
winning configuration back for canonical adoption.

Each experiment is a **pure function**:
  - Input: DataFrame(s) with the schema from Account 1's normalized files
  - Output: candidate pairs DataFrame + experiment-result dict
  - No S3 reads/writes, no hardcoded paths, no hardcoded account prefix

This keeps the functions directly importable into SageMaker Processing Jobs
without modification.

GROUND TRUTH CONTRACT (locked in from Account 1's audit)
---------------------------------------------------------
- train_ground_truth.tsv: source1_entity_id -> comma-separated matched_entity_ids
- ALL matches are cross-source: S1->S2 or S1->S3 only
- NO S1xS1, S2xS2, S3xS3, or direct S2xS3 pairs exist in GT
- One S1 entity can match MULTIPLE S2 AND multiple S3 records simultaneously
- Some S1 entities have EMPTY matched_entity_ids (singletons -- no true match)
- Singletons must be tracked SEPARATELY; they must NEVER affect recall metrics

RECALL DEFINITION
-----------------
For pair type "S1_S2":
  recall = (S2 targets found in S1->S2 candidate set) / (total S2 targets in GT)
  [computed over only those S1 entities that have >= 1 S2 target]

The same applies symmetrically for "S1_S3".
Singletons for a given pair type are tracked via singleton_s1_count and
singleton_cand_vol fields in ExperimentResult, never mixed into the recall %.

KNOWN NORMALIZATION EDGE CASES TO WATCH
----------------------------------------
- Legal suffixes (LLC, Inc, Ltd, Co) can appear in LEADING position
  (e.g. "LLC Moncada Learning Center"). Account 1 fixed trailing+leading strip,
  but there may be other unseen patterns at scale (suffix mid-string, multiple
  legal tokens, non-English legal forms). If experiments surface new failure
  modes, report them -- do not silently work around them in blocking code.

AUTHORS: Account 2 (blocking experiments)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column constants (must match normalization.py output and blocking.py)
# ---------------------------------------------------------------------------
_ENTITY_ID    = "entity_id"
_NORM_NAME    = "normalized_name"
_NORM_ADDR    = "normalized_address"
_ADDR_MISSING = "address_missing"
_COUNTRY      = "country"
_SOURCE       = "source"
_TOKEN_SORTED = "token_sorted_name"

_LEFT_ID  = "entity_id_left"
_RIGHT_ID = "entity_id_right"
_SCORE    = "blocking_score"
_METHOD   = "blocking_method"


# ---------------------------------------------------------------------------
# Ground-truth helpers (mirrored from recall_s1_s2_s3.py -- kept pure/local)
# ---------------------------------------------------------------------------

@dataclass
class GTRecord:
    s1_id:        str
    s2_targets:   Set[str]
    s3_targets:   Set[str]
    is_singleton: bool   # True if matched_entity_ids was empty


def parse_ground_truth_df(df_gt: pd.DataFrame) -> List[GTRecord]:
    """
    Parse a ground-truth DataFrame into GTRecord objects.

    Expected columns: source1_entity_id, matched_entity_ids.
    The matched_entity_ids column may be empty/NaN for singleton S1 entities.

    Parameters
    ----------
    df_gt : pd.DataFrame
        Raw GT DataFrame (loaded from train_ground_truth.tsv or a sample).

    Returns
    -------
    List[GTRecord]
    """
    df_gt = df_gt.copy()
    df_gt.columns = df_gt.columns.str.strip()
    records: List[GTRecord] = []

    for _, row in df_gt.iterrows():
        s1_id   = str(row["source1_entity_id"]).strip()
        raw_ids = str(row.get("matched_entity_ids", "")).strip()

        if not raw_ids or raw_ids == "nan":
            records.append(GTRecord(s1_id=s1_id, s2_targets=set(),
                                    s3_targets=set(), is_singleton=True))
            continue

        all_ids    = [x.strip() for x in raw_ids.split(",") if x.strip()]
        s2_targets = {x for x in all_ids if x.startswith("S2-")}
        s3_targets = {x for x in all_ids if x.startswith("S3-")}
        records.append(GTRecord(s1_id=s1_id, s2_targets=s2_targets,
                                s3_targets=s3_targets, is_singleton=False))

    n_sing = sum(1 for r in records if r.is_singleton)
    logger.info("GT: %d S1 entities (%d singletons, %d with matches)",
                len(records), n_sing, len(records) - n_sing)
    return records


def _build_cand_map(df_cands: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build {s1_id: {candidate_right_ids}} from a candidate DataFrame.
    Direction-aware: entity_id_left is always S1.
    """
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for lid, rid in zip(df_cands[_LEFT_ID], df_cands[_RIGHT_ID]):
        mapping[str(lid)].add(str(rid))
    return dict(mapping)


# ---------------------------------------------------------------------------
# Recall evaluation (correct GT structure)
# ---------------------------------------------------------------------------

@dataclass
class PairRecall:
    pair_type:                 str    # "S1_S2" or "S1_S3"
    s1_entities_in_gt:         int
    total_gt_targets:          int
    found_targets:             int
    recall:                    float
    total_candidates:          int
    avg_candidates_per_entity: float  # over ALL S1 entities -- compute proxy
    singleton_s1_count:        int
    singleton_cand_vol:        int


def evaluate_candidates(
    gt_records: List[GTRecord],
    df_cands:   pd.DataFrame,
    pair_type:  str,           # "S1_S2" or "S1_S3"
) -> PairRecall:
    """
    Compute recall for one pair type.

    Singletons (no targets of this type) are excluded from the recall numerator
    and denominator -- they are tracked separately to avoid metric contamination.

    Parameters
    ----------
    gt_records : List[GTRecord] from parse_ground_truth_df().
    df_cands   : Candidate DataFrame with entity_id_left (S1) and entity_id_right.
    pair_type  : "S1_S2" or "S1_S3".
    """
    cand_map    = _build_cand_map(df_cands)
    total_cands = sum(len(v) for v in cand_map.values())
    n_s1_total  = len(cand_map)
    avg_cands   = total_cands / n_s1_total if n_s1_total > 0 else 0.0

    attr = "s2_targets" if pair_type == "S1_S2" else "s3_targets"

    n_entities    = 0
    total_targets = 0
    found         = 0
    sing_count    = 0
    sing_vol      = 0

    for rec in gt_records:
        targets = getattr(rec, attr)
        if not targets:
            sing_count += 1
            sing_vol   += len(cand_map.get(rec.s1_id, set()))
            continue
        n_entities    += 1
        total_targets += len(targets)
        cands_for_s1   = cand_map.get(rec.s1_id, set())
        found         += len(targets & cands_for_s1)

    recall = found / total_targets if total_targets > 0 else 0.0

    return PairRecall(
        pair_type                 = pair_type,
        s1_entities_in_gt         = n_entities,
        total_gt_targets          = total_targets,
        found_targets             = found,
        recall                    = recall,
        total_candidates          = total_cands,
        avg_candidates_per_entity = avg_cands,
        singleton_s1_count        = sing_count,
        singleton_cand_vol        = sing_vol,
    )


def evaluate_address_missing_subset(
    gt_records: List[GTRecord],
    df_cands:   pd.DataFrame,
    df_s1:      pd.DataFrame,
    pair_type:  str,
) -> Dict:
    """
    Compute recall specifically for the address_missing=True subset of S1 entities.

    Returns a dict with: n_s1_addr_missing_with_targets, recall, found, total_gt_targets.
    """
    addr_missing_ids: Set[str] = set(
        df_s1.loc[df_s1[_ADDR_MISSING] == True, _ENTITY_ID].astype(str)
    )

    attr     = "s2_targets" if pair_type == "S1_S2" else "s3_targets"
    cand_map = _build_cand_map(df_cands)

    n_entities    = 0
    total_targets = 0
    found         = 0

    for rec in gt_records:
        if rec.s1_id not in addr_missing_ids:
            continue
        targets = getattr(rec, attr)
        if not targets:
            continue
        n_entities    += 1
        total_targets += len(targets)
        found         += len(targets & cand_map.get(rec.s1_id, set()))

    recall = found / total_targets if total_targets > 0 else None  # None if subset empty
    return {
        "n_s1_addr_missing_with_targets": n_entities,
        "total_gt_targets":               total_targets,
        "found_targets":                  found,
        "recall":                         recall,
    }


def evaluate_by_country(
    gt_records: List[GTRecord],
    df_cands:   pd.DataFrame,
    df_s1:      pd.DataFrame,
    pair_type:  str,
    top_n:      int = 10,
) -> List[Dict]:
    """
    Break down recall by country (from df_s1).
    Returns the top_n most frequent countries with their recall stats.
    """
    attr     = "s2_targets" if pair_type == "S1_S2" else "s3_targets"
    cand_map = _build_cand_map(df_cands)

    # Build s1_id -> country map
    id_to_country: Dict[str, str] = {}
    if _COUNTRY in df_s1.columns:
        for eid, cntry in zip(df_s1[_ENTITY_ID].astype(str),
                               df_s1[_COUNTRY].fillna("unknown")):
            id_to_country[eid] = str(cntry)

    country_stats: Dict[str, Dict] = defaultdict(lambda: {
        "n_entities": 0, "total_targets": 0, "found": 0
    })

    for rec in gt_records:
        targets = getattr(rec, attr)
        if not targets:
            continue
        cntry = id_to_country.get(rec.s1_id, "unknown")
        s = country_stats[cntry]
        s["n_entities"]    += 1
        s["total_targets"] += len(targets)
        s["found"]         += len(targets & cand_map.get(rec.s1_id, set()))

    rows = []
    for cntry, s in sorted(country_stats.items(),
                            key=lambda x: x[1]["total_targets"], reverse=True)[:top_n]:
        tt = s["total_targets"]
        rows.append({
            "country":          cntry,
            "s1_entities":      s["n_entities"],
            "total_gt_targets": tt,
            "found":            s["found"],
            "recall":           round(s["found"] / tt, 4) if tt > 0 else None,
        })
    return rows


# ---------------------------------------------------------------------------
# ExperimentResult -- logged to the consolidated JSON report
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Single experiment outcome, serialisable to JSON."""
    name:               str
    experiment_set:     int          # 1, 2, or 3
    config:             Dict         # hyper-params / choices tested
    s1_s2:              PairRecall
    s1_s3:              PairRecall
    addr_missing_s1_s2: Dict
    addr_missing_s1_s3: Dict
    country_s1_s2:      List[Dict]
    country_s1_s3:      List[Dict]
    recommendation:     str          # "keep" | "drop" | "tune-further"
    notes:              str          # one-line human rationale
    generated_at:       str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Utility: empty candidates DataFrame
# ---------------------------------------------------------------------------

def _empty_cands() -> pd.DataFrame:
    return pd.DataFrame(columns=[_LEFT_ID, _RIGHT_ID, _SCORE, _METHOD])


# ---------------------------------------------------------------------------
# Core blocking primitives used by experiments
# ---------------------------------------------------------------------------

def _tfidf_candidates(
    df_left:    pd.DataFrame,
    df_right:   pd.DataFrame,
    top_k:      int,
    ngram_min:  int   = 2,
    ngram_max:  int   = 3,
    min_sim:    float = 0.05,
    text_col:   str   = _NORM_NAME,
    method_tag: str   = "tfidf_name",
) -> pd.DataFrame:
    """
    TF-IDF character n-gram top-K blocking on a chosen text column.
    Pure function: no I/O.
    """
    left  = df_left.dropna(subset=[text_col]).reset_index(drop=True)
    right = df_right.dropna(subset=[text_col]).reset_index(drop=True)

    if left.empty or right.empty:
        return _empty_cands()

    all_text = pd.concat([left[text_col], right[text_col]], ignore_index=True)
    vectorizer = TfidfVectorizer(
        analyzer     = "char_wb",
        ngram_range  = (ngram_min, ngram_max),
        min_df       = 1,
        sublinear_tf = True,
    )
    vectorizer.fit(all_text)
    mat_left  = vectorizer.transform(left[text_col])
    mat_right = vectorizer.transform(right[text_col])

    rows: List[Dict] = []
    batch_size = 1_000

    for start in range(0, mat_left.shape[0], batch_size):
        end   = min(start + batch_size, mat_left.shape[0])
        sims  = linear_kernel(mat_left[start:end], mat_right)

        for local_i, sim_row in enumerate(sims):
            actual_k = min(top_k, len(sim_row))
            top_idx  = np.argpartition(sim_row, -actual_k)[-actual_k:]
            top_idx  = top_idx[sim_row[top_idx] >= min_sim]
            top_idx  = top_idx[np.argsort(sim_row[top_idx])[::-1]]

            lid = str(left[_ENTITY_ID].iloc[start + local_i])
            for j in top_idx:
                rid = str(right[_ENTITY_ID].iloc[j])
                if lid == rid:
                    continue
                rows.append({_LEFT_ID: lid, _RIGHT_ID: rid,
                             _SCORE: float(sim_row[j]), _METHOD: method_tag})

    return pd.DataFrame(rows, columns=[_LEFT_ID, _RIGHT_ID, _SCORE, _METHOD])


def _token_sort_candidates(
    df_left:    pd.DataFrame,
    df_right:   pd.DataFrame,
    prefix_len: int = 5,
    method_tag: str = "token_sort",
) -> pd.DataFrame:
    """Token-sort prefix shard blocking on token_sorted_name."""
    left  = df_left.dropna(subset=[_TOKEN_SORTED]).copy()
    right = df_right.dropna(subset=[_TOKEN_SORTED]).copy()
    if left.empty or right.empty:
        return _empty_cands()

    left["_shard"]  = left[_TOKEN_SORTED].str[:prefix_len]
    right["_shard"] = right[_TOKEN_SORTED].str[:prefix_len]

    merged = pd.merge(
        left[[_ENTITY_ID, "_shard"]].rename(columns={_ENTITY_ID: _LEFT_ID}),
        right[[_ENTITY_ID, "_shard"]].rename(columns={_ENTITY_ID: _RIGHT_ID}),
        on="_shard",
    )
    merged = merged[merged[_LEFT_ID] != merged[_RIGHT_ID]]
    merged[_SCORE]  = 1.0
    merged[_METHOD] = method_tag
    return merged[[_LEFT_ID, _RIGHT_ID, _SCORE, _METHOD]].reset_index(drop=True)


def _address_prefix_candidates(
    df_left:    pd.DataFrame,
    df_right:   pd.DataFrame,
    prefix_len: int = 8,
    method_tag: str = "address_prefix",
) -> pd.DataFrame:
    """Address numeric-prefix blocking (only for rows with address_missing=False)."""
    def _key(row):
        if row.get(_ADDR_MISSING, True):
            return None
        addr = row.get(_NORM_ADDR)
        if pd.isna(addr) or not str(addr).strip():
            return None
        return str(addr)[:prefix_len]

    left  = df_left.copy()
    right = df_right.copy()
    left["_akey"]  = left.apply(_key, axis=1)
    right["_akey"] = right.apply(_key, axis=1)
    left  = left.dropna(subset=["_akey"])
    right = right.dropna(subset=["_akey"])
    if left.empty or right.empty:
        return _empty_cands()

    merged = pd.merge(
        left[[_ENTITY_ID, "_akey"]].rename(columns={_ENTITY_ID: _LEFT_ID}),
        right[[_ENTITY_ID, "_akey"]].rename(columns={_ENTITY_ID: _RIGHT_ID}),
        on="_akey",
    )
    merged = merged[merged[_LEFT_ID] != merged[_RIGHT_ID]]
    merged[_SCORE]  = 1.0
    merged[_METHOD] = method_tag
    return merged[[_LEFT_ID, _RIGHT_ID, _SCORE, _METHOD]].reset_index(drop=True)


def _rare_token_candidates(
    df_left:        pd.DataFrame,
    df_right:       pd.DataFrame,
    rarity_pct:     float = 0.005,   # tokens appearing in < X% of corpus
    text_col:       str   = _NORM_NAME,
    method_tag:     str   = "rare_token",
    max_per_entity: int   = 50,
) -> pd.DataFrame:
    """
    Rare-token inverted index blocking.

    Tokens that appear in fewer than `rarity_pct` fraction of records in the
    combined corpus are considered "rare" and act as high-priority blocking keys.
    For each left entity, we collect all right entities that share at least one
    rare token, scored by the number of shared rare tokens normalized by total
    rare tokens of the left entity.

    Parameters
    ----------
    df_left / df_right : DataFrames with text_col and entity_id.
    rarity_pct         : Corpus fraction threshold (e.g. 0.005 = <0.5%).
    text_col           : Column to tokenize (default: normalized_name).
    method_tag         : blocking_method label.
    max_per_entity     : Cap on candidates per left entity.
    """
    left  = df_left.dropna(subset=[text_col]).copy()
    right = df_right.dropna(subset=[text_col]).copy()
    if left.empty or right.empty:
        return _empty_cands()

    right_n = len(right)
    token_doc_freq: Dict[str, int] = defaultdict(int)
    right_token_sets: List[Set[str]] = []

    for txt in right[text_col]:
        toks = set(str(txt).split())
        right_token_sets.append(toks)
        for t in toks:
            token_doc_freq[t] += 1

    rarity_threshold = max(int(rarity_pct * right_n), 1)

    rare_tokens: Set[str] = {t for t, f in token_doc_freq.items()
                             if f <= rarity_threshold}

    if not rare_tokens:
        logger.warning(
            "rare_token: no tokens meet rarity_pct=%.4f (corpus=%d, threshold=%d)",
            rarity_pct, right_n, rarity_threshold,
        )
        return _empty_cands()

    # Build inverted index: rare_token -> list of right indices
    inverted: Dict[str, List[int]] = defaultdict(list)
    for i, tset in enumerate(right_token_sets):
        for t in tset:
            if t in rare_tokens:
                inverted[t].append(i)

    right_ids = right[_ENTITY_ID].astype(str).tolist()

    rows: List[Dict] = []
    for _, lrow in left.iterrows():
        lid  = str(lrow[_ENTITY_ID])
        toks = set(str(lrow[text_col]).split()) & rare_tokens
        if not toks:
            continue

        right_counts: Dict[int, int] = defaultdict(int)
        for t in toks:
            for ri in inverted.get(t, []):
                right_counts[ri] += 1

        n_left_rare = len(toks)
        scored = sorted(right_counts.items(), key=lambda x: x[1], reverse=True)

        added = 0
        for ri, cnt in scored:
            rid = right_ids[ri]
            if rid == lid:
                continue
            rows.append({
                _LEFT_ID: lid,
                _RIGHT_ID: rid,
                _SCORE: float(cnt) / n_left_rare,
                _METHOD: method_tag,
            })
            added += 1
            if added >= max_per_entity:
                break

    return pd.DataFrame(rows, columns=[_LEFT_ID, _RIGHT_ID, _SCORE, _METHOD])


def _postal_char_ngram_candidates(
    df_left:   pd.DataFrame,
    df_right:  pd.DataFrame,
    top_k:     int   = 50,
    ngram_min: int   = 2,
    ngram_max: int   = 4,
    min_sim:   float = 0.05,
) -> pd.DataFrame:
    """
    Experiment Set 3 -- character n-gram blocking on the NUMERIC tokens of
    normalized_address only (postal-code / street-number clustering).

    Extracts only digit sequences from the normalized address, joins them into
    a synthetic key, and TF-IDF-blocks on that key.  Tighter than Strategy G
    (full address prefix) -- targets postal-code-like clustering.

    Only applied to rows where address_missing=False.
    """
    def _numeric_key(addr: str) -> Optional[str]:
        tokens = re.findall(r"\d+", str(addr))
        return " ".join(tokens) if tokens else None

    left  = df_left[df_left[_ADDR_MISSING] == False].copy()
    right = df_right[df_right[_ADDR_MISSING] == False].copy()
    left  = left.dropna(subset=[_NORM_ADDR])
    right = right.dropna(subset=[_NORM_ADDR])

    left["_numkey"]  = left[_NORM_ADDR].apply(_numeric_key)
    right["_numkey"] = right[_NORM_ADDR].apply(_numeric_key)

    left  = left.dropna(subset=["_numkey"]).reset_index(drop=True)
    right = right.dropna(subset=["_numkey"]).reset_index(drop=True)

    if left.empty or right.empty:
        return _empty_cands()

    # Reuse _tfidf_candidates on the numeric key column
    left_tmp  = left[[_ENTITY_ID, "_numkey"]].rename(columns={"_numkey": _NORM_NAME})
    right_tmp = right[[_ENTITY_ID, "_numkey"]].rename(columns={"_numkey": _NORM_NAME})

    return _tfidf_candidates(
        left_tmp, right_tmp, top_k=top_k,
        ngram_min=ngram_min, ngram_max=ngram_max,
        min_sim=min_sim, method_tag="postal_ngram",
    )


def _country_scoped_tfidf_candidates(
    df_left:    pd.DataFrame,
    df_right:   pd.DataFrame,
    top_k:      int   = 50,
    ngram_min:  int   = 2,
    ngram_max:  int   = 3,
    min_sim:    float = 0.05,
    text_col:   str   = _NORM_NAME,
    method_tag: str   = "country_tfidf_name",
) -> pd.DataFrame:
    """
    Experiment Set 3 -- TF-IDF blocking restricted to same-country pairs.

    For each country present in df_left, runs TF-IDF only against df_right
    entities from the same country.  Falls back to unconstrained TF-IDF for
    entities whose country is missing/unknown.

    Hypothesis: cross-country false candidates consume budget without adding
    recall, so scoping to country improves precision at equal recall.
    """
    if _COUNTRY not in df_left.columns or _COUNTRY not in df_right.columns:
        logger.warning(
            "country_scoped_tfidf: 'country' column missing -- "
            "falling back to unconstrained TF-IDF"
        )
        return _tfidf_candidates(
            df_left, df_right, top_k=top_k,
            ngram_min=ngram_min, ngram_max=ngram_max,
            min_sim=min_sim, text_col=text_col, method_tag=method_tag,
        )

    left  = df_left.dropna(subset=[text_col]).copy()
    right = df_right.dropna(subset=[text_col]).copy()
    left[_COUNTRY]  = left[_COUNTRY].fillna("unknown").astype(str)
    right[_COUNTRY] = right[_COUNTRY].fillna("unknown").astype(str)

    countries = set(left[_COUNTRY].unique()) | set(right[_COUNTRY].unique())
    all_dfs: List[pd.DataFrame] = []

    for country in countries:
        l_sub = left[left[_COUNTRY] == country]
        r_sub = right[right[_COUNTRY] == country]
        if l_sub.empty or r_sub.empty:
            continue
        chunk = _tfidf_candidates(
            l_sub, r_sub, top_k=top_k,
            ngram_min=ngram_min, ngram_max=ngram_max,
            min_sim=min_sim, text_col=text_col, method_tag=method_tag,
        )
        all_dfs.append(chunk)

    if not all_dfs:
        return _empty_cands()
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Candidate merging (dedup + per-entity cap)
# ---------------------------------------------------------------------------

def merge_and_cap(
    dfs:            List[pd.DataFrame],
    max_per_entity: int = 100,
) -> pd.DataFrame:
    """
    Union candidate DataFrames, deduplicate pairs (keeping max score),
    and apply a per-entity candidate cap.

    For cross-source (S1xS2, S1xS3) the convention is that entity_id_left is
    always the S1 id.  Direction swapping must happen at the caller level.

    Scoring: when a pair appears via multiple methods, keep the MAX score.
    Methods: concatenated as a '+'-joined sorted string.
    """
    non_empty = [d for d in dfs if not d.empty]
    if not non_empty:
        return _empty_cands()

    combined = pd.concat(non_empty, ignore_index=True)
    if combined.empty:
        return _empty_cands()

    agg = (
        combined
        .groupby([_LEFT_ID, _RIGHT_ID], sort=False)
        .agg(
            blocking_score  = (_SCORE,  "max"),
            blocking_method = (_METHOD, lambda m: "+".join(sorted(set(m)))),
        )
        .reset_index()
    )

    agg = (
        agg
        .sort_values(_SCORE, ascending=False)
        .groupby(_LEFT_ID, sort=False)
        .head(max_per_entity)
        .reset_index(drop=True)
    )
    return agg


# ============================================================================
# EXPERIMENT SET 1 -- Parameter tuning on existing strategies
# ============================================================================

def exp1_tfidf_topk_sweep(
    df_s1:        pd.DataFrame,
    df_s2:        pd.DataFrame,
    df_s3:        pd.DataFrame,
    gt_records:   List[GTRecord],
    top_k_values: List[int]  = (10, 25, 50, 100),
    global_cap:   int        = 100,
) -> List[ExperimentResult]:
    """
    Experiment 1a -- TF-IDF top-K sweep.

    Tests top-K in {10, 25, 50, 100} for TF-IDF name blocking in isolation.
    Measures recall vs. candidate volume at each K to identify the
    diminishing-returns point for parameter selection.
    """
    results: List[ExperimentResult] = []

    for k in top_k_values:
        logger.info("exp1_tfidf_topk_sweep: K=%d", k)

        cands_s1_s2 = merge_and_cap(
            [_tfidf_candidates(df_s1, df_s2, top_k=k, method_tag="tfidf_name")],
            max_per_entity=global_cap,
        )
        cands_s1_s3 = merge_and_cap(
            [_tfidf_candidates(df_s1, df_s3, top_k=k, method_tag="tfidf_name")],
            max_per_entity=global_cap,
        )

        pr_s2 = evaluate_candidates(gt_records, cands_s1_s2, "S1_S2")
        pr_s3 = evaluate_candidates(gt_records, cands_s1_s3, "S1_S3")
        am_s2 = evaluate_address_missing_subset(gt_records, cands_s1_s2, df_s1, "S1_S2")
        am_s3 = evaluate_address_missing_subset(gt_records, cands_s1_s3, df_s1, "S1_S3")
        cy_s2 = evaluate_by_country(gt_records, cands_s1_s2, df_s1, "S1_S2")
        cy_s3 = evaluate_by_country(gt_records, cands_s1_s3, df_s1, "S1_S3")

        avg_recall = (pr_s2.recall + pr_s3.recall) / 2

        results.append(ExperimentResult(
            name           = f"exp1a_tfidf_topk_{k}",
            experiment_set = 1,
            config         = {
                "strategy":    "tfidf_name_only",
                "top_k":       k,
                "global_cap":  global_cap,
                "ngram_range": "(2,3)",
            },
            s1_s2               = pr_s2,
            s1_s3               = pr_s3,
            addr_missing_s1_s2  = am_s2,
            addr_missing_s1_s3  = am_s3,
            country_s1_s2       = cy_s2,
            country_s1_s3       = cy_s3,
            recommendation      = "tune-further",
            notes               = (
                f"TF-IDF name only, K={k}. avg_recall={avg_recall:.4f}. "
                f"avg_cands_s1s2={pr_s2.avg_candidates_per_entity:.1f}, "
                f"avg_cands_s1s3={pr_s3.avg_candidates_per_entity:.1f}"
            ),
        ))

    return results


def exp1_rare_token_rarity_sweep(
    df_s1:          pd.DataFrame,
    df_s2:          pd.DataFrame,
    df_s3:          pd.DataFrame,
    gt_records:     List[GTRecord],
    rarity_pcts:    List[float] = (0.001, 0.005, 0.01),
    max_per_entity: int         = 50,
    global_cap:     int         = 100,
) -> List[ExperimentResult]:
    """
    Experiment 1b -- Rare-token inverted index rarity threshold sweep.

    Tests token-rarity thresholds (<0.1%, <0.5%, <1% of corpus records).
    Tighter rarity = more distinctive tokens = potentially higher precision
    with less volume, but may hurt recall if true matches lack rare tokens.
    """
    results: List[ExperimentResult] = []

    for pct in rarity_pcts:
        logger.info("exp1_rare_token_rarity_sweep: rarity_pct=%.4f", pct)

        cands_s1_s2 = merge_and_cap(
            [_rare_token_candidates(df_s1, df_s2, rarity_pct=pct,
                                   max_per_entity=max_per_entity)],
            max_per_entity=global_cap,
        )
        cands_s1_s3 = merge_and_cap(
            [_rare_token_candidates(df_s1, df_s3, rarity_pct=pct,
                                   max_per_entity=max_per_entity)],
            max_per_entity=global_cap,
        )

        pr_s2 = evaluate_candidates(gt_records, cands_s1_s2, "S1_S2")
        pr_s3 = evaluate_candidates(gt_records, cands_s1_s3, "S1_S3")
        am_s2 = evaluate_address_missing_subset(gt_records, cands_s1_s2, df_s1, "S1_S2")
        am_s3 = evaluate_address_missing_subset(gt_records, cands_s1_s3, df_s1, "S1_S3")
        cy_s2 = evaluate_by_country(gt_records, cands_s1_s2, df_s1, "S1_S2")
        cy_s3 = evaluate_by_country(gt_records, cands_s1_s3, df_s1, "S1_S3")

        avg_recall = (pr_s2.recall + pr_s3.recall) / 2

        results.append(ExperimentResult(
            name           = f"exp1b_rare_token_rarity_{pct}",
            experiment_set = 1,
            config         = {
                "strategy":        "rare_token_name",
                "rarity_pct":      pct,
                "max_per_entity":  max_per_entity,
                "global_cap":      global_cap,
            },
            s1_s2               = pr_s2,
            s1_s3               = pr_s3,
            addr_missing_s1_s2  = am_s2,
            addr_missing_s1_s3  = am_s3,
            country_s1_s2       = cy_s2,
            country_s1_s3       = cy_s3,
            recommendation      = "tune-further",
            notes               = (
                f"rare_token name, rarity_pct={pct:.4f}. "
                f"avg_recall={avg_recall:.4f}. "
                f"avg_cands_s1s2={pr_s2.avg_candidates_per_entity:.1f}"
            ),
        ))

    return results


def exp1_global_cap_sweep(
    df_s1:      pd.DataFrame,
    df_s2:      pd.DataFrame,
    df_s3:      pd.DataFrame,
    gt_records: List[GTRecord],
    cap_values: List[int] = (50, 100, 200),
) -> List[ExperimentResult]:
    """
    Experiment 1c -- Global candidate cap sweep.

    Runs the full baseline 3-pass stack (tfidf_name + token_sort + address_prefix)
    at each global cap level to measure recall gain vs. compute cost.
    Builds the uncapped candidates once and applies different caps.
    """
    results: List[ExperimentResult] = []
    max_k = max(cap_values)

    logger.info("exp1_global_cap_sweep: building uncapped candidates (3-pass, top_k=%d)...", max_k)

    raw_s2 = [
        _tfidf_candidates(df_s1, df_s2, top_k=max_k, method_tag="tfidf_name"),
        _token_sort_candidates(df_s1, df_s2, method_tag="token_sort"),
        _address_prefix_candidates(df_s1, df_s2, method_tag="address_prefix"),
    ]
    raw_s3 = [
        _tfidf_candidates(df_s1, df_s3, top_k=max_k, method_tag="tfidf_name"),
        _token_sort_candidates(df_s1, df_s3, method_tag="token_sort"),
        _address_prefix_candidates(df_s1, df_s3, method_tag="address_prefix"),
    ]

    for cap in cap_values:
        logger.info("exp1_global_cap_sweep: cap=%d", cap)

        cands_s1_s2 = merge_and_cap(raw_s2, max_per_entity=cap)
        cands_s1_s3 = merge_and_cap(raw_s3, max_per_entity=cap)

        pr_s2 = evaluate_candidates(gt_records, cands_s1_s2, "S1_S2")
        pr_s3 = evaluate_candidates(gt_records, cands_s1_s3, "S1_S3")
        am_s2 = evaluate_address_missing_subset(gt_records, cands_s1_s2, df_s1, "S1_S2")
        am_s3 = evaluate_address_missing_subset(gt_records, cands_s1_s3, df_s1, "S1_S3")
        cy_s2 = evaluate_by_country(gt_records, cands_s1_s2, df_s1, "S1_S2")
        cy_s3 = evaluate_by_country(gt_records, cands_s1_s3, df_s1, "S1_S3")

        avg_recall = (pr_s2.recall + pr_s3.recall) / 2

        results.append(ExperimentResult(
            name           = f"exp1c_global_cap_{cap}",
            experiment_set = 1,
            config         = {
                "strategy":     "baseline_3pass",
                "global_cap":   cap,
                "tfidf_top_k":  max_k,
            },
            s1_s2               = pr_s2,
            s1_s3               = pr_s3,
            addr_missing_s1_s2  = am_s2,
            addr_missing_s1_s3  = am_s3,
            country_s1_s2       = cy_s2,
            country_s1_s3       = cy_s3,
            recommendation      = "tune-further",
            notes               = (
                f"Baseline 3-pass, global_cap={cap}. "
                f"avg_recall={avg_recall:.4f}. "
                f"avg_cands_s1s2={pr_s2.avg_candidates_per_entity:.1f}"
            ),
        ))

    return results


# ============================================================================
# EXPERIMENT SET 2 -- Strategy combination logic
# ============================================================================

def exp2_union_vs_weighted(
    df_s1:       pd.DataFrame,
    df_s2:       pd.DataFrame,
    df_s3:       pd.DataFrame,
    gt_records:  List[GTRecord],
    tfidf_top_k: int = 50,
    global_cap:  int = 100,
) -> List[ExperimentResult]:
    """
    Experiment 2a -- Union (baseline) vs. weighted-priority combination.

    UNION: all 3 passes merged with equal footing, capped by score.
    WEIGHTED: TF-IDF gets 2x score multiplier so it ranks first in the cap,
              while token-sort and address-prefix keep their original scores.

    Rationale: if TF-IDF already covers most true matches, the noisier
    token-sort/address-prefix entries may be crowding out true matches under
    a tight cap.  This experiment quantifies the effect.
    """
    results: List[ExperimentResult] = []

    def _build_union(df_r):
        return merge_and_cap([
            _tfidf_candidates(df_r_s1, df_r, top_k=tfidf_top_k, method_tag="tfidf_name"),
            _token_sort_candidates(df_r_s1, df_r, method_tag="token_sort"),
            _address_prefix_candidates(df_r_s1, df_r, method_tag="address_prefix"),
        ], max_per_entity=global_cap)

    def _build_weighted(df_r):
        tfidf   = _tfidf_candidates(df_s1, df_r, top_k=tfidf_top_k, method_tag="tfidf_name")
        tsort   = _token_sort_candidates(df_s1, df_r, method_tag="token_sort")
        aprefix = _address_prefix_candidates(df_s1, df_r, method_tag="address_prefix")
        if not tfidf.empty:
            tfidf = tfidf.copy()
            tfidf[_SCORE] = (tfidf[_SCORE] * 2.0).clip(upper=1.0)
        return merge_and_cap([tfidf, tsort, aprefix], max_per_entity=global_cap)

    df_r_s1 = df_s1  # alias

    # UNION
    cu_s2 = _build_union(df_s2)
    cu_s3 = _build_union(df_s3)
    pr_u_s2 = evaluate_candidates(gt_records, cu_s2, "S1_S2")
    pr_u_s3 = evaluate_candidates(gt_records, cu_s3, "S1_S3")

    results.append(ExperimentResult(
        name           = "exp2a_union_baseline",
        experiment_set = 2,
        config         = {
            "combination":  "union",
            "tfidf_top_k":  tfidf_top_k,
            "global_cap":   global_cap,
        },
        s1_s2               = pr_u_s2,
        s1_s3               = pr_u_s3,
        addr_missing_s1_s2  = evaluate_address_missing_subset(gt_records, cu_s2, df_s1, "S1_S2"),
        addr_missing_s1_s3  = evaluate_address_missing_subset(gt_records, cu_s3, df_s1, "S1_S3"),
        country_s1_s2       = evaluate_by_country(gt_records, cu_s2, df_s1, "S1_S2"),
        country_s1_s3       = evaluate_by_country(gt_records, cu_s3, df_s1, "S1_S3"),
        recommendation      = "keep",
        notes               = (
            f"Baseline union 3-pass. Comparator for exp2a_weighted. "
            f"avg_recall={(pr_u_s2.recall + pr_u_s3.recall) / 2:.4f}"
        ),
    ))

    # WEIGHTED
    cw_s2 = _build_weighted(df_s2)
    cw_s3 = _build_weighted(df_s3)
    pr_w_s2 = evaluate_candidates(gt_records, cw_s2, "S1_S2")
    pr_w_s3 = evaluate_candidates(gt_records, cw_s3, "S1_S3")

    avg_union    = (pr_u_s2.recall + pr_u_s3.recall) / 2
    avg_weighted = (pr_w_s2.recall + pr_w_s3.recall) / 2
    delta        = avg_weighted - avg_union

    results.append(ExperimentResult(
        name           = "exp2a_weighted_tfidf_priority",
        experiment_set = 2,
        config         = {
            "combination":           "weighted_tfidf_2x",
            "tfidf_top_k":           tfidf_top_k,
            "global_cap":            global_cap,
            "tfidf_score_multiplier": 2.0,
        },
        s1_s2               = pr_w_s2,
        s1_s3               = pr_w_s3,
        addr_missing_s1_s2  = evaluate_address_missing_subset(gt_records, cw_s2, df_s1, "S1_S2"),
        addr_missing_s1_s3  = evaluate_address_missing_subset(gt_records, cw_s3, df_s1, "S1_S3"),
        country_s1_s2       = evaluate_by_country(gt_records, cw_s2, df_s1, "S1_S2"),
        country_s1_s3       = evaluate_by_country(gt_records, cw_s3, df_s1, "S1_S3"),
        recommendation      = "keep" if delta >= 0 else "drop",
        notes               = (
            f"TF-IDF 2x priority. avg_recall={avg_weighted:.4f}. "
            f"delta vs union={delta:+.4f}. "
            f"{'Improvement' if delta >= 0 else 'Regression'} over union baseline."
        ),
    ))

    return results


def exp2_strategy_drop_analysis(
    df_s1:       pd.DataFrame,
    df_s2:       pd.DataFrame,
    df_s3:       pd.DataFrame,
    gt_records:  List[GTRecord],
    tfidf_top_k: int = 50,
    global_cap:  int = 100,
) -> List[ExperimentResult]:
    """
    Experiment 2b -- Strategy drop analysis (unique recall contribution).

    For each of the 3 baseline strategies, measures the recall achieved when
    that strategy is dropped.  A strategy with near-zero unique contribution
    is a candidate for removal at full scale.

    Stacks tested:
    - all3              : tfidf_name + token_sort + address_prefix (full baseline)
    - drop_token_sort   : tfidf_name + address_prefix
    - drop_address      : tfidf_name + token_sort
    - drop_tfidf        : token_sort + address_prefix (diagnostic only)
    - name_only         : tfidf_name only (for address_missing path reference)
    """
    stacks = {
        "all3":            ["tfidf_name", "token_sort", "address_prefix"],
        "drop_token_sort": ["tfidf_name",               "address_prefix"],
        "drop_address":    ["tfidf_name", "token_sort"],
        "drop_tfidf":      [              "token_sort", "address_prefix"],
        "name_only":       ["tfidf_name"],
    }

    def _build(strategies, df_r):
        dfs = []
        if "tfidf_name"      in strategies:
            dfs.append(_tfidf_candidates(df_s1, df_r, top_k=tfidf_top_k,
                                         method_tag="tfidf_name"))
        if "token_sort"      in strategies:
            dfs.append(_token_sort_candidates(df_s1, df_r, method_tag="token_sort"))
        if "address_prefix"  in strategies:
            dfs.append(_address_prefix_candidates(df_s1, df_r, method_tag="address_prefix"))
        return merge_and_cap(dfs, max_per_entity=global_cap)

    results: List[ExperimentResult] = []

    for stack_name, strategies in stacks.items():
        logger.info("exp2_strategy_drop_analysis: stack=%s", stack_name)
        c_s2 = _build(strategies, df_s2)
        c_s3 = _build(strategies, df_s3)

        pr_s2 = evaluate_candidates(gt_records, c_s2, "S1_S2")
        pr_s3 = evaluate_candidates(gt_records, c_s3, "S1_S3")
        am_s2 = evaluate_address_missing_subset(gt_records, c_s2, df_s1, "S1_S2")
        am_s3 = evaluate_address_missing_subset(gt_records, c_s3, df_s1, "S1_S3")

        avg_recall = (pr_s2.recall + pr_s3.recall) / 2
        rec   = "tune-further"
        notes = (
            f"Stack={strategies}. avg_recall={avg_recall:.4f}. "
            f"avg_cands_s1s2={pr_s2.avg_candidates_per_entity:.1f}"
        )
        if stack_name.startswith("drop_"):
            notes += " -- compare recall delta vs all3 to assess unique contribution."

        results.append(ExperimentResult(
            name           = f"exp2b_{stack_name}",
            experiment_set = 2,
            config         = {
                "strategies":   strategies,
                "tfidf_top_k":  tfidf_top_k,
                "global_cap":   global_cap,
            },
            s1_s2               = pr_s2,
            s1_s3               = pr_s3,
            addr_missing_s1_s2  = am_s2,
            addr_missing_s1_s3  = am_s3,
            country_s1_s2       = evaluate_by_country(gt_records, c_s2, df_s1, "S1_S2"),
            country_s1_s3       = evaluate_by_country(gt_records, c_s3, df_s1, "S1_S3"),
            recommendation      = rec,
            notes               = notes,
        ))

    return results


# ============================================================================
# EXPERIMENT SET 3 -- New strategy candidates (exploratory)
# ============================================================================

def exp3_postal_char_ngram(
    df_s1:      pd.DataFrame,
    df_s2:      pd.DataFrame,
    df_s3:      pd.DataFrame,
    gt_records: List[GTRecord],
    top_k:      int = 50,
    global_cap: int = 100,
) -> ExperimentResult:
    """
    Experiment 3a -- Postal/numeric-token character n-gram blocking.

    Extracts digit-only tokens from normalized_address and TF-IDF-blocks on
    the resulting numeric key string.  Tests whether this is meaningfully
    complementary to Strategy G (address prefix).
    Only applied to rows with address_missing=False.
    """
    logger.info("exp3_postal_char_ngram: top_k=%d", top_k)

    cands_s1_s2 = merge_and_cap(
        [_postal_char_ngram_candidates(df_s1, df_s2, top_k=top_k)],
        max_per_entity=global_cap,
    )
    cands_s1_s3 = merge_and_cap(
        [_postal_char_ngram_candidates(df_s1, df_s3, top_k=top_k)],
        max_per_entity=global_cap,
    )

    pr_s2 = evaluate_candidates(gt_records, cands_s1_s2, "S1_S2")
    pr_s3 = evaluate_candidates(gt_records, cands_s1_s3, "S1_S3")
    am_s2 = evaluate_address_missing_subset(gt_records, cands_s1_s2, df_s1, "S1_S2")
    am_s3 = evaluate_address_missing_subset(gt_records, cands_s1_s3, df_s1, "S1_S3")

    avg_recall = (pr_s2.recall + pr_s3.recall) / 2
    rec        = "tune-further" if avg_recall > 0.1 else "drop"

    return ExperimentResult(
        name           = "exp3a_postal_char_ngram",
        experiment_set = 3,
        config         = {
            "strategy":         "postal_char_ngram",
            "top_k":            top_k,
            "global_cap":       global_cap,
            "ngram_range":      "(2,4)",
            "addr_missing":     "skipped",
        },
        s1_s2               = pr_s2,
        s1_s3               = pr_s3,
        addr_missing_s1_s2  = am_s2,
        addr_missing_s1_s3  = am_s3,
        country_s1_s2       = evaluate_by_country(gt_records, cands_s1_s2, df_s1, "S1_S2"),
        country_s1_s3       = evaluate_by_country(gt_records, cands_s1_s3, df_s1, "S1_S3"),
        recommendation      = rec,
        notes               = (
            f"Postal numeric-token char ngram. avg_recall={avg_recall:.4f}. "
            f"avg_cands_s1s2={pr_s2.avg_candidates_per_entity:.1f}. "
            f"addr_missing rows skipped."
        ),
    )


def exp3_country_scoped_tfidf(
    df_s1:      pd.DataFrame,
    df_s2:      pd.DataFrame,
    df_s3:      pd.DataFrame,
    gt_records: List[GTRecord],
    top_k:      int = 50,
    global_cap: int = 100,
) -> List[ExperimentResult]:
    """
    Experiment 3b -- Country-scoped TF-IDF (name and address variants).

    Tests two sub-variants:
    (i)  Country-scoped TF-IDF on normalized_name
    (ii) Country-scoped TF-IDF on normalized_address (address-present rows only)

    Hypothesis: restricting to same-country improves precision without hurting
    recall, since true matches are typically within the same country.
    """
    results: List[ExperimentResult] = []

    variants = [
        (_NORM_NAME, "name", df_s1, df_s2, df_s3),
    ]
    # Address variant: filter to address-present rows
    l_addr  = df_s1[df_s1[_ADDR_MISSING] == False]
    r2_addr = df_s2[df_s2[_ADDR_MISSING] == False]
    r3_addr = df_s3[df_s3[_ADDR_MISSING] == False]
    variants.append((_NORM_ADDR, "addr", l_addr, r2_addr, r3_addr))

    for col, tag, l, r2, r3 in variants:
        logger.info("exp3_country_scoped_tfidf: col=%s, top_k=%d", col, top_k)

        c_s2 = merge_and_cap(
            [_country_scoped_tfidf_candidates(l, r2, top_k=top_k, text_col=col,
                                              method_tag=f"country_tfidf_{tag}")],
            max_per_entity=global_cap,
        )
        c_s3 = merge_and_cap(
            [_country_scoped_tfidf_candidates(l, r3, top_k=top_k, text_col=col,
                                              method_tag=f"country_tfidf_{tag}")],
            max_per_entity=global_cap,
        )

        pr_s2 = evaluate_candidates(gt_records, c_s2, "S1_S2")
        pr_s3 = evaluate_candidates(gt_records, c_s3, "S1_S3")
        am_s2 = evaluate_address_missing_subset(gt_records, c_s2, df_s1, "S1_S2")
        am_s3 = evaluate_address_missing_subset(gt_records, c_s3, df_s1, "S1_S3")

        avg_recall = (pr_s2.recall + pr_s3.recall) / 2

        results.append(ExperimentResult(
            name           = f"exp3b_country_tfidf_{tag}",
            experiment_set = 3,
            config         = {
                "strategy":   f"country_scoped_tfidf_{tag}",
                "top_k":      top_k,
                "global_cap": global_cap,
                "text_col":   col,
                "addr_missing_note": ("skipped for addr variant"
                                      if col == _NORM_ADDR else "all rows included"),
            },
            s1_s2               = pr_s2,
            s1_s3               = pr_s3,
            addr_missing_s1_s2  = am_s2,
            addr_missing_s1_s3  = am_s3,
            country_s1_s2       = evaluate_by_country(gt_records, c_s2, df_s1, "S1_S2"),
            country_s1_s3       = evaluate_by_country(gt_records, c_s3, df_s1, "S1_S3"),
            recommendation      = "tune-further" if avg_recall > 0.1 else "drop",
            notes               = (
                f"Country-scoped TF-IDF on {col}. avg_recall={avg_recall:.4f}. "
                f"avg_cands_s1s2={pr_s2.avg_candidates_per_entity:.1f}"
            ),
        ))

    return results


def exp3_address_missing_subset_quality(
    df_s1:       pd.DataFrame,
    df_s2:       pd.DataFrame,
    df_s3:       pd.DataFrame,
    gt_records:  List[GTRecord],
    tfidf_top_k: int = 50,
    global_cap:  int = 100,
) -> ExperimentResult:
    """
    Experiment 3c -- Address-missing fallback quality check.

    Specifically measures whether name-only blocking achieves acceptable recall
    for the ~3.4% of S2/S3 rows with address_missing=True.

    Strategy: tfidf_name + token_sort + rare_token (no address-based strategies).
    We evaluate globally but surface the addr_missing subset recall prominently.
    """
    logger.info("exp3_address_missing_quality: name-only (tfidf+token_sort+rare_token)")

    cands_s1_s2 = merge_and_cap([
        _tfidf_candidates(df_s1, df_s2, top_k=tfidf_top_k, method_tag="tfidf_name"),
        _token_sort_candidates(df_s1, df_s2, method_tag="token_sort"),
        _rare_token_candidates(df_s1, df_s2, rarity_pct=0.005, method_tag="rare_token"),
    ], max_per_entity=global_cap)

    cands_s1_s3 = merge_and_cap([
        _tfidf_candidates(df_s1, df_s3, top_k=tfidf_top_k, method_tag="tfidf_name"),
        _token_sort_candidates(df_s1, df_s3, method_tag="token_sort"),
        _rare_token_candidates(df_s1, df_s3, rarity_pct=0.005, method_tag="rare_token"),
    ], max_per_entity=global_cap)

    pr_s2 = evaluate_candidates(gt_records, cands_s1_s2, "S1_S2")
    pr_s3 = evaluate_candidates(gt_records, cands_s1_s3, "S1_S3")
    am_s2 = evaluate_address_missing_subset(gt_records, cands_s1_s2, df_s1, "S1_S2")
    am_s3 = evaluate_address_missing_subset(gt_records, cands_s1_s3, df_s1, "S1_S3")

    am_recall_s2 = am_s2.get("recall") or 0.0
    am_recall_s3 = am_s3.get("recall") or 0.0
    avg_am_recall = (am_recall_s2 + am_recall_s3) / 2
    rec = "keep" if avg_am_recall > 0.5 else "tune-further"

    return ExperimentResult(
        name           = "exp3c_addr_missing_name_only_quality",
        experiment_set = 3,
        config         = {
            "strategy":            "tfidf_name+token_sort+rare_token",
            "tfidf_top_k":         tfidf_top_k,
            "global_cap":          global_cap,
            "rare_token_pct":      0.005,
            "focus":               "address_missing_subset_recall",
        },
        s1_s2               = pr_s2,
        s1_s3               = pr_s3,
        addr_missing_s1_s2  = am_s2,
        addr_missing_s1_s3  = am_s3,
        country_s1_s2       = evaluate_by_country(gt_records, cands_s1_s2, df_s1, "S1_S2"),
        country_s1_s3       = evaluate_by_country(gt_records, cands_s1_s3, df_s1, "S1_S3"),
        recommendation      = rec,
        notes               = (
            f"Name-only (tfidf+token_sort+rare_token) for all rows. "
            f"addr_missing recall S1xS2={am_recall_s2:.4f}, "
            f"S1xS3={am_recall_s3:.4f}. "
            f"avg_addr_missing_recall={avg_am_recall:.4f}."
        ),
    )


# ============================================================================
# Consolidated runner
# ============================================================================

def run_all_experiments(
    df_s1:      pd.DataFrame,
    df_s2:      pd.DataFrame,
    df_s3:      pd.DataFrame,
    gt_records: List[GTRecord],
    *,
    topk_sweep:   List[int]   = (10, 25, 50, 100),
    rarity_sweep: List[float] = (0.001, 0.005, 0.01),
    cap_sweep:    List[int]   = (50, 100, 200),
    tfidf_top_k:  int         = 50,
    global_cap:   int         = 100,
) -> List[ExperimentResult]:
    """
    Run all three experiment sets and return a flat list of ExperimentResult objects.

    Parameters
    ----------
    df_s1 / df_s2 / df_s3 : Normalized DataFrames from Account 1's processed/ prefix.
                             Required columns: entity_id, normalized_name,
                             normalized_address, address_missing, country,
                             token_sorted_name, source.
    gt_records             : Parsed ground truth from parse_ground_truth_df().
    topk_sweep             : K values for Experiment 1a.
    rarity_sweep           : Rarity fractions for Experiment 1b.
    cap_sweep              : Global cap values for Experiment 1c.
    tfidf_top_k            : Default K for Sets 2 and 3.
    global_cap             : Default global cap for Sets 2 and 3.

    Returns
    -------
    List[ExperimentResult]  -- all results, ordered: Set1, Set2, Set3.
    """
    all_results: List[ExperimentResult] = []

    logger.info("=" * 70)
    logger.info("EXPERIMENT SET 1 -- Parameter tuning")
    logger.info("=" * 70)
    all_results.extend(
        exp1_tfidf_topk_sweep(df_s1, df_s2, df_s3, gt_records,
                              top_k_values=topk_sweep, global_cap=global_cap))
    all_results.extend(
        exp1_rare_token_rarity_sweep(df_s1, df_s2, df_s3, gt_records,
                                     rarity_pcts=rarity_sweep, global_cap=global_cap))
    all_results.extend(
        exp1_global_cap_sweep(df_s1, df_s2, df_s3, gt_records, cap_values=cap_sweep))

    logger.info("=" * 70)
    logger.info("EXPERIMENT SET 2 -- Combination logic")
    logger.info("=" * 70)
    all_results.extend(
        exp2_union_vs_weighted(df_s1, df_s2, df_s3, gt_records,
                               tfidf_top_k=tfidf_top_k, global_cap=global_cap))
    all_results.extend(
        exp2_strategy_drop_analysis(df_s1, df_s2, df_s3, gt_records,
                                    tfidf_top_k=tfidf_top_k, global_cap=global_cap))

    logger.info("=" * 70)
    logger.info("EXPERIMENT SET 3 -- New strategies")
    logger.info("=" * 70)
    all_results.append(
        exp3_postal_char_ngram(df_s1, df_s2, df_s3, gt_records,
                               top_k=tfidf_top_k, global_cap=global_cap))
    all_results.extend(
        exp3_country_scoped_tfidf(df_s1, df_s2, df_s3, gt_records,
                                  top_k=tfidf_top_k, global_cap=global_cap))
    all_results.append(
        exp3_address_missing_subset_quality(df_s1, df_s2, df_s3, gt_records,
                                            tfidf_top_k=tfidf_top_k, global_cap=global_cap))

    logger.info("All experiments complete -- %d results total", len(all_results))
    return all_results
