"""
scripts/feature_importance_and_leakage_diagnostic.py
======================================================
Diagnostic script to investigate whether the perfect AUC is driven by
blocking-metadata leakage (blocking_score, candidate_rank, candidate_margin)
rather than genuine text-similarity learning.

Steps
-----
1. Load trained model from /tmp/model/model.txt and print gain-based feature
   importance ranked. Flag if metadata features dominate top-5.
2. Re-train with metadata features REMOVED (name/address/country only).
   Report new AUC and F0.5. If meaningful drop → metadata was the driver.
3. Compute blocking_score distribution for true positives vs hard negatives.
   Report gap and overlap. Flag if hard-negative sampling threshold is too loose.

Author: Account 3
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.entity_resolution.train import (
    FEATURE_COLS,
    LABEL_COL,
    ENTITY_COL,
    SOURCE_COL,
    entity_level_split,
    check_no_entity_leakage,
    evaluate_at_thresholds,
    evaluate_split_by_source,
    train,
    build_feature_importance,
)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
SEED = 42

# Blocking metadata features suspected of leakage
BLOCKING_META_FEATURES = [
    "blocking_score",
    "candidate_rank",
    "candidate_margin",
    # Method indicators are derived from blocking but less likely to encode the answer
    # — leave them in the stripped model for now. Only the raw score/rank leaks.
]

# Text-similarity-only features (everything except the three above)
TEXT_ONLY_FEATURES = [f for f in FEATURE_COLS if f not in BLOCKING_META_FEATURES]


# ---------------------------------------------------------------------------
# 1. Feature importance from saved model
# ---------------------------------------------------------------------------

def print_feature_importance(model_path: str) -> dict:
    print("\n" + "=" * 70)
    print("1. FEATURE IMPORTANCE (GAIN) — FULL MODEL")
    print("=" * 70)

    booster = lgb.Booster(model_file=model_path)
    gain  = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    names = booster.feature_name()

    total_gain = gain.sum() or 1.0
    rows = sorted(zip(names, gain, split), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<5} {'Feature':<32} {'Gain':>12} {'Gain%':>8} {'Split':>8}")
    print("-" * 70)
    for rank, (name, g, s) in enumerate(rows, 1):
        flag = "  *** METADATA" if name in BLOCKING_META_FEATURES else ""
        print(f"{rank:<5} {name:<32} {g:>12.1f} {g/total_gain*100:>7.1f}%  {s:>6}{flag}")

    top3_names = [r[0] for r in rows[:3]]
    top5_names = [r[0] for r in rows[:5]]
    meta_in_top3 = [n for n in top3_names if n in BLOCKING_META_FEATURES]
    meta_in_top5 = [n for n in top5_names if n in BLOCKING_META_FEATURES]
    meta_gain_pct = sum(g for n, g, _ in rows if n in BLOCKING_META_FEATURES) / total_gain * 100

    print(f"\n  Metadata features in top-3: {meta_in_top3}")
    print(f"  Metadata features in top-5: {meta_in_top5}")
    print(f"  Total gain % from blocking metadata: {meta_gain_pct:.1f}%")

    if meta_gain_pct > 50:
        print("\n  ⚠️  LEAKAGE LIKELY: blocking metadata accounts for >50% of model gain.")
        print("     The perfect AUC is almost certainly driven by blocking_score/rank,")
        print("     not text-similarity learning.")
    elif meta_gain_pct > 20:
        print("\n  ⚠️  PARTIAL LEAKAGE RISK: metadata contributes >20% of gain.")
    else:
        print("\n  ✅ Text features dominate — metadata contribution is modest.")

    return {"gain_by_feature": {n: float(g) for n, g, _ in rows},
            "meta_gain_pct": round(meta_gain_pct, 2),
            "meta_in_top3": meta_in_top3,
            "meta_in_top5": meta_in_top5}


# ---------------------------------------------------------------------------
# 2. Retrain without blocking metadata, compare AUC
# ---------------------------------------------------------------------------

def retrain_without_metadata(df_all: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("2. RETRAIN: TEXT-ONLY FEATURES (no blocking_score/rank/margin)")
    print("=" * 70)
    print(f"\n  Dropped features: {BLOCKING_META_FEATURES}")
    print(f"  Retained features ({len(TEXT_ONLY_FEATURES)}): {TEXT_ONLY_FEATURES}\n")

    df_train, df_val = entity_level_split(df_all, val_frac=0.20, seed=SEED)
    assert check_no_entity_leakage(df_train, df_val)

    # Monkey-patch: override FEATURE_COLS used by train() via a wrapper
    # train() reads FEATURE_COLS directly, so we rebuild X manually
    X_train = df_train[TEXT_ONLY_FEATURES].values.astype(np.float32)
    y_train = df_train[LABEL_COL].values.astype(int)
    X_val   = df_val[TEXT_ONLY_FEATURES].values.astype(np.float32)
    y_val   = df_val[LABEL_COL].values.astype(int)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw   = round(n_neg / max(n_pos, 1), 2)

    lgb_params = {
        "objective":         "binary",
        "metric":            "auc",
        "verbosity":         -1,
        "num_leaves":        63,
        "learning_rate":     0.05,
        "max_depth":         -1,
        "min_child_samples": 20,
        "scale_pos_weight":  spw,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "random_state":      SEED,
        "n_jobs":            -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=TEXT_ONLY_FEATURES)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, feature_name=TEXT_ONLY_FEATURES)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    booster_text = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    y_prob = booster_text.predict(X_val)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_val, y_prob))

    thr_results = evaluate_at_thresholds(y_val, y_prob, THRESHOLDS)

    # Per-source breakdown using df_val
    source_report = evaluate_split_by_source(df_val, y_prob, THRESHOLDS)

    print(f"\n  TEXT-ONLY MODEL RESULTS:")
    print(f"    Val AUC (global):   {auc:.4f}")
    print(f"    Best iteration:     {booster_text.best_iteration}")

    print(f"\n  --- F0.5 by threshold ---")
    for k, v in thr_results.items():
        print(f"    {k}: P={v['precision']}  R={v['recall']}  F0.5={v['f0.5']}")

    print(f"\n  --- By source ---")
    for src, sm in source_report.items():
        if "auc_roc" in sm:
            print(f"    [{src}] AUC={sm['auc_roc']}  n_pos={sm.get('n_positives','?')}")
            for k, v in sm.items():
                if k.startswith("thr_"):
                    print(f"      {k}: P={v['precision']}  R={v['recall']}  F0.5={v['f0.5']}")

    # Feature importance for text-only model
    gain  = booster_text.feature_importance(importance_type="gain")
    total = gain.sum() or 1
    imp_rows = sorted(zip(TEXT_ONLY_FEATURES, gain), key=lambda x: x[1], reverse=True)
    print(f"\n  Top-10 text features by gain:")
    for name, g in imp_rows[:10]:
        print(f"    {name:<35} {g/total*100:>6.1f}%")

    # Save text-only model
    booster_text.save_model("/tmp/model/model_text_only.txt")
    print("\n  Text-only model saved to /tmp/model/model_text_only.txt")

    return {
        "auc": round(auc, 4),
        "best_iteration": booster_text.best_iteration,
        "threshold_results": thr_results,
        "by_source": source_report,
    }


# ---------------------------------------------------------------------------
# 3. Blocking score distribution: TPs vs Hard Negatives
# ---------------------------------------------------------------------------

def score_distribution_analysis(df_all: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("3. BLOCKING_SCORE DISTRIBUTION: TRUE POSITIVES vs HARD NEGATIVES")
    print("=" * 70)

    tp   = df_all[df_all["negative_type"] == "positive"]["blocking_score"]
    hard = df_all[df_all["negative_type"] == "hard_negative"]["blocking_score"]
    easy = df_all[df_all["negative_type"] == "easy_negative"]["blocking_score"]
    sing = df_all[df_all["negative_type"] == "singleton_negative"]["blocking_score"]

    def stats(s: pd.Series, name: str) -> dict:
        if len(s) == 0:
            print(f"\n  {name}: (empty)")
            return {}
        p = dict(
            n     = len(s),
            min   = round(float(s.min()),  4),
            p5    = round(float(s.quantile(0.05)), 4),
            p25   = round(float(s.quantile(0.25)), 4),
            median= round(float(s.median()),4),
            p75   = round(float(s.quantile(0.75)), 4),
            p95   = round(float(s.quantile(0.95)), 4),
            max   = round(float(s.max()),  4),
            mean  = round(float(s.mean()), 4),
        )
        print(f"\n  {name} (n={p['n']:,}):")
        print(f"    min={p['min']:.3f}  p5={p['p5']:.3f}  p25={p['p25']:.3f}  "
              f"median={p['median']:.3f}  p75={p['p75']:.3f}  p95={p['p95']:.3f}  max={p['max']:.3f}")
        return p

    tp_stats   = stats(tp,   "True Positives")
    hard_stats = stats(hard, "Hard Negatives")
    easy_stats = stats(easy, "Easy Negatives")
    sing_stats = stats(sing, "Singleton Negatives")

    # Overlap analysis
    print("\n  --- Overlap Analysis: True Positives vs Hard Negatives ---")
    if tp_stats and hard_stats:
        tp_min    = tp_stats["min"]
        tp_max    = tp_stats["max"]
        hard_min  = hard_stats["min"]
        hard_max  = hard_stats["max"]

        # What fraction of hard negatives exceed the minimum TP score?
        hard_above_tp_min   = (hard >= tp_min).mean() * 100
        # What fraction of TPs are below the hard-neg median?
        tp_below_hard_med   = (tp < hard_stats["median"]).mean() * 100
        # Score range overlap: [max(tp_min,hard_min), min(tp_max,hard_max)]
        overlap_lo = max(tp_min, hard_min)
        overlap_hi = min(tp_max, hard_max)
        has_overlap = overlap_lo <= overlap_hi

        print(f"    TP score range:         [{tp_min:.3f}, {tp_max:.3f}]")
        print(f"    Hard-neg score range:   [{hard_min:.3f}, {hard_max:.3f}]")
        print(f"    Score ranges overlap:   {'YES' if has_overlap else 'NO'} "
              f"[{overlap_lo:.3f}, {overlap_hi:.3f}]")
        print(f"    Hard negs >= TP min:    {hard_above_tp_min:.1f}%")
        print(f"    TPs below hard-neg med: {tp_below_hard_med:.1f}%")

        # Per-entity: for each S1 entity, what's the gap between its TP score
        # and its hardest negative score?
        if "source1_entity_id" in df_all.columns:
            tp_df   = df_all[df_all["negative_type"] == "positive"][["source1_entity_id","blocking_score"]]
            hard_df = df_all[df_all["negative_type"] == "hard_negative"][["source1_entity_id","blocking_score"]]
            if not tp_df.empty and not hard_df.empty:
                tp_max_per  = tp_df.groupby("source1_entity_id")["blocking_score"].max().rename("tp_max")
                hard_max_per= hard_df.groupby("source1_entity_id")["blocking_score"].max().rename("hard_max")
                per_entity  = pd.concat([tp_max_per, hard_max_per], axis=1).dropna()
                per_entity["gap"] = per_entity["tp_max"] - per_entity["hard_max"]
                n_hard_beats_tp = (per_entity["gap"] < 0).sum()
                pct_hard_beats  = n_hard_beats_tp / len(per_entity) * 100

                gap_med = per_entity["gap"].median()
                gap_p5  = per_entity["gap"].quantile(0.05)

                print(f"\n    Per-entity (entities with both TP and hard neg, n={len(per_entity)}):")
                print(f"      TP max > hard-neg max:       {(per_entity['gap']>0).sum()} / {len(per_entity)} entities")
                print(f"      Hard-neg beats TP:           {n_hard_beats_tp} ({pct_hard_beats:.1f}%) entities")
                print(f"      Median gap (TP-max - HN-max):{gap_med:.4f}")
                print(f"      p5 gap:                      {gap_p5:.4f}")

                if pct_hard_beats > 20:
                    print("\n    ⚠️  >20% of entities have a hard negative that SCORES HIGHER than")
                    print("       their true positive. The current hard-neg sampling (70th pct of")
                    print("       entity negatives) is too loose — hard negs are not truly 'hard'")
                    print("       relative to the TP. Recommend: sample negatives within X% of the")
                    print("       entity's max TP blocking_score instead of a global percentile cut.")
                elif gap_med < 0.05:
                    print("\n    ⚠️  Median gap < 0.05: hard negatives are very close in score to TPs.")
                    print("       This is good for learning — but verify the model isn't just")
                    print("       memorizing the score boundary.")
                else:
                    print("\n    ✅ Good separation: TP scores consistently above hard-negative scores.")

        if has_overlap:
            print(f"\n    ⚠️  Score ranges overlap in [{overlap_lo:.3f}, {overlap_hi:.3f}].")
            print(f"       The model CANNOT perfectly separate TPs from hard-negs on blocking_score alone.")
            print(f"       → If AUC drops significantly when blocking_score is removed, the model was")
            print(f"         exploiting the non-overlapping portion of the range — not text similarity.")
        else:
            print(f"\n    ⚠️  Score ranges DO NOT overlap at all. TP scores are always "
                  f"{'higher' if tp_min > hard_max else 'lower'} than hard-neg scores.")
            print(f"       This means a single threshold on blocking_score can perfectly classify TPs.")
            print(f"       The model has trivial leakage — text features contribute nothing.")

    return {
        "true_positives":      tp_stats,
        "hard_negatives":      hard_stats,
        "easy_negatives":      easy_stats,
        "singleton_negatives": sing_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load feature files
    print("Loading feature files from /tmp/ ...")
    feat_s1_s2 = pd.read_parquet("/tmp/features_labeled_S1_S2.parquet")
    feat_s1_s3 = pd.read_parquet("/tmp/features_labeled_S1_S3.parquet")
    df_all = pd.concat([feat_s1_s2, feat_s1_s3], ignore_index=True)
    print(f"Combined: {len(df_all):,} rows | {df_all[LABEL_COL].sum():,} positives")

    # 1. Feature importance from saved model
    imp = print_feature_importance("/tmp/model/model.txt")

    # 2. Score distribution analysis (uses negative_type column from labeling)
    score_dist = score_distribution_analysis(df_all)

    # 3. Text-only retrain
    text_results = retrain_without_metadata(df_all)

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    meta_pct = imp["meta_gain_pct"]
    text_auc = text_results["auc"]

    print(f"\n  Full model AUC:       1.0000")
    print(f"  Text-only model AUC:  {text_auc:.4f}")
    print(f"  AUC drop:             {1.0 - text_auc:.4f}")
    print(f"  Blocking meta gain%:  {meta_pct:.1f}%")

    if text_auc >= 0.95 and meta_pct < 50:
        print("\n  ✅ RECOMMENDATION: Text features are doing real work.")
        print("     The model generalises beyond blocking metadata.")
        print("     Submit the FULL MODEL to SageMaker (metadata is complementary, not cheating).")
    elif text_auc >= 0.85:
        print("\n  ⚠️  RECOMMENDATION: Moderate metadata contribution.")
        print("     Submit BOTH models. Use text-only for production (conservative),")
        print("     full model for recall-maximisation experiments.")
    else:
        print("\n  🔴 RECOMMENDATION: Significant leakage from blocking metadata.")
        print("     Submit TEXT-ONLY MODEL to SageMaker.")
        print("     The full model's AUC is not trustworthy at full scale.")

    # Save full report
    report = {
        "feature_importance":  imp,
        "score_distribution":  {k: v for k, v in score_dist.items()},
        "full_model_auc":      1.0,
        "text_only_model_auc": text_auc,
        "auc_drop":            round(1.0 - text_auc, 4),
        "meta_gain_pct":       meta_pct,
    }
    with open("/tmp/leakage_diagnostic_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n  Full report saved to /tmp/leakage_diagnostic_report.json")


if __name__ == "__main__":
    main()
