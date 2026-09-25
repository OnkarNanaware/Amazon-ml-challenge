"""
train.py
========
Task C — SageMaker training script for entity-resolution binary classifier.

Design contract
---------------
* Accepts feature parquets from SageMaker input channels (SM_CHANNEL_TRAIN,
  SM_CHANNEL_VALIDATION) or via --train-data / --val-data argparse flags.
* Supports both features_labeled_S1_S2.parquet and features_labeled_S1_S3.parquet
  as separate input channels; controlled by --pair-type (s1_s2 | s1_s3 | both).
  When "both", concatenates with candidate_source_s2/candidate_source_s3 retained
  as features so the model can learn source-specific patterns.
* Entity-level train/val split by source1_entity_id (NOT by row) — no leakage.
  Fixed seed, configurable 80/20 ratio.
* LightGBM binary classifier. scale_pos_weight defaults to the actual imbalance
  ratio found in the training data — never guessed.
* Early stopping on validation AUC.
* Evaluates F0.5 at thresholds [0.5, 0.6, 0.7, 0.8], reported separately for
  S1-S2 and S1-S3 rows within the validation set.
* Saves model to /opt/ml/model/ as LightGBM Booster (.txt) + feature importance.
* All paths via SageMaker env vars or argparse — no hardcoded strings.
* Logs to stdout for CloudWatch.

Usage (local)
-------------
    python src/entity_resolution/train.py \\
        --train-data /tmp/features_labeled_S1_S2.parquet \\
                     /tmp/features_labeled_S1_S3.parquet \\
        --model-dir /tmp/model/ \\
        --pair-type both

Authors: Account 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column definitions (must match features.py _FEATURE_NAMES)
# ---------------------------------------------------------------------------
FEATURE_COLS: List[str] = [
    "name_exact", "name_levenshtein", "name_jaro_winkler", "name_token_jaccard",
    "name_token_sort_ratio", "name_ngram_cosine", "name_length_difference",
    "address_exact", "address_jaccard", "address_ngram_similarity",
    "address_edit_distance", "numeric_token_similarity", "postal_similarity",
    "address_length_difference",
    "country_exact", "country_normalized", "country_unseen",
    "blocking_score", "blocking_method_tfidf", "blocking_method_token_sort",
    "blocking_method_addr_prefix", "candidate_rank", "candidate_margin",
    "candidate_source_s2", "candidate_source_s3",
    "address_missing_left", "address_missing_right", "address_missing_either",
]

LABEL_COL    = "label"
ENTITY_COL   = "source1_entity_id"
SOURCE_COL   = "candidate_source"   # 's2' or 's3'


# ---------------------------------------------------------------------------
# Entity-level train/val split — core anti-leakage guarantee
# ---------------------------------------------------------------------------

def entity_level_split(
    df: pd.DataFrame,
    val_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/val BY source1_entity_id (not by row).

    This prevents any entity that appears in the training set from leaking
    into the validation set, which would inflate metrics for a model that
    simply memorises entity embeddings.

    Parameters
    ----------
    df       : DataFrame with ENTITY_COL column.
    val_frac : Fraction of unique entities to hold out for validation.
    seed     : Random seed for reproducibility.

    Returns
    -------
    (df_train, df_val) — two DataFrames with no shared source1_entity_id values.
    """
    rng = np.random.default_rng(seed)
    all_entities = df[ENTITY_COL].unique()
    rng.shuffle(all_entities)

    n_val = max(1, int(len(all_entities) * val_frac))
    val_entities  = set(all_entities[:n_val])
    train_entities = set(all_entities[n_val:])

    df_val   = df[df[ENTITY_COL].isin(val_entities)].reset_index(drop=True)
    df_train = df[df[ENTITY_COL].isin(train_entities)].reset_index(drop=True)

    logger.info(
        "Entity-level split: %d train entities (%d rows) | %d val entities (%d rows)",
        len(train_entities), len(df_train), len(val_entities), len(df_val),
    )
    return df_train, df_val


def check_no_entity_leakage(df_train: pd.DataFrame, df_val: pd.DataFrame) -> bool:
    """Assert no entity appears in both train and val. Returns True if clean."""
    overlap = set(df_train[ENTITY_COL].unique()) & set(df_val[ENTITY_COL].unique())
    if overlap:
        logger.error("LEAKAGE DETECTED: %d entities in both train and val!", len(overlap))
        return False
    logger.info("Leakage check PASSED: zero entity overlap between train and val.")
    return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_feature_data(
    paths: List[str],
    pair_type: str = "both",
) -> pd.DataFrame:
    """
    Load one or more feature parquets and concatenate.

    Parameters
    ----------
    paths     : list of local file paths to feature parquets.
    pair_type : 's1_s2', 's1_s3', or 'both' — filters or combines pair types.

    Returns
    -------
    pd.DataFrame with FEATURE_COLS + LABEL_COL + ENTITY_COL + SOURCE_COL columns.
    """
    dfs: List[pd.DataFrame] = []
    for p in paths:
        logger.info("Loading feature file: %s", p)
        df = pd.read_parquet(p)

        # Validate required columns present
        missing = [c for c in FEATURE_COLS + [LABEL_COL, ENTITY_COL]
                   if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {p}: {missing}")

        if pair_type == "s1_s2" and SOURCE_COL in df.columns:
            df = df[df[SOURCE_COL] == "s2"]
        elif pair_type == "s1_s3" and SOURCE_COL in df.columns:
            df = df[df[SOURCE_COL] == "s3"]
        # "both" → keep all

        logger.info("  Loaded %d rows, %d positives (%.1f%%)",
                    len(df), df[LABEL_COL].sum(),
                    df[LABEL_COL].mean() * 100)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Combined: %d rows total, %d positives (%.1f%%)",
                len(combined), combined[LABEL_COL].sum(),
                combined[LABEL_COL].mean() * 100)
    return combined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def fbeta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """F-beta score. F0.5 weights precision 2× more than recall."""
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def evaluate_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
) -> Dict[str, Any]:
    """Evaluate P/R/F0.5 at multiple probability thresholds."""
    results: Dict[str, Any] = {}
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        f05 = fbeta_score(p, r, beta=0.5)
        results[f"thr_{thr}"] = {
            "threshold":  thr,
            "precision":  round(p,   4),
            "recall":     round(r,   4),
            "f0.5":       round(f05, 4),
            "n_predicted_positive": int(y_pred.sum()),
        }
    return results


def evaluate_split_by_source(
    df_val: pd.DataFrame,
    y_prob: np.ndarray,
    thresholds: List[float],
) -> Dict[str, Any]:
    """
    Evaluate separately for S1-S2 and S1-S3 rows in the validation set.
    Returns a nested dict: {source: {metric: value}}.
    """
    report: Dict[str, Any] = {}
    df_val = df_val.reset_index(drop=True)

    for source_val, source_label in [("s2", "S1_S2"), ("s3", "S1_S3"), (None, "ALL")]:
        if source_val is not None and SOURCE_COL in df_val.columns:
            mask = df_val[SOURCE_COL] == source_val
        else:
            mask = pd.Series([True] * len(df_val))

        sub_true = df_val.loc[mask, LABEL_COL].values
        sub_prob = y_prob[mask.values]

        if len(sub_true) == 0 or sub_true.sum() == 0:
            report[source_label] = {"n_pairs": int(mask.sum()), "note": "no positives"}
            continue

        try:
            auc = float(roc_auc_score(sub_true, sub_prob))
        except Exception:
            auc = float("nan")

        thr_results = evaluate_at_thresholds(sub_true, sub_prob, thresholds)
        report[source_label] = {
            "n_pairs":    int(mask.sum()),
            "n_positives": int(sub_true.sum()),
            "auc_roc":    round(auc, 4),
            **thr_results,
        }

    return report


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    hyperparams: Dict[str, Any],
    thresholds:  List[float],
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    Train a LightGBM binary classifier with early stopping on validation AUC.

    Parameters
    ----------
    df_train / df_val : DataFrames with FEATURE_COLS + LABEL_COL.
    hyperparams       : LightGBM hyperparameters dict.
    thresholds        : probability thresholds to evaluate F0.5 at.

    Returns
    -------
    (booster, metrics_report)
    """
    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    y_train = df_train[LABEL_COL].values.astype(int)
    X_val   = df_val[FEATURE_COLS].values.astype(np.float32)
    y_val   = df_val[LABEL_COL].values.astype(int)

    # Compute scale_pos_weight from actual class imbalance if not overridden
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    default_spw = round(n_neg / max(n_pos, 1), 2)
    hyperparams.setdefault("scale_pos_weight", default_spw)

    logger.info("Class imbalance: %d pos / %d neg  → scale_pos_weight=%.2f",
                n_pos, n_neg, hyperparams["scale_pos_weight"])

    lgb_params = {
        "objective":        "binary",
        "metric":           "auc",
        "verbosity":        -1,
        "num_leaves":       hyperparams.get("num_leaves",         63),
        "learning_rate":    hyperparams.get("learning_rate",      0.05),
        "n_estimators":     hyperparams.get("n_estimators",       500),
        "max_depth":        hyperparams.get("max_depth",          -1),
        "min_child_samples":hyperparams.get("min_child_samples",  20),
        "scale_pos_weight": hyperparams["scale_pos_weight"],
        "subsample":        hyperparams.get("subsample",          0.8),
        "colsample_bytree": hyperparams.get("colsample_bytree",   0.8),
        "random_state":     hyperparams.get("seed",               42),
        "n_jobs":           -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, feature_name=FEATURE_COLS)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    booster = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=lgb_params.pop("n_estimators"),
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    y_prob = booster.predict(X_val)
    source_report = evaluate_split_by_source(df_val, y_prob, thresholds)

    # Global AUC
    try:
        global_auc = float(roc_auc_score(y_val, y_prob))
    except Exception:
        global_auc = float("nan")

    metrics = {
        "best_iteration":        booster.best_iteration,
        "val_auc_global":        round(global_auc, 4),
        "n_train":               len(df_train),
        "n_val":                 len(df_val),
        "n_train_pos":           n_pos,
        "n_val_pos":             int(y_val.sum()),
        "scale_pos_weight_used": hyperparams["scale_pos_weight"],
        "by_source":             source_report,
    }

    return booster, metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def build_feature_importance(booster: lgb.Booster) -> Dict[str, Any]:
    """Return feature importance dict sorted by gain."""
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    names = booster.feature_name()
    sorted_idx = np.argsort(gain)[::-1]
    return {
        "by_gain":  {names[i]: float(gain[i])  for i in sorted_idx},
        "by_split": {names[i]: float(split[i]) for i in sorted_idx},
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _resolve_sagemaker_paths() -> Tuple[str, str]:
    """Resolve SageMaker env-var based paths for model dir and data dir."""
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Task C — LightGBM training for entity resolution.")

    # ---- Data inputs --------------------------------------------------------
    parser.add_argument(
        "--train-data", nargs="+", default=None,
        help="Local path(s) to feature parquet(s). Also reads SM_CHANNEL_TRAIN.",
    )
    parser.add_argument(
        "--val-data", nargs="+", default=None,
        help="Local path(s) to val feature parquet(s). If omitted, entity-level split is used.",
    )
    parser.add_argument(
        "--pair-type", choices=["s1_s2", "s1_s3", "both"], default="both",
        help="Which pair type to train on (default: both).",
    )

    # ---- Output ---------------------------------------------------------------
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory to save model artefacts.",
    )

    # ---- Split config ---------------------------------------------------------
    parser.add_argument("--val-frac",  type=float, default=0.20, help="Validation fraction.")
    parser.add_argument("--seed",      type=int,   default=42)

    # ---- Hyperparameters ------------------------------------------------------
    parser.add_argument("--num-leaves",         type=int,   default=63)
    parser.add_argument("--learning-rate",       type=float, default=0.05)
    parser.add_argument("--n-estimators",        type=int,   default=500)
    parser.add_argument("--max-depth",           type=int,   default=-1)
    parser.add_argument("--min-child-samples",   type=int,   default=20)
    parser.add_argument("--scale-pos-weight",    type=float, default=None,
                        help="Overrides auto-computed class-imbalance weight.")
    parser.add_argument("--subsample",           type=float, default=0.8)
    parser.add_argument("--colsample-bytree",    type=float, default=0.8)

    # ---- Evaluation thresholds ------------------------------------------------
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.5, 0.6, 0.7, 0.8],
        help="Probability thresholds to evaluate F0.5 at.",
    )
    args = parser.parse_args()

    # ---- Resolve data paths ---------------------------------------------------
    # SageMaker provides data via SM_CHANNEL_TRAIN env var as a directory
    sm_train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    sm_val_dir   = os.environ.get("SM_CHANNEL_VALIDATION")

    train_paths: List[str] = []
    if args.train_data:
        train_paths = args.train_data
    elif sm_train_dir:
        train_paths = [
            str(p) for p in Path(sm_train_dir).glob("*.parquet")
        ]
    else:
        raise ValueError("Provide --train-data or set SM_CHANNEL_TRAIN env var.")

    val_paths: Optional[List[str]] = None
    if args.val_data:
        val_paths = args.val_data
    elif sm_val_dir:
        val_paths = [str(p) for p in Path(sm_val_dir).glob("*.parquet")]

    # ---- Load data ------------------------------------------------------------
    df_all = load_feature_data(train_paths, pair_type=args.pair_type)

    if val_paths:
        df_val_external = load_feature_data(val_paths, pair_type=args.pair_type)
        df_train = df_all
        df_val   = df_val_external
        logger.info("Using externally provided validation set.")
    else:
        logger.info("No separate val set; performing entity-level split ...")
        df_train, df_val = entity_level_split(df_all, val_frac=args.val_frac, seed=args.seed)

    # Anti-leakage assertion
    assert check_no_entity_leakage(df_train, df_val), "Entity leakage detected — aborting."

    # ---- Hyperparams ----------------------------------------------------------
    hyperparams: Dict[str, Any] = {
        "num_leaves":         args.num_leaves,
        "learning_rate":      args.learning_rate,
        "n_estimators":       args.n_estimators,
        "max_depth":          args.max_depth,
        "min_child_samples":  args.min_child_samples,
        "subsample":          args.subsample,
        "colsample_bytree":   args.colsample_bytree,
        "seed":               args.seed,
    }
    if args.scale_pos_weight is not None:
        hyperparams["scale_pos_weight"] = args.scale_pos_weight

    # ---- Train ----------------------------------------------------------------
    logger.info("Starting LightGBM training ...")
    booster, metrics = train(df_train, df_val, hyperparams, args.thresholds)

    # ---- Feature importance ---------------------------------------------------
    importance = build_feature_importance(booster)
    metrics["feature_importance"] = importance

    # ---- Save artefacts -------------------------------------------------------
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.txt"
    booster.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    importance_path = model_dir / "feature_importance.json"
    with open(importance_path, "w") as fh:
        json.dump(importance, fh, indent=2)
    logger.info("Feature importance saved to %s", importance_path)

    # ---- Final report ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK C — TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best iteration: {metrics['best_iteration']}")
    print(f"Val AUC (global): {metrics['val_auc_global']}")
    print("\n--- Evaluation by pair source ---")
    for src, src_metrics in metrics["by_source"].items():
        print(f"\n  [{src}]  n_pairs={src_metrics.get('n_pairs')}  "
              f"n_pos={src_metrics.get('n_positives', 'N/A')}  "
              f"AUC={src_metrics.get('auc_roc', 'N/A')}")
        for k, v in src_metrics.items():
            if k.startswith("thr_"):
                print(f"    {k}: P={v['precision']}  R={v['recall']}  F0.5={v['f0.5']}")
    print()


if __name__ == "__main__":
    main()
