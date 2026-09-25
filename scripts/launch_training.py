"""
launch_training.py
==================
Task D — SageMaker Estimator launcher for entity-resolution training.

Design contract
---------------
* Wraps train.py in a SageMaker SKLearn Estimator.
* Input channels point at features_labeled_S1_S2.parquet and
  features_labeled_S1_S3.parquet in the account3 FEATURES S3 prefix.
* output_path points at account3 MODELS S3 prefix.
* Hyperparameters are fully overridable via CLI.
* --dry-run: prints full Estimator config without launching anything.
* --local:   uses SageMaker local mode (Docker) for fast iteration.
* Default scale_pos_weight is derived from the actual class-imbalance
  ratio found by Task A — stored in labeling_stats.json.

Usage
-----
    python scripts/launch_training.py --dry-run
    python scripts/launch_training.py --local
    python scripts/launch_training.py   # real job (requires approval)

Authors: Account 3
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entity_resolution.config import (
    BUCKET,
    add_account_prefix_arg,
    resolve_config_from_args,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SageMaker container / framework version
# ---------------------------------------------------------------------------
# SKLearn container 1.2-1 ships Python 3.10 and allows requirements.txt installs.
# LightGBM is NOT in the base SKLearn container, so requirements.txt is essential.
_FRAMEWORK_VERSION = "1.2-1"
_PY_VERSION        = "py3"

# Default instance for sample-scale runs — 4 vCPUs, 16 GiB RAM, cost-effective.
_DEFAULT_INSTANCE  = "ml.m5.xlarge"

AWS_PROFILE = "amazon-ml-account3"


# ---------------------------------------------------------------------------
# Resolve scale_pos_weight from Task A's labeling stats
# ---------------------------------------------------------------------------

def _fetch_labeling_stats(
    s3_uri: str,
    aws_profile: str,
) -> Optional[Dict[str, Any]]:
    """
    Download labeling_stats.json from S3 using AWS CLI.
    Returns parsed dict or None if not found.
    """
    cmd = ["aws", "s3", "cp", s3_uri, "-", "--profile", aws_profile]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        logger.warning("Could not fetch labeling stats from %s: %s", s3_uri, e)
    return None


def _default_scale_pos_weight(
    labeling_stats: Optional[Dict[str, Any]],
    pair_type: str,
) -> float:
    """
    Derive scale_pos_weight from labeling_stats.json.
    When pair_type='both', use the average of S1_S2 and S1_S3 ratios.
    """
    if labeling_stats is None:
        logger.warning("No labeling stats available — using default scale_pos_weight=5.0")
        return 5.0

    ratios = []
    for key in ("S1_S2", "S1_S3"):
        stats = labeling_stats.get(key, {})
        r = stats.get("imbalance_ratio_neg_per_pos")
        if r is not None:
            ratios.append(float(r))

    if pair_type == "s1_s2" and labeling_stats.get("S1_S2"):
        r = labeling_stats["S1_S2"].get("imbalance_ratio_neg_per_pos")
        return float(r) if r else 5.0
    if pair_type == "s1_s3" and labeling_stats.get("S1_S3"):
        r = labeling_stats["S1_S3"].get("imbalance_ratio_neg_per_pos")
        return float(r) if r else 5.0

    if ratios:
        avg = sum(ratios) / len(ratios)
        logger.info("Derived scale_pos_weight=%.2f from labeling stats (avg of %s)", avg, ratios)
        return round(avg, 2)

    logger.warning("Could not derive scale_pos_weight from labeling stats — using 5.0")
    return 5.0


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_estimator_config(
    paths,
    aws_profile: str,
    aws_region: str,
    pair_type: str,
    instance_type: str,
    instance_count: int,
    hyperparams: Dict[str, Any],
    local: bool,
) -> Dict[str, Any]:
    """
    Build the full Estimator configuration dict (for dry-run display or real launch).
    """
    features_s1_s2 = f"{paths.FEATURES}features_labeled_S1_S2.parquet"
    features_s1_s3 = f"{paths.FEATURES}features_labeled_S1_S3.parquet"
    output_path    = paths.MODELS
    role_arn       = f"arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole"  # to be filled

    # Requirements file path
    requirements_path = str(REPO_ROOT / "requirements-sagemaker.txt")

    config = {
        "estimator_type":  "SKLearn",
        "framework_version": _FRAMEWORK_VERSION,
        "py_version":      _PY_VERSION,
        "entry_point":     "src/entity_resolution/train.py",
        "source_dir":      str(REPO_ROOT),
        "dependencies":    [requirements_path] if Path(requirements_path).exists() else [],
        "role":            role_arn,
        "instance_type":   "local" if local else instance_type,
        "instance_count":  instance_count,
        "output_path":     output_path,
        "base_job_name":   "entity-resolution-lgbm",
        "sagemaker_session_kwargs": {
            "boto_session_profile": aws_profile,
            "region_name":          aws_region,
        },
        "hyperparameters": {
            "pair-type":         pair_type,
            **hyperparams,
        },
        "input_channels": {
            "train": {
                "S3Input_S1_S2": features_s1_s2,
                "S3Input_S1_S3": features_s1_s3,
            },
        },
        "metric_definitions": [
            {"Name": "val:auc",    "Regex": r"val.*auc: ([\d\.]+)"},
            {"Name": "best_iter",  "Regex": r"Best iteration: (\d+)"},
        ],
        "enable_sagemaker_metrics": True,
    }
    return config


# ---------------------------------------------------------------------------
# Actual SageMaker launch
# ---------------------------------------------------------------------------

def launch_job(config: Dict[str, Any], aws_region: str, aws_profile: str) -> None:
    """
    Launch a real SageMaker training job using the sagemaker SDK.
    boto3 / sagemaker SDK are required on the LAUNCHER machine (not the container).
    """
    try:
        import boto3
        import sagemaker
        from sagemaker.sklearn import SKLearn
        from sagemaker.inputs import TrainingInput
    except ImportError as e:
        raise ImportError(
            "sagemaker and boto3 are required on the LAUNCHER machine to submit jobs. "
            "Install via: pip install sagemaker boto3"
        ) from e

    session_kwargs = config.get("sagemaker_session_kwargs", {})
    boto_sess = boto3.Session(
        profile_name=session_kwargs.get("boto_session_profile"),
        region_name=session_kwargs.get("region_name", aws_region),
    )
    sm_session = sagemaker.Session(boto_session=boto_sess)

    estimator = SKLearn(
        entry_point     = config["entry_point"],
        source_dir      = config["source_dir"],
        framework_version = config["framework_version"],
        py_version      = config["py_version"],
        role            = config["role"],
        instance_type   = config["instance_type"],
        instance_count  = config["instance_count"],
        output_path     = config["output_path"],
        base_job_name   = config["base_job_name"],
        hyperparameters = config["hyperparameters"],
        metric_definitions = config.get("metric_definitions", []),
        sagemaker_session  = sm_session,
        dependencies    = config.get("dependencies", []),
    )

    channels = config.get("input_channels", {}).get("train", {})
    train_inputs = {
        "train": TrainingInput(list(channels.values())[0])
        if len(channels) == 1
        else sagemaker.inputs.FileSystemInput(  # fallback
            file_system_id="",
            file_system_type="",
            directory_path="",
        )
    }

    # For two separate parquets, pass both as a comma-list via HP
    estimator.hyperparameters["train-data"] = " ".join(channels.values())

    logger.info("Submitting training job ...")
    estimator.fit(wait=False)
    logger.info("Job submitted: %s", estimator.latest_training_job.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task D — SageMaker launcher for entity-resolution LightGBM training."
    )
    add_account_prefix_arg(parser)
    parser.add_argument("--aws-profile",    default="amazon-ml-account3")
    parser.add_argument("--aws-region",     default="us-east-1")

    # Job mode
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print full Estimator config without launching.")
    parser.add_argument("--local",    action="store_true",
                        help="Use SageMaker local mode (Docker) for fast iteration.")

    # Instance
    parser.add_argument("--instance-type",  default=_DEFAULT_INSTANCE)
    parser.add_argument("--instance-count", type=int, default=1)

    # Pair type
    parser.add_argument("--pair-type", choices=["s1_s2", "s1_s3", "both"], default="both")

    # Hyperparameters (overridable)
    parser.add_argument("--num-leaves",         type=int,   default=63)
    parser.add_argument("--learning-rate",       type=float, default=0.05)
    parser.add_argument("--n-estimators",        type=int,   default=500)
    parser.add_argument("--max-depth",           type=int,   default=-1)
    parser.add_argument("--min-child-samples",   type=int,   default=20)
    parser.add_argument("--scale-pos-weight",    type=float, default=None,
                        help="Override auto-derived class-imbalance weight.")
    parser.add_argument("--subsample",           type=float, default=0.8)
    parser.add_argument("--colsample-bytree",    type=float, default=0.8)
    parser.add_argument("--val-frac",            type=float, default=0.20)
    parser.add_argument("--seed",                type=int,   default=42)
    args = parser.parse_args()

    cfg = resolve_config_from_args(args)
    paths = cfg.paths

    # Fetch labeling stats to derive scale_pos_weight
    stats_uri = f"{paths.REPORTS}labeling_stats.json"
    labeling_stats = _fetch_labeling_stats(stats_uri, aws_profile=cfg.aws_profile)

    spw = args.scale_pos_weight or _default_scale_pos_weight(labeling_stats, args.pair_type)
    logger.info("Using scale_pos_weight=%.2f for pair_type=%s", spw, args.pair_type)

    hyperparams: Dict[str, Any] = {
        "num-leaves":         args.num_leaves,
        "learning-rate":      args.learning_rate,
        "n-estimators":       args.n_estimators,
        "max-depth":          args.max_depth,
        "min-child-samples":  args.min_child_samples,
        "scale-pos-weight":   spw,
        "subsample":          args.subsample,
        "colsample-bytree":   args.colsample_bytree,
        "val-frac":           args.val_frac,
        "seed":               args.seed,
    }

    config = build_estimator_config(
        paths          = paths,
        aws_profile    = cfg.aws_profile,
        aws_region     = args.aws_region,
        pair_type      = args.pair_type,
        instance_type  = args.instance_type,
        instance_count = args.instance_count,
        hyperparams    = hyperparams,
        local          = args.local,
    )

    print("\n" + "=" * 70)
    print("TASK D — SAGEMAKER ESTIMATOR CONFIGURATION")
    print("=" * 70)
    print(json.dumps(config, indent=2, default=str))

    if args.dry_run:
        print("\n[DRY RUN] — Estimator config printed above. No job submitted.")
        print("Run without --dry-run to submit (after approval).")
        return

    if args.local:
        logger.info("Local mode: submitting via SageMaker local mode (Docker required).")

    print("\n⚠  About to submit a real SageMaker training job.")
    print("   This will incur AWS costs. Type 'yes' to confirm:")
    confirm = input().strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    launch_job(config, aws_region=args.aws_region, aws_profile=cfg.aws_profile)


if __name__ == "__main__":
    main()
