"""
config.py
=========
Centralized S3 path & pipeline configuration for the entity-resolution pipeline.

Design contract
---------------
* RAW and SHARED are NOT account-prefixed  — everyone reads the same raw data.
* Every other path IS account-prefixed, defaulting to "account1" but fully
  overridable via the ACCOUNT_PREFIX environment variable or --account-prefix
  CLI flag so that Account 2 / Account 3 can run this exact codebase against
  their own S3 prefix without touching any source files.

Usage
-----
    from src.entity_resolution.config import Paths, load_pipeline_config

    cfg = load_pipeline_config()          # honours ACCOUNT_PREFIX env var
    print(cfg.paths.REPORTS)             # → s3://amzn-s3-ml-c/account1/reports/

    # Or override at CLI parse time:
    cfg = load_pipeline_config(account_prefix="account3")
    print(cfg.paths.PROCESSED)           # → s3://amzn-s3-ml-c/account3/processed/

Environment variables
---------------------
    ACCOUNT_PREFIX   Override the account prefix (default: "account1")
    AWS_PROFILE      AWS CLI profile (default: "amazon-ml-account1")

Authors: Account 1 (canonical / production pipeline)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_BUCKET         = "amzn-s3-ml-c"
_DEFAULT_ACCOUNT_PREFIX = "account1"
_DEFAULT_AWS_PROFILE    = "amazon-ml-account1"
_DEFAULT_AWS_REGION     = "us-east-1"
_DEFAULT_CONFIG_PATH    = Path("configs/config.yaml")


# ---------------------------------------------------------------------------
# Paths dataclass — all S3 URIs live here
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    """
    Immutable collection of S3 URIs derived from bucket + account_prefix.

    RAW and SHARED are shared across all accounts (no prefix).
    Everything else is scoped to the account prefix.
    """
    bucket:         str
    account_prefix: str

    # ---- Shared / raw (NO account prefix) ---------------------------------
    @property
    def RAW(self) -> str:
        return f"s3://{self.bucket}/raw/"

    @property
    def SHARED(self) -> str:
        return f"s3://{self.bucket}/shared/"

    # ---- Account-scoped paths ---------------------------------------------
    @property
    def PROCESSED(self) -> str:
        return f"s3://{self.bucket}/{self.account_prefix}/processed/"

    @property
    def CANDIDATES(self) -> str:
        return f"s3://{self.bucket}/{self.account_prefix}/candidates/"

    @property
    def FEATURES(self) -> str:
        return f"s3://{self.bucket}/{self.account_prefix}/features/"

    @property
    def MODELS(self) -> str:
        return f"s3://{self.bucket}/{self.account_prefix}/models/"

    @property
    def OUTPUTS(self) -> str:
        return f"s3://{self.bucket}/{self.account_prefix}/outputs/"

    @property
    def REPORTS(self) -> str:
        return f"s3://{self.bucket}/{self.account_prefix}/reports/"

    @property
    def EXPERIMENTS(self) -> str:
        return f"s3://{self.bucket}/{self.account_prefix}/experiments/"

    # ---- Raw data file keys (S3 object keys, not full URIs) ---------------
    # These are shared across all accounts; only the raw/ prefix is used.
    @property
    def raw_source_keys(self) -> Dict[str, str]:
        return {
            "train_source1": "raw/train_source1.tsv",
            "train_source2": "raw/train_source2.tsv",
            "train_source3": "raw/train_source3.tsv",
            "ground_truth":  "raw/train_ground_truth.tsv",
            "test_source1":  "raw/test_source1.tsv",
            "test_source2":  "raw/test_source2.tsv",
            "test_source3":  "raw/test_source3.tsv",
        }

    # ---- Helper: strip s3://bucket/ → bare S3 key ------------------------
    def key(self, s3_uri: str) -> str:
        """Convert a full s3://bucket/key URI back to the bare S3 key."""
        prefix = f"s3://{self.bucket}/"
        if not s3_uri.startswith(prefix):
            raise ValueError(f"URI {s3_uri!r} does not belong to bucket {self.bucket!r}")
        return s3_uri[len(prefix):]

    # ---- Helper: build a full s3:// URI from a bare key ------------------
    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"


# ---------------------------------------------------------------------------
# Top-level pipeline config object
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """
    Combines S3 paths with all other pipeline hyper-parameters loaded from
    configs/config.yaml.  Immutable paths; mutable hyper-params for
    experiment-level overrides.
    """
    paths:       Paths
    aws_profile: str
    aws_region:  str

    # Sub-sections of config.yaml — kept as plain dicts for flexibility
    profiling:   Dict[str, Any] = field(default_factory=dict)
    blocking:    Dict[str, Any] = field(default_factory=dict)
    model:       Dict[str, Any] = field(default_factory=dict)
    validation:  Dict[str, Any] = field(default_factory=dict)
    threshold:   Dict[str, Any] = field(default_factory=dict)

    # Raw YAML dict for any key not explicitly modelled
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_pipeline_config(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    account_prefix: Optional[str] = None,
    aws_profile: Optional[str] = None,
) -> PipelineConfig:
    """
    Load pipeline configuration.

    Priority order for account_prefix (highest → lowest):
      1. ``account_prefix`` argument passed directly
      2. ``ACCOUNT_PREFIX`` environment variable
      3. ``account_prefix`` key inside config.yaml
      4. Hard-coded default ``"account1"``

    Priority order for aws_profile (highest → lowest):
      1. ``aws_profile`` argument passed directly
      2. ``AWS_PROFILE`` environment variable
      3. ``aws.profile`` key inside config.yaml
      4. Hard-coded default ``"amazon-ml-account1"``

    Parameters
    ----------
    config_path    : Path to configs/config.yaml (default: "configs/config.yaml")
    account_prefix : Override account prefix (e.g. "account3" for teammate use)
    aws_profile    : Override AWS CLI profile name

    Returns
    -------
    PipelineConfig
    """
    config_path = Path(config_path)
    raw: Dict[str, Any] = {}
    if config_path.exists():
        with config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}

    aws_cfg = raw.get("aws", {})

    # ---- Resolve account prefix (4-level priority) -----------------------
    resolved_prefix = (
        account_prefix
        or os.environ.get("ACCOUNT_PREFIX")
        or raw.get("account_prefix")
        or _DEFAULT_ACCOUNT_PREFIX
    )

    # ---- Resolve AWS profile (4-level priority) --------------------------
    resolved_profile = (
        aws_profile
        or os.environ.get("AWS_PROFILE")
        or aws_cfg.get("profile")
        or _DEFAULT_AWS_PROFILE
    )

    bucket = aws_cfg.get("bucket", _DEFAULT_BUCKET)
    region = aws_cfg.get("region", _DEFAULT_AWS_REGION)

    paths = Paths(bucket=bucket, account_prefix=resolved_prefix)

    return PipelineConfig(
        paths       = paths,
        aws_profile = resolved_profile,
        aws_region  = region,
        profiling   = raw.get("profiling", {}),
        blocking    = raw.get("blocking", {}),
        model       = raw.get("model", {}),
        validation  = raw.get("validation", {}),
        threshold   = raw.get("threshold", {}),
        _raw        = raw,
    )


# ---------------------------------------------------------------------------
# Convenience: expose a module-level default config for scripts that just
# do `from src.entity_resolution.config import PATHS`
# ---------------------------------------------------------------------------
PATHS: Paths = load_pipeline_config().paths


# ---------------------------------------------------------------------------
# CLI helper — add_account_prefix_arg / resolve_paths_from_args
# ---------------------------------------------------------------------------

def add_account_prefix_arg(parser: "argparse.ArgumentParser") -> None:
    """
    Attach the standard --account-prefix argument to any argparse parser.

    Call this from every CLI entry-point so that Account 2/3 can simply
    pass --account-prefix account2 without modifying any code.
    """
    parser.add_argument(
        "--account-prefix",
        default=None,
        metavar="PREFIX",
        help=(
            "S3 account prefix (default: 'account1', or ACCOUNT_PREFIX env var). "
            "Set to 'account2' / 'account3' to route all outputs to a different "
            "S3 prefix without touching any source files."
        ),
    )


def resolve_config_from_args(
    args: "argparse.Namespace",
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
) -> PipelineConfig:
    """
    Build a PipelineConfig from parsed argparse args.

    Expects args to have: account_prefix (from add_account_prefix_arg),
    and optionally aws_profile (if added by the calling parser).
    """
    account_prefix = getattr(args, "account_prefix", None)
    aws_profile    = getattr(args, "aws_profile", None)
    return load_pipeline_config(
        config_path    = config_path,
        account_prefix = account_prefix,
        aws_profile    = aws_profile,
    )
