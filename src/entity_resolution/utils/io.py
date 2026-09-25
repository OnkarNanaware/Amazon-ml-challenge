"""
io.py
=====
S3 I/O utilities implemented using AWS CLI subprocess calls.
boto3 is NOT used in this environment. All S3 I/O goes through `aws s3 cp`.

Authors: Account 3
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_AWS_PROFILE = "amazon-ml-account3"


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    logger.debug("Executing shell command: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("Command failed [%d]: %s\nStderr: %s", proc.returncode, " ".join(cmd), proc.stderr)
        raise RuntimeError(f"AWS CLI command failed with code {proc.returncode}: {proc.stderr.strip()}")
    return proc


def s3_cp(src: str, dst: str, profile: str = DEFAULT_AWS_PROFILE) -> None:
    """Copy a file between local and S3 or between S3 locations using AWS CLI."""
    cmd = ["aws", "s3", "cp", src, dst, "--profile", profile]
    _run_cmd(cmd)


def read_parquet_s3(s3_uri: str, profile: str = DEFAULT_AWS_PROFILE) -> pd.DataFrame:
    """Download a parquet file from S3 to a temporary file and load into DataFrame."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        s3_cp(s3_uri, tmp_path, profile=profile)
        return pd.read_parquet(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_parquet_s3(df: pd.DataFrame, s3_uri: str, profile: str = DEFAULT_AWS_PROFILE) -> None:
    """Save a DataFrame to temporary parquet and upload to S3 using AWS CLI."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        df.to_parquet(tmp_path, index=False)
        s3_cp(tmp_path, s3_uri, profile=profile)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_json_s3(s3_uri: str, profile: str = DEFAULT_AWS_PROFILE) -> Any:
    """Download JSON from S3 and parse into Python object."""
    cmd = ["aws", "s3", "cp", s3_uri, "-", "--profile", profile]
    proc = _run_cmd(cmd)
    return json.loads(proc.stdout)


def write_json_s3(data: Any, s3_uri: str, profile: str = DEFAULT_AWS_PROFILE, indent: int = 2) -> None:
    """Write Python object to temporary JSON file and upload to S3."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=indent, default=str)
        tmp_path = tmp.name
    try:
        s3_cp(tmp_path, s3_uri, profile=profile)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_csv_s3(
    s3_uri: str,
    sep: str = "\t",
    profile: str = DEFAULT_AWS_PROFILE,
    nrows: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Download a CSV/TSV from S3 (or sample nrows) and load into DataFrame."""
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        s3_cp(s3_uri, tmp_path, profile=profile)
        return pd.read_csv(tmp_path, sep=sep, nrows=nrows, low_memory=False, **kwargs)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
