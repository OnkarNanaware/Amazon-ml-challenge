"""
normalization.py
================
Step 2 of the Amazon-ML-Challenge entity-resolution pipeline.

Exports a single pure function:

    normalize_batch(df: pd.DataFrame) -> pd.DataFrame

Design contract
---------------
* Operates on business_name and business_address INDEPENDENTLY — never
  concatenates them.  Outputs normalized_name / normalized_address as
  separate columns alongside the untouched original_name / original_address.
* Country normalization is OPEN-SET: only trim + casing + whitespace.
  No US/India-specific logic.  Generalises to any country (e.g. France).
* Null-safe for business_address (~3.4% missing in S2/S3): rows with no
  address get normalized_address=None and address_missing=True, allowing
  blocking.py to route them to a name-only path.
* The function itself has NO hardcoded S3 paths and NO account prefix.
  All I/O wiring lives in the CLI entry-point at the bottom, which imports
  paths from src.entity_resolution.config.

SageMaker handoff
-----------------
Because normalize_batch() is a pure DataFrame→DataFrame transform, it can be:
  (a) called directly here on small samples for validation,
  (b) imported unchanged into a SageMaker Processing Job script for full-scale
      execution, and
  (c) reused as-is by Account 2/3 by just changing ACCOUNT_PREFIX.

Authors: Account 1 (canonical / production pipeline)
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from typing import Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema column names (confirmed from profiling run)
# ---------------------------------------------------------------------------
_COL_ENTITY_ID      = "entity_id"
_COL_NAME           = "business_name"
_COL_ADDRESS        = "business_address"
_COL_COUNTRY        = "country"

_COL_ORIG_NAME      = "original_name"
_COL_ORIG_ADDRESS   = "original_address"
_COL_ORIG_COUNTRY   = "original_country"

_COL_NORM_NAME      = "normalized_name"
_COL_NORM_ADDRESS   = "normalized_address"
_COL_NORM_COUNTRY   = "normalized_country"
_COL_TOKEN_NAME     = "token_sorted_name"   # token-sorted variant for blocking
_COL_ADDR_MISSING   = "address_missing"     # True when business_address was null


# ---------------------------------------------------------------------------
# Dictionaries
# ---------------------------------------------------------------------------

# Business-name: legal entity suffixes to STRIP from the end of names.
# Lower-case, longest match first (so "incorporated" is tried before "inc").
# These are stripped (not replaced) because they carry no discriminative signal
# for entity matching but cause false non-matches ("Acme Inc" vs "Acme LLC").
_LEGAL_SUFFIXES = [
    "incorporated",
    "corporation",
    "associates",
    "unlimited",
    "solutions",
    "holdings",
    "services",
    "partners",
    "group",
    "limited",
    "company",
    "corp",
    "inc",
    "ltd",
    "llc",
    "llp",
    "lllp",
    "lp",
    "plc",
    "pllc",
    "pc",
    "co",
    "gmbh",
    "ag",
    "sa",
    "sas",
    "bv",
    "nv",
    "pty",
    "pvt",
    "srl",
    "aps",
    "oy",
    "ab",
    "as",
]

# Build a single alternation regex anchored at end-of-string.
# \b ensures we don't strip "co" from "costco".
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _LEGAL_SUFFIXES) + r")\.?\s*$",
    re.IGNORECASE,
)

# Business-name: modest abbreviation expansion applied AFTER punctuation pass.
# Note: & → and is handled inside _normalize_punctuation_name, not here.
# Tuples of (pattern, replacement) — applied in order, after lowercasing.
_NAME_ABBREV: list[tuple[str, str]] = [
    (r"\bintl\b",     "international"),
    (r"\bnat'?l\b",   "national"),
    (r"\bmfg\b",      "manufacturing"),
    (r"\bmgmt\b",     "management"),
    (r"\bsvcs\b",     "services"),
    (r"\bsvc\b",      "service"),
    (r"\btech\b",     "technology"),
    (r"\bdev\b",      "development"),
    (r"\bprod\b",     "products"),
    (r"\bpubs\b",     "publications"),
    (r"\bpub\b",      "publication"),
    (r"\bdist\b",     "distribution"),
    (r"\bcorp\b",     "corporation"),   # expand before suffix strip sees it
]
_NAME_ABBREV_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _NAME_ABBREV]

# Address: abbreviation expansions — applied AFTER lowercasing and punctuation pass.
# Word-boundary anchored so "st" only matches the whole token.
_ADDR_ABBREV: list[tuple[str, str]] = [
    # Street type expansions
    (r"\bst\b",    "street"),
    (r"\bave\b",   "avenue"),
    (r"\bav\b",    "avenue"),
    (r"\bblvd\b",  "boulevard"),
    (r"\brd\b",    "road"),
    (r"\bdr\b",    "drive"),
    (r"\bln\b",    "lane"),
    (r"\bct\b",    "court"),
    (r"\bctr\b",   "center"),
    (r"\bpl\b",    "place"),
    (r"\bsq\b",    "square"),
    (r"\bhwy\b",   "highway"),
    (r"\bfwy\b",   "freeway"),
    (r"\bpkwy\b",  "parkway"),
    (r"\bexpy\b",  "expressway"),
    (r"\bbrg\b",   "bridge"),
    (r"\btrk\b",   "track"),
    (r"\btrce\b",  "trace"),
    # Unit designators
    (r"\bste\b",   "suite"),
    (r"\bapt\b",   "apartment"),
    (r"\bfl\b",    "floor"),
    (r"\bflr\b",   "floor"),
    (r"\bbldg\b",  "building"),
    (r"\brm\b",    "room"),
    (r"\bdept\b",  "department"),
    # Cardinal directions
    (r"\bn\b",     "north"),
    (r"\bs\b",     "south"),
    (r"\be\b",     "east"),
    (r"\bw\b",     "west"),
    (r"\bne\b",    "northeast"),
    (r"\bnw\b",    "northwest"),
    (r"\bse\b",    "southeast"),
    (r"\bsw\b",    "southwest"),
]
_ADDR_ABBREV_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _ADDR_ABBREV]


# ---------------------------------------------------------------------------
# Low-level string helpers
# ---------------------------------------------------------------------------

def _nfkc(s: str) -> str:
    """NFKC Unicode normalization — handles accented chars, ligatures, full-width."""
    return unicodedata.normalize("NFKC", s)


def _collapse_whitespace(s: str) -> str:
    """Collapse runs of whitespace to a single space and strip edges."""
    return re.sub(r"\s+", " ", s).strip()


def _normalize_punctuation_name(s: str) -> str:
    """
    For business names:
    - Expand '&' to 'and' FIRST (before generic punctuation removal).
    - Replace hyphens/slashes used as separators with space.
    - Remove periods that appear to be abbreviation dots (A.B.C. → ABC).
    - Remove all remaining punctuation except apostrophes within words.
    """
    # & → and  (must happen before the generic punctuation strip below)
    s = re.sub(r"\s*&\s*", " and ", s)
    # Abbreviation dots: letter DOT letter → remove dot
    s = re.sub(r"(?<=[a-zA-Z])\.(?=[a-zA-Z])", "", s)
    # Trailing dot on a word: "Inc." → "Inc"
    s = re.sub(r"\.(?=\s|$)", "", s)
    # Hyphens / slashes / pipes as word separators → space
    s = re.sub(r"[\-/|\\]", " ", s)
    # Remove all remaining non-alphanumeric except apostrophes and spaces
    s = re.sub(r"[^\w\s']", " ", s)
    return s



def _normalize_punctuation_addr(s: str) -> str:
    """
    For addresses:
    - Keep hyphens inside numeric ranges (e.g. "123-45" → keep as is).
    - Remove commas (common US/India address separator).
    - Remove periods.
    - Replace other punctuation with space.
    """
    # Remove commas and periods
    s = re.sub(r"[,.]", " ", s)
    # Replace anything that isn't alphanumeric, space, or hyphen with space
    s = re.sub(r"[^\w\s\-]", " ", s)
    return s


# ---------------------------------------------------------------------------
# Field-level normalisation pipelines
# ---------------------------------------------------------------------------

def normalize_name_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Apply the business-name normalization pipeline to a pandas Series.

    Returns
    -------
    (normalized_name_series, token_sorted_name_series)
    """
    def _normalize_one(raw: object) -> tuple[str | None, str | None]:
        if pd.isna(raw) or str(raw).strip() == "":
            return None, None

        s = str(raw)
        # 1. NFKC — handles accented chars (Café → Cafe), ligatures, full-width
        s = _nfkc(s)
        # 2. Lowercase
        s = s.lower()
        # 3. Whitespace normalize (early pass — removes tabs, newlines)
        s = _collapse_whitespace(s)
        # 4. Punctuation handling
        s = _normalize_punctuation_name(s)
        # 5. Abbreviation expansion (& → and, intl → international, …)
        for pattern, replacement in _NAME_ABBREV_RE:
            s = pattern.sub(replacement, s)
        # 6. Legal suffix strip (at end of string only)
        s = _LEGAL_SUFFIX_RE.sub("", s).rstrip(" .,;:-")
        # 7. Final whitespace collapse
        s = _collapse_whitespace(s)

        # 8. Token-sorted variant (for blocking / approximate matching)
        tokens = s.split()
        token_sorted = " ".join(sorted(tokens))

        return s or None, token_sorted or None

    results = series.map(_normalize_one)
    norm   = results.map(lambda x: x[0])
    tsorted = results.map(lambda x: x[1])
    return norm, tsorted


def normalize_address_series(series: pd.Series) -> pd.Series:
    """
    Apply the address normalization pipeline to a pandas Series.

    Returns
    -------
    normalized_address series (None where input was null/empty)
    """
    def _normalize_one(raw: object) -> str | None:
        if pd.isna(raw) or str(raw).strip() == "":
            return None

        s = str(raw)
        # 1. NFKC
        s = _nfkc(s)
        # 2. Lowercase
        s = s.lower()
        # 3. Punctuation normalization
        s = _normalize_punctuation_addr(s)
        # 4. Whitespace collapse
        s = _collapse_whitespace(s)
        # 5. Address abbreviation expansion (word-boundary safe)
        for pattern, replacement in _ADDR_ABBREV_RE:
            s = pattern.sub(replacement, s)
        # 6. Final whitespace collapse
        s = _collapse_whitespace(s)
        return s or None

    return series.map(_normalize_one)


def normalize_country_series(series: pd.Series) -> pd.Series:
    """
    Open-set country normalization: trim + title-case + whitespace collapse.
    No country-specific logic — generalises to any country value.
    """
    def _normalize_one(raw: object) -> str | None:
        if pd.isna(raw) or str(raw).strip() == "":
            return None
        s = str(raw)
        s = _nfkc(s)
        s = _collapse_whitespace(s)
        # Title-case preserves "United States", "United Kingdom", etc.
        s = s.title()
        return s or None

    return series.map(_normalize_one)


# ---------------------------------------------------------------------------
# Public API — pure function, no S3 / path dependencies
# ---------------------------------------------------------------------------

def normalize_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a batch of entity records.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: entity_id, business_name, business_address, country.
        Extra columns are passed through unchanged.

    Returns
    -------
    pd.DataFrame with additional columns:
        original_name         — copy of raw business_name
        original_address      — copy of raw business_address
        original_country      — copy of raw country
        normalized_name       — normalized business_name
        normalized_address    — normalized business_address (None if missing)
        normalized_country    — normalized country (open-set)
        token_sorted_name     — space-joined sorted tokens of normalized_name
        address_missing       — bool True when business_address was null/empty

    Original columns (entity_id, business_name, etc.) are preserved unchanged.
    """
    out = df.copy()

    # --- Preserve originals (never overwrite raw columns) ------------------
    out[_COL_ORIG_NAME]    = out[_COL_NAME].copy()
    out[_COL_ORIG_ADDRESS] = out[_COL_ADDRESS].copy() if _COL_ADDRESS in out.columns else None
    out[_COL_ORIG_COUNTRY] = out[_COL_COUNTRY].copy() if _COL_COUNTRY in out.columns else None

    # --- Address-missing flag (before normalization, based on raw value) ----
    if _COL_ADDRESS in out.columns:
        out[_COL_ADDR_MISSING] = out[_COL_ADDRESS].isna() | (
            out[_COL_ADDRESS].astype(str).str.strip() == ""
        )
    else:
        out[_COL_ADDR_MISSING] = True

    # --- Normalize business_name -------------------------------------------
    if _COL_NAME in out.columns:
        norm_name, token_sorted = normalize_name_series(out[_COL_NAME])
        out[_COL_NORM_NAME]  = norm_name
        out[_COL_TOKEN_NAME] = token_sorted
    else:
        out[_COL_NORM_NAME]  = None
        out[_COL_TOKEN_NAME] = None

    # --- Normalize business_address ----------------------------------------
    if _COL_ADDRESS in out.columns:
        out[_COL_NORM_ADDRESS] = normalize_address_series(out[_COL_ADDRESS])
    else:
        out[_COL_NORM_ADDRESS] = None

    # --- Normalize country (open-set) --------------------------------------
    if _COL_COUNTRY in out.columns:
        out[_COL_NORM_COUNTRY] = normalize_country_series(out[_COL_COUNTRY])
    else:
        out[_COL_NORM_COUNTRY] = None

    return out


# ---------------------------------------------------------------------------
# Before / after comparison printer
# ---------------------------------------------------------------------------

def print_before_after(df_raw: pd.DataFrame, df_norm: pd.DataFrame, n: int = 15) -> None:
    """
    Print a side-by-side before/after table for the first *n* rows.
    Truncates long strings for readability.
    """
    sep = "=" * 110

    def _trunc(s: object, width: int = 35) -> str:
        t = "" if pd.isna(s) else str(s)
        return t[:width] + "…" if len(t) > width else t

    print(f"\n{sep}")
    print("  NORMALIZATION — BEFORE / AFTER SAMPLE")
    print(f"  Showing first {n} rows")
    print(sep)
    header = (
        f"{'#':>4}  "
        f"{'original_name':<36} {'normalized_name':<36} "
        f"{'token_sorted':<36}"
    )
    print(header)
    print("-" * 110)

    sample = df_norm.head(n)
    for i, row in sample.iterrows():
        orig_name  = _trunc(row.get(_COL_ORIG_NAME, ""))
        norm_name  = _trunc(row.get(_COL_NORM_NAME, ""))
        tok_sorted = _trunc(row.get(_COL_TOKEN_NAME, ""))
        print(f"{i:>4}  {orig_name:<36} {norm_name:<36} {tok_sorted:<36}")

    print()
    print(f"{'#':>4}  "
          f"{'original_address':<55} {'normalized_address':<55}")
    print("-" * 110)
    for i, row in sample.iterrows():
        orig_addr = _trunc(row.get(_COL_ORIG_ADDRESS, ""), 52)
        norm_addr = _trunc(row.get(_COL_NORM_ADDRESS, ""), 52)
        missing   = "⚠ NULL" if row.get(_COL_ADDR_MISSING, False) else ""
        print(f"{i:>4}  {orig_addr:<55} {norm_addr:<55} {missing}")

    print()
    print(f"{'#':>4}  {'original_country':<20} {'normalized_country':<20}")
    print("-" * 50)
    for i, row in sample.iterrows():
        print(f"{i:>4}  {_trunc(row.get(_COL_ORIG_COUNTRY,''), 18):<20} "
              f"{_trunc(row.get(_COL_NORM_COUNTRY,''), 18):<20}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# S3 I/O helpers (only used in the CLI — not in normalize_batch)
# ---------------------------------------------------------------------------

def _get_s3_client(aws_profile: str, region: str):
    session = boto3.Session(profile_name=aws_profile, region_name=region)
    return session.client("s3")


def _read_tsv_from_s3(s3, bucket: str, key: str, nrows: Optional[int] = None) -> pd.DataFrame:
    logger.info("Reading s3://%s/%s (nrows=%s)", bucket, key, nrows)
    obj  = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]
    if nrows is not None:
        lines = [body.readline()]
        for i, line in enumerate(body.iter_lines()):
            if i >= nrows:
                break
            lines.append(line)
        raw = b"\n".join(lines)
    else:
        raw = body.read()
    df = pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False)
    logger.info("  -> loaded %d rows x %d cols", len(df), len(df.columns))
    return df


def _write_parquet_to_s3(df: pd.DataFrame, s3, bucket: str, key: str) -> None:
    table  = pa.Table.from_pandas(df, preserve_index=False)
    buf    = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info("Parquet written -> s3://%s/%s  (%d rows)", bucket, key, len(df))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    End-to-end normalization of a sampled source file:
      1. Read sample from S3 (RAW prefix)
      2. normalize_batch()
      3. Write normalized Parquet → {PROCESSED}normalized_sample.parquet
      4. Print before/after comparison table
    """
    import argparse

    from src.entity_resolution.config import (
        add_account_prefix_arg,
        resolve_config_from_args,
    )

    parser = argparse.ArgumentParser(
        description="Normalize a sample of entity records from S3."
    )
    add_account_prefix_arg(parser)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--aws-profile", default=None, dest="aws_profile")
    parser.add_argument(
        "--source",
        default="train_source1",
        choices=["train_source1", "train_source2", "train_source3"],
        help="Which source file to normalize (default: train_source1)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10_000,
        help="Rows to read (default: 10000). Full-scale runs belong on SageMaker.",
    )
    parser.add_argument(
        "--output-key", default=None,
        help="Override the output S3 key (default: <PROCESSED>normalized_sample.parquet)",
    )
    parser.add_argument(
        "--show-rows", type=int, default=15,
        help="Number of rows to show in before/after table (default: 15)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cfg    = resolve_config_from_args(args, config_path=args.config)
    paths  = cfg.paths
    bucket = paths.bucket

    s3 = _get_s3_client(cfg.aws_profile, cfg.aws_region)

    # Input: raw TSV
    src_key = paths.raw_source_keys[args.source]
    df_raw  = _read_tsv_from_s3(s3, bucket, src_key, nrows=args.sample_size)

    # Normalize
    logger.info("Running normalize_batch on %d rows...", len(df_raw))
    df_norm = normalize_batch(df_raw)
    logger.info("Normalization complete.")

    # Output: Parquet to PROCESSED
    out_key = args.output_key or (paths.key(paths.PROCESSED) + "normalized_sample.parquet")
    _write_parquet_to_s3(df_norm, s3, bucket, out_key)

    # Console comparison table
    print_before_after(df_raw, df_norm, n=args.show_rows)

    # Summary stats
    addr_miss = int(df_norm["address_missing"].sum())
    total     = len(df_norm)
    print(f"  Summary: {total:,} rows | "
          f"address_missing: {addr_miss:,} ({addr_miss/total*100:.2f}%) | "
          f"output: s3://{bucket}/{out_key}")
    print()


if __name__ == "__main__":
    main()
