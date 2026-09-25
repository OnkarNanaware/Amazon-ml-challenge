"""
audit_processed_v2.py  (Task 0b)
=================================
Runs all 8 audit checks against S1, S2, and S3 normalized samples individually.
Verifies source-column fix and the S2/S3 address-missing null-safety code path.

Output: {REPORTS}processed_data_audit_v2.json + stdout summary

Run:
    PYTHONPATH=. python scripts/audit_processed_v2.py --aws-profile amazon-ml-account1
"""
from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd

from src.entity_resolution.config import add_account_prefix_arg, resolve_config_from_args

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLS = {
    "entity_id", "source",                              # NEW: source tag required
    "original_name", "normalized_name",
    "original_address", "normalized_address", "address_missing",
    "normalized_country", "original_country", "token_sorted_name",
    "business_name", "business_address", "country",
}


# ── S3 helpers ───────────────────────────────────────────────────────────────

def _s3(profile, region):
    return boto3.Session(profile_name=profile, region_name=region).client("s3")

def _parquet(s3c, bucket, key):
    return pd.read_parquet(io.BytesIO(s3c.get_object(Bucket=bucket, Key=key)["Body"].read()))

def _tsv_head(s3c, bucket, key, nrows=10_000):
    body = s3c.get_object(Bucket=bucket, Key=key)["Body"]
    lines = [body.readline()]
    for i, ln in enumerate(body.iter_lines()):
        if i >= nrows: break
        lines.append(ln)
    return pd.read_csv(io.BytesIO(b"\n".join(lines)), sep="\t", low_memory=False)

def _ls(s3c, bucket, prefix):
    pag, out = s3c.get_paginator("list_objects_v2"), []
    for pg in pag.paginate(Bucket=bucket, Prefix=prefix):
        for o in pg.get("Contents", []):
            if not o["Key"].endswith("/"):
                out.append({"key": o["Key"], "size_bytes": o["Size"],
                             "last_modified": o["LastModified"].isoformat()})
    return out


# ── Per-source audit ─────────────────────────────────────────────────────────

def audit_source(
    s3c, bucket: str, parquet_key: str,
    raw_key: str, expected_source_tag: str,
) -> Dict[str, Any]:
    """Run all 8 checks for one source's normalized parquet."""

    # Load files
    logger.info("Auditing: s3://%s/%s", bucket, parquet_key)
    df_norm = _parquet(s3c, bucket, parquet_key)
    df_raw  = _tsv_head(s3c, bucket, raw_key, 10_000)
    nc      = "business_name"    if "business_name"    in df_raw.columns else "name"
    ac      = "business_address" if "business_address" in df_raw.columns else "address"

    checks: Dict[str, Any] = {}

    # 1. Schema
    present = set(df_norm.columns)
    missing = sorted(REQUIRED_COLS - present)
    checks["2_schema"] = {
        "pass": len(missing) == 0,
        "columns": sorted(present),
        "missing": missing,
        "source_col_present": "source" in present,
        "source_values":      df_norm["source"].unique().tolist() if "source" in present else [],
        "source_matches_expected": (
            df_norm["source"].iloc[0] == expected_source_tag
            if "source" in present and len(df_norm) > 0 else False
        ),
    }

    # 3. Originals preserved (10 random rows)
    ids      = df_norm["entity_id"].sample(min(10, len(df_norm)), random_state=42).tolist()
    ns       = df_norm[df_norm["entity_id"].isin(ids)][["entity_id","original_name","original_address"]]
    rs       = df_raw [df_raw ["entity_id"].isin(ids)][["entity_id", nc, ac]]
    mg       = ns.merge(rs, on="entity_id")
    bad3 = []
    for _, row in mg.iterrows():
        if str(row["original_name"]) != str(row[nc]):
            bad3.append({"eid": row["entity_id"], "field": "name",
                         "parquet": row["original_name"], "raw": row[nc]})
        ra, pa = row[ac], row["original_address"]
        if not (pd.isna(ra) and pd.isna(pa)) and str(ra) != str(pa):
            bad3.append({"eid": row["entity_id"], "field": "address",
                         "parquet": pa, "raw": ra})
    checks["3_originals"] = {"pass": len(bad3)==0, "rows_checked": len(mg), "mismatches": bad3}

    # 4. Null-safety — THE KEY CHECK for S2/S3
    if "address_missing" not in df_norm.columns:
        checks["4_null_safety"] = {"pass": False, "error": "address_missing column absent"}
    else:
        miss   = df_norm[df_norm["address_missing"] == True]
        total  = len(df_norm)
        # Rows flagged missing but norm_addr is not null → BUG
        bad_not_null  = miss[miss["normalized_address"].notna()]
        # Rows not flagged missing but norm_addr is null → BUG
        present_rows  = df_norm[df_norm["address_missing"] == False]
        wrongly_null  = present_rows[present_rows["normalized_address"].isna()]
        checks["4_null_safety"] = {
            "pass":                len(bad_not_null) == 0,
            "total_rows":          total,
            "address_missing_count": len(miss),
            "address_missing_pct": round(len(miss)/total*100, 4) if total else 0,
            "flag_true_norm_addr_not_null": len(bad_not_null),
            "flag_false_norm_addr_null":    len(wrongly_null),
            "missing_examples": miss[["entity_id","original_address","normalized_address"]] \
                .head(3).to_dict("records") if len(miss)>0 else [],
            "bad_examples": bad_not_null[["entity_id","original_address","normalized_address"]] \
                .head(3).to_dict("records"),
        }

    # 5. Name normalization spot-check
    legal = ["Inc", "LLC", "Ltd", "Corp", "Limited", "Incorporated"]
    rows5 = df_norm[df_norm["original_name"].str.contains("|".join(legal), na=False, case=False)].head(8)
    ex5 = []
    all5_ok = True
    for _, r in rows5.iterrows():
        orig  = str(r.get("original_name",""))
        norm  = str(r.get("normalized_name",""))
        tsort = str(r.get("token_sorted_name",""))
        suffix_still = any(t.lower() in norm.split()
                           for t in ["inc","llc","ltd","corp","limited","incorporated"])
        amp_ok = "&" not in norm if "&" in orig else True
        tok_ok = tsort == " ".join(sorted(norm.split())) if norm and norm!="nan" else True
        if suffix_still or not amp_ok or not tok_ok:
            all5_ok = False
        ex5.append({"original": orig[:55], "normalized": norm[:55],
                    "suffix_stripped": not suffix_still, "amp_ok": amp_ok, "tok_ok": tok_ok})
    checks["5_name_norm"] = {"pass": all5_ok, "rows_checked": len(ex5), "examples": ex5[:4]}

    # 6. Non-ASCII
    def _na(s):
        if pd.isna(s): return False
        try: str(s).encode("ascii"); return False
        except UnicodeEncodeError: return True
    mask6   = df_norm["original_name"].apply(_na)
    na_rows = df_norm[mask6]
    mangled = na_rows[na_rows["normalized_name"].str.contains("\ufffd", na=False)]
    ctries  = sorted(df_norm["normalized_country"].dropna().astype(str).unique().tolist())
    checks["6_non_ascii"] = {
        "pass": len(mangled)==0,
        "non_ascii_found": int(mask6.sum()), "mangled": len(mangled),
        "unique_countries": ctries,
        "examples": na_rows[["entity_id","original_name","normalized_name"]].head(3).to_dict("records"),
    }

    # 7. Row count
    checks["7_rowcount"] = {
        "pass": len(df_norm)==len(df_raw),
        "norm": len(df_norm), "raw": len(df_raw), "delta": len(df_norm)-len(df_raw)
    }

    # 8. Duplicates
    n_dup = int(df_norm["entity_id"].duplicated().sum())
    checks["8_duplicates"] = {"pass": n_dup==0, "duplicate_entity_ids": n_dup}

    overall = all(v.get("pass", False) for v in checks.values())
    return {
        "source_tag":    expected_source_tag,
        "parquet_uri":   f"s3://{bucket}/{parquet_key}",
        "row_count":     len(df_norm),
        "col_count":     len(df_norm.columns),
        "overall_pass":  overall,
        "checks":        checks,
    }


# ── Print summary ────────────────────────────────────────────────────────────

def _print_source(tag: str, result: Dict) -> None:
    sep = "-" * 70
    ov  = "✅ PASS" if result["overall_pass"] else "❌ FAIL"
    print(f"\n  ── {tag.upper()} ({result['row_count']:,} rows, {result['col_count']} cols)  {ov}")
    for cid, res in result["checks"].items():
        st  = "✅" if res.get("pass") else "❌"
        lbl = cid.replace("_"," ").title()
        print(f"     {st}  {lbl}")
        if cid == "2_schema":
            print(f"         source_col: {res['source_col_present']}  "
                  f"values={res['source_values']}  "
                  f"matches_expected({tag}): {res['source_matches_expected']}")
            if res["missing"]:
                print(f"         ❌ missing cols: {res['missing']}")
        elif cid == "3_originals":
            print(f"         rows_checked={res['rows_checked']}  mismatches={len(res['mismatches'])}")
        elif cid == "4_null_safety":
            cnt = res.get("address_missing_count", 0)
            tot = res.get("total_rows", 0)
            pct = res.get("address_missing_pct", 0)
            print(f"         address_missing: {cnt}/{tot} = {pct:.2f}%  "
                  f"[expected ~3.3-3.5% for s2/s3, 0% for s1]")
            print(f"         flag=True+norm_not_null: {res.get('flag_true_norm_addr_not_null',0)}  "
                  f"flag=False+norm_null: {res.get('flag_false_norm_addr_null',0)}")
            if res.get("missing_examples"):
                ex = res["missing_examples"][0]
                print(f"         Example missing row: entity={ex['entity_id']} "
                      f"orig_addr={repr(str(ex['original_address'])[:40])} "
                      f"norm_addr={repr(str(ex['normalized_address'])[:20])}")
        elif cid == "5_name_norm":
            for ex in res.get("examples",[])[:2]:
                print(f"         [{('✓' if ex['suffix_stripped'] else '✗')}suf]  "
                      f"{ex['original'][:35]:35s} -> {ex['normalized'][:35]}")
        elif cid == "6_non_ascii":
            print(f"         non-ASCII={res['non_ascii_found']}  mangled={res['mangled']}  "
                  f"countries={res['unique_countries']}")
        elif cid == "7_rowcount":
            print(f"         norm={res['norm']}  raw={res['raw']}  delta={res['delta']}")
        elif cid == "8_duplicates":
            print(f"         duplicate_entity_ids={res['duplicate_entity_ids']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    add_account_prefix_arg(p)
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--aws-profile", default=None, dest="aws_profile")
    args = p.parse_args()

    cfg    = resolve_config_from_args(args, config_path=args.config)
    paths  = cfg.paths
    bucket = paths.bucket
    s3c    = _s3(cfg.aws_profile, cfg.aws_region)

    proc   = paths.key(paths.PROCESSED)
    raw_ks = paths.raw_source_keys

    # 1. List {PROCESSED}
    files = _ls(s3c, bucket, proc)
    logger.info("Files in {PROCESSED}:")
    for f in files:
        logger.info("  %s  %d B  %s", f["key"].split("/")[-1], f["size_bytes"], f["last_modified"][:19])

    # Check all three named files exist
    needed = {
        "s1": proc + "normalized_s1_sample.parquet",
        "s2": proc + "normalized_s2_sample.parquet",
        "s3": proc + "normalized_s3_sample.parquet",
    }
    keys_present = {f["key"] for f in files}
    missing_files = [k for k in needed.values() if k not in keys_present]
    if missing_files:
        logger.error("Missing processed files: %s", missing_files)
        raise SystemExit(1)

    # Audit each source
    sources = {
        "s1": (needed["s1"], raw_ks["train_source1"]),
        "s2": (needed["s2"], raw_ks["train_source2"]),
        "s3": (needed["s3"], raw_ks["train_source3"]),
    }
    results = {}
    for tag, (pk, rk) in sources.items():
        results[tag] = audit_source(s3c, bucket, pk, rk, expected_source_tag=tag)

    # Assemble report
    report = {
        "generated_at":    datetime.now(tz=timezone.utc).isoformat(),
        "account_prefix":  paths.account_prefix,
        "processed_files": files,
        "source_audits":   results,
        "overall_pass":    all(r["overall_pass"] for r in results.values()),
    }

    # Upload
    rk = paths.key(paths.REPORTS) + "processed_data_audit_v2.json"
    body = json.dumps(report, indent=2, default=str).encode()
    s3c.put_object(Bucket=bucket, Key=rk, Body=body, ContentType="application/json")
    logger.info("Audit v2 uploaded -> s3://%s/%s", bucket, rk)

    # Print
    print(f"\n{'='*72}")
    print("  PROCESSED DATA AUDIT v2  (S1 + S2 + S3)")
    print(f"{'='*72}")
    for tag, res in results.items():
        _print_source(tag, res)
    ov = "✅ ALL PASS" if report["overall_pass"] else "❌ SOME FAILED"
    print(f"\n  OVERALL: {ov}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
