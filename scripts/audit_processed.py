"""
audit_processed.py  (Task 0)
============================
Validates everything in {PROCESSED} against an 8-point checklist.

Output: {REPORTS}processed_data_audit.json + stdout summary.

Run:
    PYTHONPATH=. python scripts/audit_processed.py --aws-profile amazon-ml-account1
"""
from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
import pandas as pd

from src.entity_resolution.config import add_account_prefix_arg, resolve_config_from_args

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLS = {
    "entity_id", "original_name", "normalized_name",
    "original_address", "normalized_address", "address_missing",
    "normalized_country", "original_country", "token_sorted_name",
    "business_name", "business_address", "country",
}


# ── S3 helpers ──────────────────────────────────────────────────────────────

def _s3(profile, region):
    return boto3.Session(profile_name=profile, region_name=region).client("s3")

def _parquet(s3, bucket, key):
    return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read()))

def _tsv_head(s3, bucket, key, nrows=10_000):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    lines = [body.readline()]
    for i, ln in enumerate(body.iter_lines()):
        if i >= nrows: break
        lines.append(ln)
    return pd.read_csv(io.BytesIO(b"\n".join(lines)), sep="\t", low_memory=False)

def _ls(s3, bucket, prefix):
    pag, out = s3.get_paginator("list_objects_v2"), []
    for pg in pag.paginate(Bucket=bucket, Prefix=prefix):
        for o in pg.get("Contents", []):
            if not o["Key"].endswith("/"):
                out.append({"key": o["Key"], "size_bytes": o["Size"],
                             "last_modified": o["LastModified"].isoformat()})
    return out


# ── Checklist functions ──────────────────────────────────────────────────────

def c1_inventory(s3, bucket, prefix):
    files = _ls(s3, bucket, prefix)
    return {"pass": len(files) > 0, "file_count": len(files), "files": files}


def c2_schema(df):
    present = set(df.columns)
    missing = sorted(REQUIRED_COLS - present)
    has_source = "source" in present
    return {
        "pass": len(missing) == 0,
        "columns_present": sorted(present),
        "missing_required": missing,
        "source_col_present": has_source,
        "note": (
            "WARN: 'source' column absent. Must be injected at pipeline-runner level "
            "(pass source tag when calling normalize_batch). This is a known gap — "
            "normalization.py does not add it automatically."
        ) if not has_source else "OK",
    }


def c3_originals(df_norm, df_raw, n=10):
    if df_raw.empty or df_norm.empty:
        return {"pass": False, "error": "empty dataframe"}
    nc = "business_name"  if "business_name"  in df_raw.columns else "name"
    ac = "business_address" if "business_address" in df_raw.columns else "address"
    ids = df_norm["entity_id"].sample(min(n, len(df_norm)), random_state=42).tolist()
    # Merge only the columns we need to avoid suffix collision
    norm_sub = df_norm[df_norm["entity_id"].isin(ids)][["entity_id","original_name","original_address"]]
    raw_sub  = df_raw [df_raw ["entity_id"].isin(ids)][["entity_id", nc, ac]]
    mg = norm_sub.merge(raw_sub, on="entity_id")

    bad = []
    for _, row in mg.iterrows():
        if str(row["original_name"]) != str(row[nc]):
            bad.append({"eid": row["entity_id"], "field": "name",
                        "parquet": row["original_name"], "raw": row[nc]})
        ra, pa = row[ac], row["original_address"]
        if not (pd.isna(ra) and pd.isna(pa)) and str(ra) != str(pa):
            bad.append({"eid": row["entity_id"], "field": "address",
                        "parquet": pa, "raw": ra})

    examples = mg.rename(columns={nc: f"raw_{nc}", ac: f"raw_{ac}"}).head(5).to_dict("records")
    return {"pass": len(bad)==0, "rows_checked": len(mg),
            "mismatches": bad, "spot_check_examples": examples}


def c4_null_safety(df):
    if "address_missing" not in df.columns:
        return {"pass": False, "error": "address_missing column absent"}
    miss  = df[df["address_missing"] == True]
    total = len(df)
    bad_not_null  = miss[miss["normalized_address"].notna()]
    present_rows  = df[df["address_missing"] == False]
    wrongly_null  = present_rows[present_rows["normalized_address"].isna()]
    return {
        "pass": len(bad_not_null) == 0,
        "total_rows": total,
        "address_missing_count": len(miss),
        "address_missing_pct":   round(len(miss)/total*100, 4) if total else 0,
        "expected_pct_range":    "0% for S1; ~3.3-3.5% for S2/S3",
        "flag_true_but_norm_addr_not_null": len(bad_not_null),
        "flag_false_but_norm_addr_null":    len(wrongly_null),
        "bad_examples": bad_not_null[["entity_id","original_address","normalized_address"]].head(5).to_dict("records"),
    }


def c5_name_norm(df):
    legal = ["Inc", "LLC", "Ltd", "Corp", "Limited", "Incorporated"]
    rows  = df[df["original_name"].str.contains("|".join(legal), na=False, case=False)].head(10)
    checks = []
    for _, r in rows.iterrows():
        orig  = str(r.get("original_name",""))
        norm  = str(r.get("normalized_name",""))
        tsort = str(r.get("token_sorted_name",""))
        suffix_still = any(t.lower() in norm.split()
                           for t in ["inc","llc","ltd","corp","limited","incorporated"])
        amp_ok = "&" not in norm if "&" in orig else True
        tok_ok = tsort == " ".join(sorted(norm.split())) if norm and norm!="nan" else True
        checks.append({
            "entity_id": r.get("entity_id"),
            "original_name": orig[:60],
            "normalized_name": norm[:60],
            "token_sorted_name": tsort[:60],
            "legal_suffix_stripped": not suffix_still,
            "amp_normalized": amp_ok,
            "token_sort_correct": tok_ok,
        })
    ok = all(c["legal_suffix_stripped"] and c["amp_normalized"] and c["token_sort_correct"]
             for c in checks) if checks else True
    return {"pass": ok, "rows_checked": len(checks), "examples": checks[:8]}


def c6_non_ascii(df):
    def has_na(s):
        if pd.isna(s): return False
        try: str(s).encode("ascii"); return False
        except UnicodeEncodeError: return True

    mask    = df["original_name"].apply(has_na)
    na_rows = df[mask]
    mangled = na_rows[na_rows["normalized_name"].str.contains("\ufffd", na=False)]
    countries  = sorted(df["normalized_country"].dropna().astype(str).unique().tolist())
    narrow_set = set(countries) <= {"Us","India","United States","US","us","india"}
    examples   = na_rows[["entity_id","original_name","normalized_name","normalized_country"]].head(5).to_dict("records")
    return {
        "pass": len(mangled)==0,
        "non_ascii_rows_found": int(mask.sum()),
        "mangled_rows": len(mangled),
        "unique_countries": countries,
        "country_narrow_warning": narrow_set,
        "note": "Only US/India visible — expected for this 10K S1 sample (training data). Normalization is still open-set." if narrow_set else "OK",
        "examples": examples,
    }


def c7_rowcount(df_norm, df_raw):
    n, r = len(df_norm), len(df_raw)
    return {"pass": n==r, "norm_rows": n, "raw_rows": r, "delta": n-r}


def c8_duplicates(df):
    n_dup = int(df["entity_id"].duplicated().sum())
    exs   = []
    if n_dup > 0:
        ids = df[df["entity_id"].duplicated(keep=False)]["entity_id"].unique()[:3]
        exs = df[df["entity_id"].isin(ids)][["entity_id","original_name"]].to_dict("records")
    return {"pass": n_dup==0, "total_rows": len(df),
            "duplicate_entity_ids": n_dup, "examples": exs}


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_audit(cfg):
    paths  = cfg.paths
    bucket = paths.bucket
    s3c    = _s3(cfg.aws_profile, cfg.aws_region)

    proc_prefix = paths.key(paths.PROCESSED)
    proc_key    = proc_prefix + "normalized_sample.parquet"
    raw_key     = paths.raw_source_keys["train_source1"]

    logger.info("Loading parquet: s3://%s/%s", bucket, proc_key)
    df_norm = _parquet(s3c, bucket, proc_key)
    logger.info("  -> %d rows x %d cols", len(df_norm), len(df_norm.columns))

    logger.info("Loading raw head: s3://%s/%s (10K)", bucket, raw_key)
    df_raw = _tsv_head(s3c, bucket, raw_key, 10_000)

    logger.info("Running checks…")
    checks = {
        "1_inventory":            c1_inventory(s3c, bucket, proc_prefix),
        "2_schema":               c2_schema(df_norm),
        "3_originals_preserved":  c3_originals(df_norm, df_raw),
        "4_null_safety":          c4_null_safety(df_norm),
        "5_name_normalization":   c5_name_norm(df_norm),
        "6_non_ascii":            c6_non_ascii(df_norm),
        "7_row_count":            c7_rowcount(df_norm, df_raw),
        "8_duplicates":           c8_duplicates(df_norm),
    }
    audit = {
        "generated_at":   datetime.now(tz=timezone.utc).isoformat(),
        "account_prefix": paths.account_prefix,
        "parquet_uri":    f"s3://{bucket}/{proc_key}",
        "checks":         checks,
        "overall_pass":   all(v.get("pass", False) for v in checks.values()),
    }
    return audit, s3c, bucket


def _print(audit):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  PROCESSED DATA AUDIT — Task 0")
    print(f"  {audit['parquet_uri']}")
    print(sep)
    for cid, res in audit["checks"].items():
        st  = "✅ PASS" if res.get("pass") else "❌ FAIL"
        lbl = cid.replace("_"," ").title()
        print(f"  {st}  {lbl}")
        if cid == "1_inventory":
            for f in res.get("files",[]):
                print(f"           {f['key'].split('/')[-1]:42s}  {f['size_bytes']:>10,} B  {f['last_modified'][:19]}")
        elif cid == "2_schema":
            if res.get("missing_required"):
                print(f"           ❌ Missing: {res['missing_required']}")
            print(f"           source_col: {res['source_col_present']}")
            print(f"           NOTE: {res.get('note','')[:100]}")
        elif cid == "3_originals_preserved":
            print(f"           rows_checked={res.get('rows_checked',0)}  mismatches={len(res.get('mismatches',[]))}")
            for ex in res.get("spot_check_examples",[])[:3]:
                print(f"           {str(ex.get('entity_id',''))}: orig_name='{str(ex.get('original_name',''))[:35]}'")
        elif cid == "4_null_safety":
            print(f"           address_missing: {res.get('address_missing_count',0)}/{res.get('total_rows',0)} = {res.get('address_missing_pct',0):.2f}%")
            print(f"           flag=True but norm_addr not null: {res.get('flag_true_but_norm_addr_not_null',0)}")
            print(f"           flag=False but norm_addr null:    {res.get('flag_false_but_norm_addr_null',0)}")
        elif cid == "5_name_normalization":
            for ex in res.get("examples",[])[:4]:
                sf = "✓" if ex.get("legal_suffix_stripped") else "✗"
                tk = "✓" if ex.get("token_sort_correct") else "✗"
                print(f"           [{sf}suf {tk}tok]  {ex['original_name'][:32]:32s} -> {ex['normalized_name'][:32]}")
        elif cid == "6_non_ascii":
            print(f"           non-ASCII rows: {res.get('non_ascii_rows_found',0)}  mangled: {res.get('mangled_rows',0)}")
            print(f"           countries: {res.get('unique_countries')}")
            if res.get("country_narrow_warning"):
                print(f"           ⚠ {res.get('note','')}")
        elif cid == "7_row_count":
            print(f"           norm={res.get('norm_rows',0)}  raw={res.get('raw_rows',0)}  delta={res.get('delta',0)}")
        elif cid == "8_duplicates":
            print(f"           duplicate entity_ids: {res.get('duplicate_entity_ids',0)}")
    overall = "✅ ALL CHECKS PASSED" if audit["overall_pass"] else "❌ SOME CHECKS FAILED"
    print(f"\n  OVERALL: {overall}")
    print(f"{sep}\n")


def main():
    import argparse
    p = argparse.ArgumentParser()
    add_account_prefix_arg(p)
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--aws-profile", default=None, dest="aws_profile")
    args = p.parse_args()

    cfg = resolve_config_from_args(args, config_path=args.config)
    audit, s3c, bucket = run_audit(cfg)

    report_key = cfg.paths.key(cfg.paths.REPORTS) + "processed_data_audit.json"
    body = json.dumps(audit, indent=2, default=str).encode()
    s3c.put_object(Bucket=bucket, Key=report_key, Body=body, ContentType="application/json")
    logger.info("Audit uploaded -> s3://%s/%s", bucket, report_key)
    _print(audit)


if __name__ == "__main__":
    main()
