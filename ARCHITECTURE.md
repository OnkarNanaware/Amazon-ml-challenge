# Architecture — Amazon ML Challenge Entity Resolution Pipeline

## Overview

This repository contains the **canonical, production-quality entity-resolution pipeline** for the Amazon ML Challenge.
All modules are independently importable so that teammates can branch off, experiment, and swap components without touching the rest of the codebase.

> **Scale policy**: This dataset is trillions of records at full scale.
> Antigravity / local IDE is **only** for code-writing and small-sample validation (≤ 100K rows).
> Full-scale execution of any pipeline stage **must** run on **SageMaker Processing Jobs**.
> Never set `--sample-size 0` locally.

---

## Multi-Account Design

All S3 output paths derive from a single `account_prefix` (default `"account1"`).
Teammates run the **same code unchanged** against their own prefix:

```bash
# Account 2 — blocking experiments
ACCOUNT_PREFIX=account2 python src/entity_resolution/normalization.py ...

# Account 3 — feature/model experiments
python src/entity_resolution/train.py --account-prefix account3 ...
```

`RAW` and `SHARED` are **not** account-scoped — everyone reads the same raw data.
See [`src/entity_resolution/config.py`](src/entity_resolution/config.py) for the single source of truth for all S3 paths.

---

## Module Map

```
src/entity_resolution/
├── config.py           ← S3 path config — import this, never hardcode paths
├── profiling.py        ← Step 1: Data profiling & quality report
├── normalization.py    ← Step 2: Text normalisation (THIS DOC)
├── blocking.py         ← Step 3: Candidate generation / blocking
├── candidate_recall.py ← Step 4: Recall evaluation for blocking
├── features.py         ← Step 5: Pairwise feature engineering
├── train.py            ← Step 6: Model training (LightGBM)
├── infer.py            ← Step 7: Inference / prediction
├── threshold.py        ← Step 8: Threshold selection (F0.5 optimisation)
├── error_analysis.py   ← Step 9: Error analysis
└── validate.py         ← Step 10: Final validation
```

---

## Step 1 — `profiling.py`

### Purpose
Profile the raw TSV files **before** any transformation to understand data characteristics that drive subsequent design decisions.

### Metrics
Row counts · column dtypes · missing values · entity_id uniqueness · duplicate rows · duplicate (name, address, country) combos · name/address length distributions · open-set country distribution (top-50) · ground-truth pair counts and cross-source distribution.

### How to run
```bash
# 10K sample (default)
python src/entity_resolution/profiling.py --aws-profile amazon-ml-account1

# 100K sample, named report
python src/entity_resolution/profiling.py \
    --aws-profile amazon-ml-account1 \
    --sample-size 100000 --report-suffix _100k
```

### Output
- `s3://amzn-s3-ml-c/account1/reports/profiling_report.json`
- `s3://amzn-s3-ml-c/account1/reports/profiling_report_100k.json`

### Unit tests
```bash
pytest tests/test_profiling.py -v    # 29 tests, no AWS needed
```

---

## Step 2 — `normalization.py`

### Purpose
Normalize raw entity text fields (business name, address, country) before blocking and feature engineering.

### Design decisions (locked in after 100K profiling)

| Decision | Rationale |
|---|---|
| Operate on `business_name` / `business_address` **independently** | Concatenation loses field-level signal needed for separate blocking keys |
| Preserve `original_name` / `original_address` columns | Error analysis and feature engineering may need the raw form |
| `address_missing = True` flag for ~3.4% null-address rows | Routes them to a name-only blocking path in Step 3 |
| Country is **open-set** — only trim + title-case | No US/India hardcoding; test set likely contains other countries (e.g. France) |
| Token-sorted `token_sorted_name` derived field | Handles word-order variation ("Smith Jones Inc" vs "Jones Smith") for blocking |

### Business name pipeline (in order)
1. **NFKC** Unicode normalization (handles accented chars, full-width, ligatures)
2. **Lowercase**
3. **Whitespace collapse** (tabs, double-spaces → single space)
4. **Punctuation** — `&` → `and` first; then remove abbreviation dots (A.B.C.→ABC); hyphens/slashes → space
5. **Abbreviation expansion** — `intl`→`international`, `tech`→`technology`, etc.
6. **Legal suffix strip** — trailing `Inc / LLC / Ltd / Corp / GmbH / SA / ...` removed
7. **Final whitespace collapse**
8. **Token-sort** — tokens sorted alphabetically → `token_sorted_name`

### Address pipeline (in order)
1. **NFKC**
2. **Lowercase**
3. **Punctuation** — commas/periods removed; other punct → space
4. **Whitespace collapse**
5. **Address abbreviation expansion** — `St`→`street`, `Ave`→`avenue`, `Blvd`→`boulevard`, `Ste`→`suite`, `N`→`north`, etc.
6. **Final whitespace collapse**
7. Numeric tokens (street numbers, zip/postal codes) **preserved as-is**

### Country pipeline
1. NFKC → title-case → whitespace collapse. Nothing else. No country list.

### Public API

```python
from src.entity_resolution.normalization import normalize_batch

df_normalized = normalize_batch(df_raw)
# New columns: original_name, original_address, original_country,
#              normalized_name, normalized_address, normalized_country,
#              token_sorted_name, address_missing
```

`normalize_batch(df)` is a **pure DataFrame → DataFrame function** with:
- No hardcoded S3 paths
- No hardcoded account prefix
- No I/O side effects

### SageMaker handoff plan

```python
# sagemaker_jobs/normalize_job.py  (future file — not yet in repo)
import pandas as pd
from src.entity_resolution.normalization import normalize_batch
from src.entity_resolution.config import load_pipeline_config

cfg   = load_pipeline_config()  # reads ACCOUNT_PREFIX from env
paths = cfg.paths

# SageMaker mounts input data at /opt/ml/processing/input/
df_raw  = pd.read_csv("/opt/ml/processing/input/train_source1.tsv", sep="\t")
df_norm = normalize_batch(df_raw)                    # same function, unchanged
df_norm.to_parquet("/opt/ml/processing/output/normalized.parquet")
```

The normalize_batch function is imported **unchanged** — only the I/O wrapper changes.

### How to run (small sample)
```bash
python src/entity_resolution/normalization.py \
    --aws-profile amazon-ml-account1 \
    --source train_source1 \
    --sample-size 10000
```

Output: `s3://amzn-s3-ml-c/account1/processed/normalized_sample.parquet`

### Unit tests
```bash
pytest tests/test_normalization.py -v    # 57 tests, no AWS needed
```
Covers: legal suffix stripping · `&`/`and` handling · missing-address rows ·
non-ASCII/accented characters (`Café`, `Société`) · mixed country address formats ·
open-set country (France, Germany) · token-sort commutativity.

---

## Step 3 — `blocking.py`

### Purpose
Generate candidate pairs from the normalized entity records.
A pair missed here **cannot be recovered** by the classifier — blocking recall is the pipeline's primary quality gate.

### Strategy: three complementary passes

| Pass | Method | Strengths | Weakness |
|---|---|---|---|
| 1 | TF-IDF character n-gram cosine similarity on `normalized_name` | Handles abbreviations, transpositions, partial matches | Compute-intensive at scale |
| 2 | Token-sort prefix blocking on `token_sorted_name` | Free; catches word-order variants | Only prefix overlap |
| 3 | Address numeric-prefix blocking on `normalized_address` | Adds recall for name-ambiguous entities | Only for rows with address present |

Results are merged, direction-canonicalized, deduplicated (max score kept, methods union-joined), and capped at `max_candidates_per_entity`.

### Null-safety
Rows with `address_missing=True` never enter Pass 3.  The TF-IDF and token-sort passes are name-only and fully safe for all rows.

### Public API

```python
from src.entity_resolution.blocking import (
    run_blocking_pair,     # full 3-pass orchestrator
    merge_candidates,      # dedup + cap utility
    BlockingConfig,        # hyper-params from config.yaml
)

cfg        = BlockingConfig.from_config_dict(pipeline_cfg._raw)
candidates = run_blocking_pair(df_left, df_right, cfg, left_tag="S1", right_tag="S2")
# Returns: pd.DataFrame [entity_id_left, entity_id_right, blocking_score, blocking_method]
```

### Config knobs (`config.yaml → blocking`)

| Key | Default | Meaning |
|---|---|---|
| `tfidf_top_k` | 50 | TF-IDF candidates per entity |
| `ngram_size` | 3 | Character n-gram max size |
| `tfidf_min_sim` | 0.05 | Minimum cosine score to keep |
| `prefix_len` | 5 | Token-sort shard prefix length |
| `addr_prefix_len` | 8 | Address shard prefix length |
| `max_candidates_per_entity` | 100 | Global cap after merge |

### How to run (small sample)
```bash
python src/entity_resolution/blocking.py \
    --aws-profile amazon-ml-account1 \
    --left s1 --right s2
```
Output: `s3://amzn-s3-ml-c/account1/candidates/candidates_S1_S2.parquet`

### SageMaker handoff plan
The pure functions (`generate_tfidf_candidates`, `generate_token_sort_candidates`, `generate_address_prefix_candidates`, `merge_candidates`, `run_blocking_pair`) contain no I/O. Import them unchanged into a SageMaker Processing Job; only wrap with S3 read/write.

### Unit tests
```bash
pytest tests/test_blocking.py -v    # 43 tests covering all three passes + merge + recall
```

---

## Step 4 — `candidate_recall.py`

### Purpose
Measure the recall of the blocking stage against ground-truth matched pairs.
This is Account 2's primary iteration metric when experimenting with blocking strategies.

### Key metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Recall** | found_GT / total_GT | Fraction of real matches that made it into candidates |
| **Upper-bound precision** | found_GT / total_candidates | Rough precision bound (not the classifier precision) |
| **Avg candidates / entity** | mean(cands per left entity) | Blocking efficiency proxy |

### Public API

```python
from src.entity_resolution.candidate_recall import compute_recall, coverage_by_method

result = compute_recall(candidates_df, ground_truth_df)
# RecallResult: recall, precision, found_pairs, missing_pairs, missing_df

method_df = coverage_by_method(candidates_df, ground_truth_df)
# Per-method breakdown of GT pairs found
```

Both functions are **pure** (no S3) and **direction-agnostic** — `(A, B)` and `(B, A)` are the same pair.

### How to run
```bash
python src/entity_resolution/candidate_recall.py \
    --aws-profile amazon-ml-account1 \
    --candidates-key account1/candidates/candidates_S1_S2.parquet \
    --gt-sample 100000
```

---

## Branching & Collaboration

| Account | Area | Branch convention |
|---|---|---|
| Account 1 (canonical) | Production pipeline | `main` |
| Account 2 | Blocking experiments | `feat/blocking-*` |
| Account 3 | Feature / model experiments | `feat/features-*` or `feat/model-*` |

**Git hygiene**:
- **Never commit** TSV / Parquet / candidate table files.
- Only commit: code, configs, tests, notebooks, small samples (≤ 1,000 rows), experiment summaries.
- Large-file types already covered by `.gitignore`.

