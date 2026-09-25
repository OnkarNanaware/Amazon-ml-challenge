# Amazon ML Challenge — Entity Resolution Pipeline

## Quick Start

```bash
# 1. Install the package in editable mode (makes src.entity_resolution.* importable everywhere)
pip install -e ".[dev]"

# 2. Run profiling on a 10K sample (default)
python src/entity_resolution/profiling.py --aws-profile amazon-ml-account1

# 3. Run profiling on a 100K sample, custom report name
python src/entity_resolution/profiling.py \
    --aws-profile amazon-ml-account1 \
    --sample-size 100000 \
    --report-suffix _100k

# 4. Run unit tests (no AWS credentials required)
pytest tests/ -v
```

## Multi-Account Usage (Account 2 / Account 3)

All S3 output paths are derived from a single `account_prefix` — **no source file changes needed**.
Teammates can run the entire pipeline against their own S3 prefix via:

```bash
# Option 1 — environment variable (recommended for CI/SageMaker)
ACCOUNT_PREFIX=account2 python src/entity_resolution/profiling.py --aws-profile amazon-ml-account2

# Option 2 — CLI flag (handy for local runs)
python src/entity_resolution/profiling.py --account-prefix account3 --aws-profile amazon-ml-account3
```

**RAW and SHARED are NOT account-scoped** — everyone reads from the same
`s3://amzn-s3-ml-c/raw/` prefix. Only processed outputs / reports / models are
written to the account-scoped prefix.

| Path | Account-scoped? | URI pattern |
|---|---|---|
| RAW | ❌ shared | `s3://amzn-s3-ml-c/raw/` |
| SHARED | ❌ shared | `s3://amzn-s3-ml-c/shared/` |
| PROCESSED | ✅ yes | `s3://amzn-s3-ml-c/<prefix>/processed/` |
| CANDIDATES | ✅ yes | `s3://amzn-s3-ml-c/<prefix>/candidates/` |
| FEATURES | ✅ yes | `s3://amzn-s3-ml-c/<prefix>/features/` |
| MODELS | ✅ yes | `s3://amzn-s3-ml-c/<prefix>/models/` |
| REPORTS | ✅ yes | `s3://amzn-s3-ml-c/<prefix>/reports/` |

## ⚠️ Scale Policy

This dataset is **trillions of records** at full scale.

- **Antigravity / local IDE** → small-sample development only (`--sample-size` ≤ 100K)
- **Full-scale execution** → SageMaker Processing Jobs exclusively
- **Never** run `--sample-size 0` (full file) locally

## Repository Layout

```
conftest.py          ← Adds repo root to sys.path for pytest
pyproject.toml       ← Editable install config (pip install -e .)
configs/
  config.yaml        ← All pipeline hyper-params + account_prefix default
src/
  entity_resolution/
    config.py        ← SINGLE SOURCE OF TRUTH for all S3 paths (import this)
    profiling.py     ← Step 1: Data profiling
    normalization.py ← Step 2: Text normalisation
    blocking.py      ← Step 3: Blocking / candidate generation
    candidate_recall.py  ← Step 4: Blocking recall evaluation
    features.py      ← Step 5: Pairwise features
    train.py         ← Step 6: Model training
    infer.py         ← Step 7: Inference
    threshold.py     ← Step 8: Threshold selection
    error_analysis.py← Step 9: Error analysis
    validate.py      ← Step 10: Final validation
tests/               ← Pytest unit tests (synthetic data, no S3 required)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed module descriptions and SageMaker handoff plans.

## AWS Setup

```bash
aws configure --profile amazon-ml-account1
```
