# Amazon ML Challenge Project Rules

## Source of truth

ARCHITECTURE.md is the authoritative architecture for this project.

Do not redesign the architecture unless explicitly requested.

## Data

Raw challenge data stays in S3.

S3 bucket:
amzn-s3-ml-c

Raw:
s3://amzn-s3-ml-c/raw/

Account 1:
s3://amzn-s3-ml-c/account1/

## Compute

Use local/interactive compute for development and debugging.

Use SageMaker CPU compute for full-dataset processing.

Do not introduce GPU requirements unless explicitly requested.

## Pipeline

1. Profiling
2. Normalization
3. Parquet conversion
4. Blocking
5. Candidate recall
6. Feature engineering
7. LightGBM training
8. Entity-level validation
9. Threshold selection
10. Test inference
11. Candidate output
12. Submission validation

## Blocking

Implement:

A. Exact normalized name
B. Exact normalized address
C. Name + country
D. Address + country
E. Name 3-gram TF-IDF
F. Address 3-gram TF-IDF
G. Postal/numeric tokens
H. Rare-token inverted index

Union and deduplicate candidates.

Never create an O(N^2) Cartesian comparison.

## Model

Initial matcher:
LightGBM binary classifier.

Do not introduce transformer matching unless explicitly requested.

## Validation

Use entity-level splits.

Primary metric:
F0.5.

Threshold selection must use validation data only.

## Reproducibility

Centralize parameters in:

configs/config.yaml

Do not hardcode repeated paths or model parameters.

## Security

Never commit AWS credentials.

Never put access keys or secret keys in source code.

Use IAM roles, AWS CLI profiles, or environment-based credentials.

## Data handling

Do not commit raw datasets.

Do not unnecessarily load the entire multi-million-row dataset into memory.

Prefer Polars, DuckDB, sparse matrices, and streaming/chunked processing.
