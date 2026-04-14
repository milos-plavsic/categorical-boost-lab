# Categorical Boost Lab

[![CI](https://github.com/milos-plavsic/categorical-boost-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/milos-plavsic/categorical-boost-lab/actions/workflows/ci.yml)
[![Python3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

Advanced tabular benchmark where **numeric and high-cardinality categorical** features coexist. Compares:

- **Random Forest** with **One-Hot Encoding** (sparse-friendly tree splits)
- **XGBoost** with **ordinal encoding** for categoricals (standard tree boosting baseline)
- **CatBoost** with **native categorical handling** (ordered boosting, no manual explosion)

Same stratified folds and ROC-AUC for fair comparison.

## Real-world data (education sector)

Uses **`data/student-mat.csv`** from the UCI *Student Performance* dataset (mathematics, secondary school, Portugal): [UCI ML Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance). Target is **pass vs not-pass** (`G3 >= 10`) with **`G1`/`G2` dropped** so categoricals from the questionnaire drive most of the signal.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make run
make test
make api
```

## Architecture

```mermaid
flowchart TB
  DF[DataFrame: numeric + cat] --> RF[RF + OneHot + Pipeline]
  DF --> XGB[XGB + Ordinal + Pipeline]
  DF --> CB[CatBoost + cat_features]
  RF --> CV[Stratified CV]
  XGB --> CV
  CB --> CV
```

## API

- `GET /health`
- `POST /v1/compare` — JSON `{"cv_splits": 3}` (optional)

## Why recruiters care

- You are not “default `fit` on a CSV” — you show **encoding choices** matched to **model families**.
- CatBoost’s categorical treatment vs XGBoost preprocessing is a common **real interview topic**.

## License

MIT
