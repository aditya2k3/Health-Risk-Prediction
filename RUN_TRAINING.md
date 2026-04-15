# Download, Train, and Export

Run this once to download datasets, train machine learning models, and export performance indexes.

## Command

```bash
python download_train_export.py
```

## What this creates

- `datasets/diabetes.csv`
- `datasets/heart.csv`
- `datasets/kidney.csv`
- `models/diabetes_model.joblib`
- `models/heart_model.joblib`
- `models/kidney_model.joblib`
- `performance/performance_index.csv`
- `performance/performance_index.json`
- `performance/model_index.json`

## Performance index fields

- `disease`
- `model`
- `roc_auc`
- `pr_auc`
- `f1`
- `precision`
- `recall`
- `threshold`
- `tp`, `tn`, `fp`, `fn`
- `n_test`
- `dataset_file`
- `model_file`
